import argparse
from tensorflow.keras import optimizers
import numpy as np
import datetime
import glob
import os
from models import twostage_Unet
from utils.data_loader import DataLoader
from utils.loss import create_psf_loss,cal_psf_2d,psf_estimator_2d
from utils.utils import prctile_norm
import tensorflow as tf
import imageio
import cv2
from tensorflow.keras import backend as K
from scipy.interpolate import interp1d

parser = argparse.ArgumentParser()
# models 模型架構參數
parser.add_argument("--conv_block_num", type=int, default=4) # U-Net 每個階層的卷積區塊數量
parser.add_argument("--conv_num", type=int, default=3) # 每個卷積區塊內的卷積層數量
# upsample_flag=1: 反卷積 + 超解析 (去模糊 + 提升解析度) upsample_flag=0: 只有反卷積 (去模糊，解析度不變)
parser.add_argument("--upsample_flag", type=int, default=1) # 是否啟用上採樣功能

# training settings 訓練環境設定
parser.add_argument("--gpu_id", type=int, default=0)
parser.add_argument("--gpu_memory_fraction", type=float, default=0.9) # GPU 記憶體使用率
parser.add_argument("--mixed_precision_training", type=int, default=1) # 是否用混合精度訓練
parser.add_argument("--iterations", type=int, default=50000) # 訓練迭代次數
parser.add_argument("--test_interval", type=int, default=1000) # 每多少迭代進行一次測試
parser.add_argument("--valid_interval", type=int, default=1000) # 每多少迭代進行一次驗證
parser.add_argument("--load_init_model_iter", type=int, default=0) # initial loading weights

# lr decay
parser.add_argument("--start_lr", type=float, default=5e-05) # 初始學習率 (5×10⁻⁵)
parser.add_argument("--lr_decay_factor", type=float, default=0.5) # 學習率衰減倍數，每 10000 次迭代會乘以此值

# about data
parser.add_argument("--otf_or_psf_path", type=str)
parser.add_argument("--psf_src_mode", type=int) # 1:psf in .tif format; 2:otf in .mrc format
parser.add_argument("--dxypsf", type=float, default=0.0313) # dxy of simulated PSF，PSF 的像素大小 (微米)
parser.add_argument("--dx", type=float, default=0.0313) # dx of input image，輸入影像的像素大小 (微米)
parser.add_argument("--dy", type=float, default=0.0313) # dy of input image，輸入影像的像素大小 (微米)

parser.add_argument("--data_dir", type=str)
parser.add_argument("--folder", type=str)
parser.add_argument("--test_images_path", type=str)
parser.add_argument("--save_weights_dir", type=str, default='../your_saved_models/')
parser.add_argument("--save_weights_suffix", type=str, default="_Hess0.02")

parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--input_y", type=int, default=128)
parser.add_argument("--input_x", type=int, default=128)
parser.add_argument("--insert_xy", type=int, default=16) # 邊緣填充大小 padding，避免卷積運算在邊緣產生問題
parser.add_argument("--input_y_test", type=int, default=512)
parser.add_argument("--input_x_test", type=int, default=512)

parser.add_argument("--valid_num", type=int, default=3) # 驗證用的影像數量

# about loss functions
parser.add_argument("--mse_flag", type=int, default=0) # 0 for mae, 1 for mse
parser.add_argument("--denoise_loss_weight", type=float, default=0.5) # 去噪損失的權重 (0.5 表示去噪和去卷積各佔 50%)
parser.add_argument("--l1_rate", type=float, default=0) # L1 正則化的權重
parser.add_argument("--TV_rate", type=float, default=0) # TV 正則化的權重 (平滑化)
parser.add_argument("--Hess_rate", type=float, default=0.02) # Hessian 正則化的權重 (邊緣保持)
# 只有 L1: 影像稀疏，對比強烈，可能丟失細節 只有 TV: 影像平滑，但可能過度模糊邊緣 只有 Hessian: 平滑與邊緣保持的良好平衡

args = parser.parse_args()

gpu_id = str(args.gpu_id)
gpu_memory_fraction = args.gpu_memory_fraction
mixed_precision_training = str(args.mixed_precision_training)
iterations = args.iterations
test_interval = args.test_interval
valid_interval = args.valid_interval
load_init_model_iter = args.load_init_model_iter

data_dir = args.data_dir
folder = args.folder
save_weights_dir = args.save_weights_dir
test_images_path = args.test_images_path
valid_data_dir = data_dir

dxypsf = args.dxypsf
dx = args.dx
dy = args.dy
batch_size = args.batch_size
input_y = args.input_y
input_x = args.input_x
input_y_test = args.input_y_test
input_x_test = args.input_x_test
insert_xy = args.insert_xy

start_lr = args.start_lr
lr_decay_factor = args.lr_decay_factor

conv_block_num = args.conv_block_num
conv_num = args.conv_num
upsample_flag = args.upsample_flag

mse_flag = args.mse_flag
psf_src_mode = args.psf_src_mode
valid_num = args.valid_num
l1_rate = args.l1_rate
TV_rate = args.TV_rate
Hess_rate = args.Hess_rate
denoise_loss_weight = args.denoise_loss_weight

# 設定環境跟路徑
os.environ["TF_ENABLE_AUTO_MIXED_PRECISION"] = mixed_precision_training
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))
       
cur_data_loader = DataLoader

# define and make paths
save_weights_name = folder+'_twostage_Unet'+args.save_weights_suffix
save_weights_path = save_weights_dir + '/' + save_weights_name + '/'
train_images_path = data_dir+folder+'/input'
valid_images_path = valid_data_dir+folder+'/input'
train_gt_path = data_dir+folder+'/gt'
valid_gt_path = valid_data_dir+folder+'/gt'
valid_path = save_weights_path + 'TrainSampled/'
test_path = save_weights_path + 'TestSampled/'

if not os.path.exists(save_weights_dir):
    os.mkdir(save_weights_dir)
if not os.path.exists(save_weights_path):
    os.mkdir(save_weights_path)
if not os.path.exists(valid_path):
    os.mkdir(valid_path)
if not os.path.exists(test_path):
    os.mkdir(test_path)
    
with open(save_weights_path+"config.txt","a") as f:
    f.write('\n\n')
    f.write(str(args))

# determine test input
source_path = glob.glob(test_images_path)
test_num = len(source_path)
img_list_p=[]
insert_xy_test = insert_xy
for i in range(test_num):
    images_path = source_path[i]
    img = np.array(imageio.mimread(images_path)).astype(np.float32)
    img = np.squeeze(np.sum(img,axis=0)) 
    img = prctile_norm(img)
    # 加 padding
    input_y_test,input_x_test = img.shape
    insert_x = np.zeros([insert_xy_test,input_x_test])
    insert_y = np.zeros([input_y_test+2*insert_xy_test,insert_xy_test])
    img = np.concatenate((insert_x,img,insert_x),axis=0)
    img = np.concatenate((insert_y,img,insert_y),axis=1)
    img_list_p.append(img)
    
    img = np.uint16(prctile_norm(img[insert_xy_test:insert_xy_test+input_y_test,insert_xy_test:insert_xy_test+input_x_test])*65535)
    imageio.imwrite(test_path + 'input' +str(i) + '.tif', img)
    
# --------------------------------------------------------------------------------
#                           select models and optimizer
# --------------------------------------------------------------------------------
modelFN = twostage_Unet.Unet
optimizer_g = optimizers.Adam(learning_rate=start_lr, beta_1=0.9, beta_2=0.999)

# --------------------------------------------------------------------------------
#                          calculate and process otf & psf
# --------------------------------------------------------------------------------

# 根據psf_src_mode選擇使用psf或otf，psf_src_mode=1表示使用psf，2表示使用otf
if psf_src_mode==1:
    print('using psf...')
    psf_g = np.float32(imageio.imread(args.otf_or_psf_path))
    psf_width,psf_height = psf_g.shape
    half_psf_width = psf_width//2
    
    # get the desired dxy
    if psf_width%2==1:
        # dxypsf = PSF的體素大小（例如：0.1微米） voxel
        # dx = 目標圖像的體素大小（例如：0.05微米） voxel
        # sr_ratio = 縮放比例（例如：0.1/0.05 = 2，表示要放大2倍）
        sr_ratio = dxypsf/dx
        sr_x = round(psf_width*sr_ratio) # 計算縮放後的寬度
        if sr_x%2==0: # 如果縮放後的寬度是偶數
            if sr_x>psf_width*sr_ratio: # 四捨五入後比原本大，就減1
                sr_x = sr_x - 1
            else: # 四捨五入後比原本小，就加1
                sr_x = sr_x + 1
        sr_y = round(psf_height*sr_ratio)
        if sr_y%2==0:
            if sr_y>psf_height*sr_ratio:
                sr_y = sr_y - 1
            else:
                sr_y = sr_y + 1
        psf_g = cv2.resize(psf_g,(sr_x,sr_y))
    else: # 如果PSF的寬度是偶數，要經過很麻煩的操作，分中心點的左右邊處理
        x = np.arange((half_psf_width+1) * dxypsf, (psf_width+0.1) * dxypsf, dxypsf)
        xi = np.arange((half_psf_width+1) * dxypsf, (psf_width+0.1) * dxypsf, dx)
        if xi[-1]>x[-1]:
            xi = xi[0:-1]
        PSF1 = np.zeros((len(xi),psf_height))
        for i in range(psf_height):
            curCol = psf_g[half_psf_width:psf_width,i]
            interp = interp1d(x, curCol, 'slinear')
            PSF1[:,i] = interp(xi)
        x2 = np.zeros(len(x))
        xi2 = np.zeros(len(xi))
        for n in range(len(x)):
            x2[len(x)-n-1]=x[0]-dxypsf*n
        for n in range(len(xi)):
            xi2[len(xi)-n-1]=xi[0]-dx*n
        PSF2 = np.zeros((len(xi2),psf_height))
        for i in range(psf_height):
            curCol = psf_g[1:half_psf_width+1+psf_width%2,i]
            interp = interp1d(x2, curCol, 'slinear')
            PSF2[:,i] = interp(xi2)
        psf_g = np.concatenate((PSF2[:-1,:],PSF1),axis=0)
        psf_g = psf_g/np.sum(psf_g)
        psf_width,psf_height = psf_g.shape
        half_psf_height = psf_height//2
        
        x = np.arange((half_psf_height+1) * dxypsf, (psf_height+0.1) * dxypsf, dxypsf)
        xi = np.arange((half_psf_height+1) * dxypsf, (psf_height+0.1) * dxypsf, dx)
        if xi[-1]>x[-1]:
            xi = xi[0:-1]
        PSF1 = np.zeros((psf_width,len(xi)))
        for i in range(psf_width):
            curCol = psf_g[i,half_psf_height:psf_height]
            interp = interp1d(x, curCol, 'slinear')
            PSF1[i,:] = interp(xi)
        x2 = np.zeros(len(x))
        xi2 = np.zeros(len(xi))
        for n in range(len(x2)):
            x2[len(x2)-n-1]=x[0]-dxypsf*n
        for n in range(len(xi2)):
            xi2[len(xi2)-n-1]=xi[0]-dx*n
        PSF2 = np.zeros((psf_width,len(xi2)))
        for i in range(psf_width):
            curCol = psf_g[i,1:half_psf_height+1+psf_height%2]
            interp = interp1d(x2, curCol, 'slinear')
            PSF2[i,:] = interp(xi2)
        psf_g = np.concatenate((PSF2[:,:-1],PSF1),axis=1)
        
    otf_g = np.fft.fftshift(np.fft.fftn(psf_g))
    otf_g = np.abs(otf_g)
    otf_g = cv2.resize(otf_g,(input_x*(1+upsample_flag),input_y*(1+upsample_flag)))
    otf_g = otf_g/np.sum(otf_g)
else:
    print('using psf transformed from otf...')
    otf_path = args.otf_or_psf_path
    if upsample_flag:
        dkx = 1/(input_y*2*dx)
        dky = 1/(input_x*2*dy)
        psf_g, otf_g = cal_psf_2d(otf_path, input_x*2, input_y*2, dky, dkx)
    else:
        dkx = 1/(input_y*dx)
        dky = 1/(input_x*dy)
        psf_g, otf_g = cal_psf_2d(otf_path, input_x, input_y, dky, dkx)       
psf_width = psf_g.shape[0]
psf_height = psf_g.shape[1]

# crop PSF for faster computation
# 裁減PSF以加快後續的卷積運算，會根據PSF的大小和sigma值來決定裁減的大小
sigma_y, sigma_x = psf_estimator_2d(psf_g) # psf_estimator_2d 函數會估計PSF的標準差
ksize = int(sigma_y * 4) # 計算一個核心尺寸 距離PSF中心4個標準差以外的區域，亮度值已經非常接近於 0，對模糊的貢獻很小
halfx = psf_width // 2
halfy = psf_height // 2
if ksize<=halfx:
    # 如果計算出來的核心尺寸小於等於PSF中心點，則裁減PSF
    # 從原始 psf_g 中提取出一個以中心點為中心、邊長為 2*ksize+1 的正方形區域
    psf_g = psf_g[halfx-ksize:halfx+ksize+1, halfy-ksize:halfy+ksize+1]
    psf_g = np.reshape(psf_g,(2*ksize+1,2*ksize+1,1,1)).astype(np.float32) # 將裁減後的 PSF 轉換為四維張量 (H, W, 輸入通道數, 輸出通道數)
else:
    psf_g = np.reshape(psf_g,(psf_width,psf_height,1,1)).astype(np.float32)
psf_g = psf_g/np.sum(psf_g) # 將 PSF 卷積核裡的所有數值除以它們的總和，卷積核中所有元素的總和變成了 1
    
# save
# 先正規化PSF，轉成uint16，儲存成.tif文件
psf_g_tosave = np.uint16(65535*prctile_norm(np.squeeze(psf_g)))
imageio.imwrite(save_weights_path+'psf.tif',psf_g_tosave)
otf_g_tosave = np.uint16(65535*prctile_norm(np.squeeze(np.abs(otf_g))))
imageio.imwrite(save_weights_path+'otf.tif',otf_g_tosave)

# --------------------------------------------------------------------------------
#                                 compile model
# --------------------------------------------------------------------------------

# g 用於訓練的模型，p 用於測試的模型
# modelFN 就是 twostage_Unet.Unet 函數
g = modelFN((input_y+2*insert_xy, input_x+2*insert_xy, 1),
            upsample_flag=upsample_flag, insert_x=insert_xy, insert_y=insert_xy)
p = modelFN((input_y_test+2*insert_xy_test, input_x_test+2*insert_xy_test, 1),
            upsample_flag=upsample_flag, insert_x=insert_xy_test, insert_y=insert_xy_test)
if os.path.exists(save_weights_path + 'weights_'+str(load_init_model_iter)+'.h5'):
    g.load_weights(save_weights_path + 'weights_'+str(load_init_model_iter)+'.h5')
    print('Loading weights successfully: ' + save_weights_path + 'weights_'+str(load_init_model_iter)+'.h5')

loss = create_psf_loss(psf_g, TV_weight=TV_rate, Hess_weight=Hess_rate,
                       laplace_weight=0,
                        l1_rate=l1_rate, mse_flag=mse_flag, 
                        upsample_flag=upsample_flag, 
                        insert_xy=insert_xy, deconv_flag=1)
# 'mean_absolute_error' (或 'mean_squared_error')。Keras 能夠識別這個字串，並將其對應到標準的 MAE/MSE 損失函數
if mse_flag:
    loss = ['mean_squared_error',loss] # 'mean_squared_error' Keras 會自動建立loss function
else:
    loss = ['mean_absolute_error',loss]

# compile models 編譯模型，將模型 g、損失函數 loss 和優化器 optimizer_g (Adam) 綁定在一起
g.compile(loss=loss, loss_weights=[denoise_loss_weight,1-denoise_loss_weight], optimizer=optimizer_g)
p.compile(loss=None, optimizer=optimizer_g)

# --------------------------------------------------------------------------------
#                               about Tensorboard
# --------------------------------------------------------------------------------
log_path = save_weights_path + 'graph'
if os.path.exists(log_path):
    for n in range(10):
        log_path = save_weights_path + 'graph'+str(n+2)
        if not os.path.exists(log_path):
            os.mkdir(log_path)
            break
else:
    os.mkdir(log_path)
writer = tf.summary.create_file_writer(log_path)

def write_log(writer, names, logs, batch_no):
    with writer.as_default():
        tf.summary.scalar(names,logs,step=batch_no)
        writer.flush()

# --------------------------------------------------------------------------------
#                                  Sample
# --------------------------------------------------------------------------------
# determine validation samples
img_list=[]
images_path = glob.glob(valid_images_path + '/*')
[input_valid, gt_valid] = cur_data_loader(images_path, valid_images_path, valid_gt_path, 
                                valid_num, norm_flag=0)
input_valid = np.reshape(input_valid, (valid_num, input_x, input_y, 1), order='F')
gt_valid = np.reshape(gt_valid, (valid_num, input_x, input_y, 1), order='F')
for i in range(valid_num):
    img = input_valid[i,:,:,0]
    img = np.uint16(prctile_norm(img)*65535)
    imageio.imwrite(valid_path + 'input_sample_' +str(i) + '.tif', img)
    
    gt = gt_valid[i,:,:,0]
    gt = np.uint16(prctile_norm(gt)*65535)
    imageio.imwrite(valid_path + 'gt_sample_' +str(i) + '.tif', gt)
insert_x = np.zeros([valid_num,insert_xy,input_x,1])
insert_y = np.zeros([valid_num,input_y+2*insert_xy,insert_xy,1])
input_valid = np.concatenate((insert_x,input_valid,insert_x),axis=1)
input_valid = np.concatenate((insert_y,input_valid,insert_y),axis=2)

def Validate(iter,img_list_val):
    
    #check on training datasets
    for i in range(valid_num):
        img = img_list_val[i,:,:,:]
        img = img.reshape((1, input_y+2*insert_xy, input_x+2*insert_xy, 1))
        output = g.predict(img)
        denoise_output = prctile_norm(np.squeeze(output[0]),3,100)
        imageio.imwrite(valid_path+str(i)+'denoised_iter'+'%05d'%iter+'.tif', np.uint16(denoise_output*65535))
        if upsample_flag:
            deconv_output = np.squeeze(output[1])[2*insert_xy:2*(insert_xy+input_y),2*insert_xy:2*(insert_xy+input_x)]
        else:
            deconv_output = np.squeeze(output[1])[insert_xy:insert_xy+input_y,insert_xy:insert_xy+input_x]
        imageio.imwrite(valid_path+str(i)+'deconved_iter'+'%05d'%iter+'.tif', np.uint16(prctile_norm(deconv_output)*65535))
                    
    curlr = K.get_value(g.optimizer.learning_rate)
    write_log(writer, 'lr', curlr, iter)
    
def Test(iter,img_list_test,load_weights):
    
    #test and show samples
    p.load_weights(load_weights)
    
    for n in range(test_num):
        #predict
        img = img_list_test[n]
        img = img.reshape((1, input_y_test+2*insert_xy_test, input_x_test+2*insert_xy_test, 1))
        output1 = np.squeeze(p.predict(img)[0])
        if upsample_flag:
            output2 = np.squeeze(p.predict(img)[1])[2*insert_xy_test:2*(insert_xy_test+input_y_test),2*insert_xy_test:2*(insert_xy_test+input_x_test)] 
        else:
            output2 = np.squeeze(p.predict(img)[1])[insert_xy_test:(insert_xy_test+input_y_test),insert_xy_test:(insert_xy_test+input_x_test)]
        
        #save img for show           
        output1 = np.uint16(prctile_norm(output1)*65535)
        imageio.imwrite(test_path+str(n)+'_denoised_iter'+str(iter)+'.tif',output1)
        output2 = np.uint16(prctile_norm(output2)*65535)
        imageio.imwrite(test_path+str(n)+'_deconved_iter'+str(iter)+'.tif',output2)
        
# --------------------------------------------------------------------------------
#                            training and validation
# --------------------------------------------------------------------------------

# train
start_time = datetime.datetime.now()
loss_record = []
loss_record2 = []
images_path = glob.glob(train_images_path + '/*')
curlr = start_lr
avg_validate_time = []

for it in range(iterations-load_init_model_iter):
    # 載入訓練資料
    [input_g, gt_g] = cur_data_loader(images_path, train_images_path, train_gt_path, 
                                    batch_size, norm_flag=0)
    input_g = np.reshape(input_g, (batch_size, input_x, input_y, 1), order='F')
    gt_g = np.reshape(gt_g, (batch_size, input_x, input_y, 1), order='F')
    insert_x = np.zeros([batch_size,insert_xy,input_x,1])
    insert_y = np.zeros([batch_size,input_y+2*insert_xy,insert_xy,1])
    # 對資料進行 padding，避免卷積運算在影像邊緣產生不好的結果
    input_g = np.concatenate((insert_x,input_g,insert_x),axis=1)
    input_g = np.concatenate((insert_y,input_g,insert_y),axis=2)
    
    # 訓練模型
    loss_generator = g.train_on_batch(x=input_g, y=[gt_g,gt_g]) # gt_g gt_g 分別是 output_1 output_2
    
    loss_record.append(loss_generator[1])
    loss_record2.append(loss_generator[2])

    elapsed_time = datetime.datetime.now() - start_time
    print("%d it: time: %s, denoise_loss = %.3e, deconv_loss = %.3e" % (it + 1 + load_init_model_iter, elapsed_time, loss_generator[1], loss_generator[2]))
    
    # 驗證、測試和儲存模型
    if (it + 1+load_init_model_iter) % valid_interval == 0 or it == 0:
        images_path = glob.glob(train_images_path + '/*')
        print('validate time:')
        valid_start = datetime.datetime.now()
        Validate(it + 1+load_init_model_iter,input_valid)
        print(datetime.datetime.now()-valid_start)
        avg_validate_time.append(datetime.datetime.now()-valid_start)
        avg_validate_time = [np.mean(avg_validate_time)]
        
    if (it + 1+load_init_model_iter) % test_interval == 0 or it == 0:
        g.save_weights(save_weights_path + 'weights_'+str(it+load_init_model_iter+1)+'.h5')
        
        write_log(writer, 'denoise_loss', np.mean(loss_record), it + 1+load_init_model_iter)
        write_log(writer, 'deconv_loss', np.mean(loss_record2), it + 1+load_init_model_iter)
        loss_record = []
        loss_record2 = []
        
        print('testing...')
        Test(it + 1+load_init_model_iter,img_list_p,save_weights_path+'weights_'+str(it + 1+load_init_model_iter)+'.h5')
             
    curlr = K.get_value(g.optimizer.learning_rate)
    if (it+load_init_model_iter+1)%10000==0:
        K.set_value(g.optimizer.learning_rate, curlr*lr_decay_factor)
