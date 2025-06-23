# todo upgrade to keras 2.0

from tensorflow.keras.layers import Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Reshape, Permute, Layer
from tensorflow.keras.layers import Input, Convolution2D, MaxPooling2D, UpSampling2D, Reshape, Dropout, add, concatenate
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, UpSampling2D, ZeroPadding2D, Conv2D, AveragePooling2D
from tensorflow.keras import backend as K
from tensorflow.keras import initializers
import tensorflow as tf

# upsample_flag: 一個布林值 (0 或 1)，決定第二階段是否進行 2 倍的超解析度上採樣
# insert_x, insert_y: 裁剪第一階段的輸出 output1 由於輸入影像在送入模型前會先 padding，這裡將 output1 裁剪回原始的有效區域大小
# conv_block_num: U-Net 中編碼器和解碼器的下採樣/上採樣區塊數量 (預設 4)
# conv_num: 每個卷積區塊中包含的卷積層數量 (預設 3)
def Unet(input_shape,upsample_flag,insert_x,insert_y,conv_block_num=4,conv_num=3):
    
    inputs = Input(input_shape)
    _,h,w,_ = inputs.shape

    pool = inputs
    
    #Encoder
    conv_list = []
    for n in range(conv_block_num):
        channels = 2 ** (n+5)
        pool, conv = conv_block(pool, conv_num, channels)
        conv_list.append(conv)

    # 中間層 Bottleneck
    mid = Conv2D(channels*2, kernel_size=3, activation='relu', padding='same')(pool)
    mid = Conv2D(channels, kernel_size=3, activation='relu', padding='same')(mid)
    
    #Decoder
    concat_block_num = conv_block_num
    init_channels = channels
    conv = mid
    for n in range(concat_block_num):
        channels = init_channels // (2 ** n)
        conv = concat_block(conv, conv_list[-(n+1)], conv_num, channels)

    #output denoised img
    output1 = Conv2D(1, kernel_size=3, activation='relu', padding='same')(conv)

    # 第二階段 (Deconvolution / Super-resolution) 
    #Encoder
    pool = output1
    conv_list = []
    for n in range(conv_block_num):
        channels = 2 ** (n+5)
        pool, conv = conv_block(pool, conv_num, channels)
        conv_list.append(conv)

    mid = Conv2D(channels*2, kernel_size=3, activation='relu', padding='same')(pool)
    mid = Conv2D(channels, kernel_size=3, activation='relu', padding='same')(mid)
    
    #Decoder
    concat_block_num = conv_block_num
    init_channels = channels
    conv = mid
    for n in range(concat_block_num):
        channels = init_channels // (2 ** n)
        conv = concat_block(conv, conv_list[-(n+1)], conv_num, channels)

    #output deconved img
    if upsample_flag:
        conv = UpSampling2D(size=(2, 2))(conv)
    conv = Conv2D(128, kernel_size=3, activation='relu', padding='same')(conv)
    conv = Conv2D(128, kernel_size=3, activation='relu', padding='same')(conv)
    output2 = Conv2D(1, kernel_size=3, activation='relu', padding='same')(conv)
    
    model = Model(inputs=inputs, outputs=[output1[:,insert_x:h-insert_x,insert_y:w-insert_y,:],output2])
    return model

def conv_block(input_layer, conv_num, channels):
    conv = input_layer
    for _ in range(conv_num):
        conv = Conv2D(channels, kernel_size=3, activation='relu', padding='same')(conv)
    #conv = conv + input_layer
    pool = MaxPooling2D(pool_size=(2, 2))(conv)
    return pool, conv

def concat_block(concat1, concat2, conv_num, channels):
    # 將上採樣後的結果與來自編碼器的跳躍連接 concat2 沿著通道軸 (axis=3) 拼接起來。
    up = concatenate([UpSampling2D(size=(2, 2))(concat1), concat2], axis=3)
    conv = Conv2D(channels, kernel_size=3, activation='relu', padding='same')(up)
    for _ in range(conv_num-1):
        conv = Conv2D(channels//2, kernel_size=3, activation='relu', padding='same')(conv)
    return conv