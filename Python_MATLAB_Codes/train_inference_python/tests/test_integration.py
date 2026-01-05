"""
Integration tests for end-to-end patch-based processing with TIFF files

Tests the complete workflow:
- Loading TIFF images
- Segmenting into patches
- Processing with super-resolution
- Fusing patches back
- Saving as TIFF
"""

import unittest
import numpy as np
import tifffile as tiff
import tempfile
import os
import shutil
import sys

# Add parent directory to path to import utils
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.utils import prctile_norm


class TestEndToEndProcessing2D(unittest.TestCase):
    """Integration tests for 2D processing with TIFF files"""
    
    def setUp(self):
        """Create temporary directory for test files"""
        self.test_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up temporary directory"""
        shutil.rmtree(self.test_dir)
        
    def test_save_and_load_tiff_2d(self):
        """Test saving and loading 2D TIFF images"""
        # Create a test image
        test_image = np.random.rand(256, 256).astype(np.float32)
        test_image_uint16 = (test_image * 65535).astype(np.uint16)
        
        # Save as TIFF
        test_path = os.path.join(self.test_dir, 'test_2d.tif')
        tiff.imwrite(test_path, test_image_uint16, dtype='uint16')
        
        # Load and verify
        loaded_image = tiff.imread(test_path).astype('float')
        loaded_image = loaded_image / 65535.0
        
        np.testing.assert_array_almost_equal(test_image, loaded_image, decimal=4)
        
    def test_save_and_load_tiff_3d(self):
        """Test saving and loading 3D TIFF stacks"""
        # Create a test volume
        test_volume = np.random.rand(32, 128, 128).astype(np.float32)
        test_volume_uint16 = (test_volume * 65535).astype(np.uint16)
        
        # Save as TIFF stack
        test_path = os.path.join(self.test_dir, 'test_3d.tif')
        tiff.imwrite(test_path, test_volume_uint16, dtype='uint16')
        
        # Load and verify
        loaded_volume = tiff.imread(test_path).astype('float')
        loaded_volume = loaded_volume / 65535.0
        
        self.assertEqual(loaded_volume.shape, test_volume.shape)
        np.testing.assert_array_almost_equal(test_volume, loaded_volume, decimal=4)
        
    def test_prctile_norm(self):
        """Test percentile normalization utility"""
        # Create test data with known range
        test_data = np.array([[0, 50, 100], [150, 200, 250]], dtype=np.float32)
        
        # Normalize
        normalized = prctile_norm(test_data, 0, 100)
        
        # Should be in range [0, 1]
        self.assertGreaterEqual(normalized.min(), 0.0)
        self.assertLessEqual(normalized.max(), 1.0)
        
        # Min should be close to 0, max should be close to 1
        self.assertAlmostEqual(normalized.min(), 0.0, places=5)
        self.assertAlmostEqual(normalized.max(), 1.0, places=5)


class TestPatchProcessingWorkflow(unittest.TestCase):
    """Test complete patch processing workflow"""
    
    def setUp(self):
        """Create temporary directory for test files"""
        self.test_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up temporary directory"""
        shutil.rmtree(self.test_dir)
        
    def test_2d_patch_workflow_no_superres(self):
        """Test complete 2D workflow without super-resolution"""
        import math
        
        # Create test image
        inp_x, inp_y = 128, 128
        test_image = np.random.rand(inp_x, inp_y).astype(np.float32)
        
        # Save as TIFF
        input_path = os.path.join(self.test_dir, 'input_2d.tif')
        test_image_uint16 = (test_image * 65535).astype(np.uint16)
        tiff.imwrite(input_path, test_image_uint16, dtype='uint16')
        
        # Load and normalize
        image = tiff.imread(input_path).astype('float')
        image[image < 0] = 0
        image = prctile_norm(image)
        
        # Segment parameters
        num_seg_window_x, num_seg_window_y = 2, 2
        overlap_x, overlap_y = 20, 20
        upsample_flag = 0
        
        seg_window_x = math.ceil((inp_x + (num_seg_window_x - 1) * overlap_x) / num_seg_window_x)
        seg_window_y = math.ceil((inp_y + (num_seg_window_y - 1) * overlap_y) / num_seg_window_y)
        
        rr_list = list(range(0, inp_x - seg_window_x + 1, seg_window_x - overlap_x))
        if rr_list[-1] != inp_x - seg_window_x:
            rr_list.append(inp_x - seg_window_x)
            
        cc_list = list(range(0, inp_y - seg_window_y + 1, seg_window_y - overlap_y))
        if cc_list[-1] != inp_y - seg_window_y:
            cc_list.append(inp_y - seg_window_y)
        
        # Segment
        segmented_inp = []
        for rr in rr_list:
            for cc in cc_list:
                segmented_inp.append(image[rr:rr+seg_window_x, cc:cc+seg_window_y])
        segmented_inp = np.array(segmented_inp).astype(np.float32)
        
        # Simulate processing (identity)
        dec_list = segmented_inp.copy()
        
        # Fuse
        output_dec = np.zeros((inp_x * (1 + upsample_flag), inp_y * (1 + upsample_flag)), dtype=np.float32)
        for r_ind, rr in enumerate(rr_list):
            for c_ind, cc in enumerate(cc_list):
                if rr == 0:
                    rr_min, rr_min_patch = 0, 0
                else:
                    rr_min = rr + math.ceil(overlap_x / 2)
                    rr_min_patch = math.ceil(overlap_x / 2)
                    
                if rr + seg_window_x == inp_x:
                    rr_max, rr_max_patch = inp_x, seg_window_x
                else:
                    rr_max = rr + seg_window_x - math.floor(overlap_x / 2)
                    rr_max_patch = seg_window_x - math.floor(overlap_x / 2)
                    
                if cc == 0:
                    cc_min, cc_min_patch = 0, 0
                else:
                    cc_min = cc + math.ceil(overlap_y / 2)
                    cc_min_patch = math.ceil(overlap_y / 2)
                    
                if cc + seg_window_y == inp_y:
                    cc_max, cc_max_patch = inp_y, seg_window_y
                else:
                    cc_max = cc + seg_window_y - math.floor(overlap_y / 2)
                    cc_max_patch = seg_window_y - math.floor(overlap_y / 2)
                    
                cur_patch = dec_list[r_ind * len(cc_list) + c_ind,
                                    rr_min_patch * (1 + upsample_flag):rr_max_patch * (1 + upsample_flag),
                                    cc_min_patch * (1 + upsample_flag):cc_max_patch * (1 + upsample_flag)].astype(np.float32)
                output_dec[rr_min * (1 + upsample_flag):rr_max * (1 + upsample_flag),
                          cc_min * (1 + upsample_flag):cc_max * (1 + upsample_flag)] = cur_patch
        
        # Save output
        output_path = os.path.join(self.test_dir, 'output_2d.tif')
        output_uint16 = np.uint16(1e4 * prctile_norm(output_dec, 3, 100))
        tiff.imwrite(output_path, output_uint16, dtype='uint16')
        
        # Verify output file exists and has correct shape
        self.assertTrue(os.path.exists(output_path))
        loaded_output = tiff.imread(output_path)
        self.assertEqual(loaded_output.shape, (inp_x, inp_y))
        
    def test_2d_patch_workflow_with_superres(self):
        """Test complete 2D workflow with 2x super-resolution"""
        import math
        
        # Create test image
        inp_x, inp_y = 64, 64
        test_image = np.random.rand(inp_x, inp_y).astype(np.float32)
        
        # Save as TIFF
        input_path = os.path.join(self.test_dir, 'input_2d_sr.tif')
        test_image_uint16 = (test_image * 65535).astype(np.uint16)
        tiff.imwrite(input_path, test_image_uint16, dtype='uint16')
        
        # Load and normalize
        image = tiff.imread(input_path).astype('float')
        image[image < 0] = 0
        image = prctile_norm(image)
        
        # Segment parameters
        num_seg_window_x, num_seg_window_y = 2, 2
        overlap_x, overlap_y = 8, 8
        upsample_flag = 1  # 2x super-resolution
        
        seg_window_x = math.ceil((inp_x + (num_seg_window_x - 1) * overlap_x) / num_seg_window_x)
        seg_window_y = math.ceil((inp_y + (num_seg_window_y - 1) * overlap_y) / num_seg_window_y)
        
        rr_list = list(range(0, inp_x - seg_window_x + 1, seg_window_x - overlap_x))
        if rr_list[-1] != inp_x - seg_window_x:
            rr_list.append(inp_x - seg_window_x)
            
        cc_list = list(range(0, inp_y - seg_window_y + 1, seg_window_y - overlap_y))
        if cc_list[-1] != inp_y - seg_window_y:
            cc_list.append(inp_y - seg_window_y)
        
        # Create upsampled patches
        seg_num = len(rr_list) * len(cc_list)
        dec_list = np.random.rand(seg_num,
                                  seg_window_x * (1 + upsample_flag),
                                  seg_window_y * (1 + upsample_flag)).astype(np.float32)
        
        # Fuse
        output_dec = np.zeros((inp_x * (1 + upsample_flag), inp_y * (1 + upsample_flag)), dtype=np.float32)
        for r_ind, rr in enumerate(rr_list):
            for c_ind, cc in enumerate(cc_list):
                if rr == 0:
                    rr_min, rr_min_patch = 0, 0
                else:
                    rr_min = rr + math.ceil(overlap_x / 2)
                    rr_min_patch = math.ceil(overlap_x / 2)
                    
                if rr + seg_window_x == inp_x:
                    rr_max, rr_max_patch = inp_x, seg_window_x
                else:
                    rr_max = rr + seg_window_x - math.floor(overlap_x / 2)
                    rr_max_patch = seg_window_x - math.floor(overlap_x / 2)
                    
                if cc == 0:
                    cc_min, cc_min_patch = 0, 0
                else:
                    cc_min = cc + math.ceil(overlap_y / 2)
                    cc_min_patch = math.ceil(overlap_y / 2)
                    
                if cc + seg_window_y == inp_y:
                    cc_max, cc_max_patch = inp_y, seg_window_y
                else:
                    cc_max = cc + seg_window_y - math.floor(overlap_y / 2)
                    cc_max_patch = seg_window_y - math.floor(overlap_y / 2)
                    
                cur_patch = dec_list[r_ind * len(cc_list) + c_ind,
                                    rr_min_patch * (1 + upsample_flag):rr_max_patch * (1 + upsample_flag),
                                    cc_min_patch * (1 + upsample_flag):cc_max_patch * (1 + upsample_flag)].astype(np.float32)
                output_dec[rr_min * (1 + upsample_flag):rr_max * (1 + upsample_flag),
                          cc_min * (1 + upsample_flag):cc_max * (1 + upsample_flag)] = cur_patch
        
        # Save output
        output_path = os.path.join(self.test_dir, 'output_2d_sr.tif')
        output_uint16 = np.uint16(1e4 * prctile_norm(output_dec, 3, 100))
        tiff.imwrite(output_path, output_uint16, dtype='uint16')
        
        # Verify output file exists and has correct super-resolution shape
        self.assertTrue(os.path.exists(output_path))
        loaded_output = tiff.imread(output_path)
        self.assertEqual(loaded_output.shape, (inp_x * 2, inp_y * 2))


if __name__ == '__main__':
    unittest.main()
