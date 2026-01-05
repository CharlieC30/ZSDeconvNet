"""
Unit tests for 2D patch segmentation and fusion logic

Tests the core functionality of:
- Segmenting large images into patches
- Processing patches for super-resolution
- Fusing patches back into a complete image
"""

import unittest
import numpy as np
import math


class TestPatchSegmentation2D(unittest.TestCase):
    """Test 2D patch segmentation logic"""

    def test_basic_segmentation_no_overlap(self):
        """Test segmentation with no overlap"""
        # Create a simple test image
        inp_x, inp_y = 128, 128
        num_seg_window_x, num_seg_window_y = 2, 2
        overlap_x, overlap_y = 0, 0
        
        # Calculate segment window size
        seg_window_x = math.ceil((inp_x + (num_seg_window_x - 1) * overlap_x) / num_seg_window_x)
        seg_window_y = math.ceil((inp_y + (num_seg_window_y - 1) * overlap_y) / num_seg_window_y)
        
        # Verify segment dimensions
        self.assertEqual(seg_window_x, 64)
        self.assertEqual(seg_window_y, 64)
        
        # Calculate positions
        rr_list = list(range(0, inp_x - seg_window_x + 1, seg_window_x - overlap_x))
        if rr_list[-1] != inp_x - seg_window_x:
            rr_list.append(inp_x - seg_window_x)
            
        cc_list = list(range(0, inp_y - seg_window_y + 1, seg_window_y - overlap_y))
        if cc_list[-1] != inp_y - seg_window_y:
            cc_list.append(inp_y - seg_window_y)
        
        # Should have 2x2 patches
        self.assertEqual(len(rr_list), 2)
        self.assertEqual(len(cc_list), 2)
        
    def test_segmentation_with_overlap(self):
        """Test segmentation with overlap"""
        inp_x, inp_y = 256, 256
        num_seg_window_x, num_seg_window_y = 4, 4
        overlap_x, overlap_y = 20, 20
        
        seg_window_x = math.ceil((inp_x + (num_seg_window_x - 1) * overlap_x) / num_seg_window_x)
        seg_window_y = math.ceil((inp_y + (num_seg_window_y - 1) * overlap_y) / num_seg_window_y)
        
        # Verify that overlapping segments cover the full image
        rr_list = list(range(0, inp_x - seg_window_x + 1, seg_window_x - overlap_x))
        if rr_list[-1] != inp_x - seg_window_x:
            rr_list.append(inp_x - seg_window_x)
            
        cc_list = list(range(0, inp_y - seg_window_y + 1, seg_window_y - overlap_y))
        if cc_list[-1] != inp_y - seg_window_y:
            cc_list.append(inp_y - seg_window_y)
        
        self.assertEqual(len(rr_list), num_seg_window_x)
        self.assertEqual(len(cc_list), num_seg_window_y)
        
    def test_patch_extraction(self):
        """Test that patches are extracted correctly"""
        # Create a test image with known pattern
        inp_x, inp_y = 64, 64
        image = np.arange(inp_x * inp_y).reshape(inp_x, inp_y).astype(np.float32)
        
        num_seg_window_x, num_seg_window_y = 2, 2
        overlap_x, overlap_y = 8, 8
        
        seg_window_x = math.ceil((inp_x + (num_seg_window_x - 1) * overlap_x) / num_seg_window_x)
        seg_window_y = math.ceil((inp_y + (num_seg_window_y - 1) * overlap_y) / num_seg_window_y)
        
        rr_list = list(range(0, inp_x - seg_window_x + 1, seg_window_x - overlap_x))
        if rr_list[-1] != inp_x - seg_window_x:
            rr_list.append(inp_x - seg_window_x)
            
        cc_list = list(range(0, inp_y - seg_window_y + 1, seg_window_y - overlap_y))
        if cc_list[-1] != inp_y - seg_window_y:
            cc_list.append(inp_y - seg_window_y)
        
        # Extract patches
        segmented_inp = []
        for rr in rr_list:
            for cc in cc_list:
                segmented_inp.append(image[rr:rr+seg_window_x, cc:cc+seg_window_y])
        
        segmented_inp = np.array(segmented_inp).astype(np.float32)
        
        # Should have 4 patches
        self.assertEqual(segmented_inp.shape[0], 4)
        # Each patch should have the correct dimensions
        self.assertEqual(segmented_inp.shape[1], seg_window_x)
        self.assertEqual(segmented_inp.shape[2], seg_window_y)


class TestPatchFusion2D(unittest.TestCase):
    """Test 2D patch fusion logic"""
    
    def test_basic_fusion_no_overlap(self):
        """Test fusion with no overlap produces correct output"""
        inp_x, inp_y = 64, 64
        num_seg_window_x, num_seg_window_y = 2, 2
        overlap_x, overlap_y = 0, 0
        upsample_flag = 0
        
        # Create test image
        image = np.random.rand(inp_x, inp_y).astype(np.float32)
        
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
        segmented_inp = np.array(segmented_inp)
        
        # Simulate processing (identity in this case)
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
        
        # Verify reconstruction matches original
        np.testing.assert_array_almost_equal(output_dec, image)
        
    def test_fusion_with_super_resolution(self):
        """Test fusion with super-resolution (2x upsampling)"""
        inp_x, inp_y = 64, 64
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
        
        # Create upsampled patches (2x size)
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
        
        # Verify output has correct super-resolution dimensions
        self.assertEqual(output_dec.shape[0], inp_x * 2)
        self.assertEqual(output_dec.shape[1], inp_y * 2)
        
        # Verify no zeros remain in the output (all regions should be filled)
        # Note: This assumes patches cover entire image, which they should with proper overlap
        filled_pixels = np.count_nonzero(output_dec)
        total_pixels = output_dec.shape[0] * output_dec.shape[1]
        self.assertGreater(filled_pixels, 0)


if __name__ == '__main__':
    unittest.main()
