import tensorflow as tf
import numpy as np
'''
CONTAINS:
    i.  generate_anchors_reference(base_size, aspect_ratios, scales)
    ii. def generate_anchors(self, feature_map_shape):
'''

def Generate_Anchors_Reference():

    base_size = 128
    aspect_ratios = np.array([0.5, 1, 2], dtype='float32')
    scales = np.array([1, 2, 4], dtype='float32')
    
    scales_grid, aspect_ratios_grid = np.meshgrid(scales, aspect_ratios)
    base_scales = scales_grid.reshape(-1)
    base_aspect_ratios = aspect_ratios_grid.reshape(-1)
    
    aspect_ratio_sqrts = np.sqrt(base_aspect_ratios)
    heights = base_scales * aspect_ratio_sqrts * base_size
    widths = base_scales / aspect_ratio_sqrts * base_size

    # Center point has the same X, Y value.
    center_xy = 0
    
    # Create anchor reference.
    anchors = np.column_stack([ center_xy - (widths - 1) / 2, center_xy - (heights - 1) / 2,
                                center_xy + (widths - 1) / 2, center_xy + (heights - 1) / 2 ])
    return anchors
    
def Generate_Anchors(f_map_shape, im_shape, output_stride=16):
#Generates all anchors by adding feature point*output stride and reference anchors        
    
    grid_width  = f_map_shape[2]  # width
    grid_height = f_map_shape[1]  # height
    
    shift_x = tf.range(grid_width) * output_stride
    shift_y = tf.range(grid_height) * output_stride
    shift_x, shift_y = tf.meshgrid(shift_x, shift_y)
    
    shift_x = tf.reshape(shift_x, [-1])
    shift_y = tf.reshape(shift_y, [-1])
    
    shifts = tf.stack([shift_x, shift_y, shift_x, shift_y], axis=0)
    
    shifts = tf.transpose(shifts)
    # Shifts now is a (H x W, 4) Tensor
    
    # Expand dims to use broadcasting sum.
    all_anchors = (np.expand_dims(Generate_Anchors_Reference(), axis=0) +
                   tf.expand_dims(shifts, axis=1))
    # Flatten
    all_anchors = tf.reshape(all_anchors, (-1, 4))
        
    return all_anchors

if __name__ == '__main__'  :            #TESTS ALL ANCHORS GENERATION
    
    all_anchors = Generate_Anchors(f_map_shape = (1,21,32,512), im_shape = (334, 500, 3 ))
        
    with tf.Session() as sess:
        output = sess.run(all_anchors)
        from Draw_bboxes import Draw_bboxes
        import cv2
        Draw_bboxes(output, cv2.imread('sample.jpg', 1))
    
    
    
    
    
    
    
    
    
    



