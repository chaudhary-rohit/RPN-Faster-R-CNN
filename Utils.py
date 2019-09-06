import tensorflow as tf

def Encode(b_boxes, gt_boxes):
    
    with tf.name_scope('Encode'):
        (bboxes_width, bboxes_height,
         bboxes_urx, bboxes_ury) = Center_Corner(b_boxes)

        (gt_boxes_width, gt_boxes_height,
         gt_boxes_urx, gt_boxes_ury) = Center_Corner(gt_boxes)

        targets_dx = (gt_boxes_urx - bboxes_urx)/(bboxes_width )
        targets_dy = (gt_boxes_ury - bboxes_ury)/(bboxes_height )

        targets_dw = tf.log(gt_boxes_width / bboxes_width) 
        targets_dh = tf.log(gt_boxes_height / bboxes_height) 

        targets = tf.concat(
            [targets_dx, targets_dy, targets_dw, targets_dh], axis=1)

        return targets

def Decode(all_anchors, bbox_pred):
    
    with tf.name_scope('Decode'):
        (roi_width, roi_height,
         roi_urx, roi_ury) = Center_Corner(all_anchors)

        dx, dy, dw, dh = tf.split(bbox_pred, 4, axis=1)

        pred_ur_x = dx * roi_width + roi_urx
        pred_ur_y = dy * roi_height + roi_ury
        pred_w = tf.exp(dw) * roi_width
        pred_h = tf.exp(dh) * roi_height

        bbox_x1 = pred_ur_x - 0.5 * pred_w
        bbox_y1 = pred_ur_y - 0.5 * pred_h

        # This -1 extra is different from reference implementation
        bbox_x2 = pred_ur_x + 0.5 * pred_w - 1.
        bbox_y2 = pred_ur_y + 0.5 * pred_h - 1.

        bboxes = tf.concat(
            [bbox_x1, bbox_y1, bbox_x2, bbox_y2], axis=1)

        return bboxes

def Center_Corner(bboxes):
    
    with tf.name_scope('Center_Corner'):
        bboxes = tf.cast(bboxes, tf.float32)
        x1, y1, x2, y2 = tf.split(bboxes, 4, axis=1)
        width = x2 - x1 + 1.
        height = y2 - y1 + 1.
    
        # Calculate up right point of bbox (urx = up right x)
        urx = x1 + .5 * width
        ury = y1 + .5 * height
    
        return width, height, urx, ury
    
def Clip_Boxes(bboxes, im_shape):
    
    with tf.name_scope('Clip_Boxes'):
        bboxes   = tf.cast(bboxes,  dtype=tf.float32)
        im_shape = tf.cast(im_shape, dtype=tf.float32)

        x1, y1, x2, y2 = tf.split(bboxes, 4, axis=1)
        width  = im_shape[1]
        height = im_shape[0]
        
        x1 = tf.maximum(tf.minimum(x1, width - 1.0), 0.0)
        x2 = tf.maximum(tf.minimum(x2, width - 1.0), 0.0)

        y1 = tf.maximum(tf.minimum(y1, height - 1.0), 0.0)
        y2 = tf.maximum(tf.minimum(y2, height - 1.0), 0.0)

        bboxes = tf.concat([x1, y1, x2, y2], axis=1)

        return bboxes
    
def Change_Order(bboxes):

    with tf.name_scope('change_order'):
        first_min, second_min, first_max, second_max = tf.unstack(bboxes, axis=1)
        bboxes = tf.stack([second_min, first_min, second_max, first_max], axis=1)
        return bboxes
    
def Filter(anchors, im_shape):
    
    with tf.name_scope('Filter_outside_anchors'):
        x_min, y_min, x_max, y_max = tf.unstack(anchors, axis=1)
        
        anchor_filter = tf.logical_and(tf.logical_and(tf.greater_equal(x_min,0), 
                                                      tf.greater_equal(y_min,0)), 
                                       tf.logical_and(tf.less_equal(x_max,im_shape[1]), 
                                                      tf.less_equal(y_max,im_shape[0])))
        anchor_filter = tf.reshape(anchor_filter, [-1])
        return anchor_filter
    
def Smooth_L1_Loss(bbox_prediction, bbox_target, sigma=3.0):

    sigma2 = sigma ** 2
    diff = bbox_prediction - bbox_target
    abs_diff = tf.abs(diff)
    abs_diff_lt_sigma2 = tf.less(abs_diff, 1.0 / sigma2)
    bbox_loss = tf.reduce_sum(tf.where(abs_diff_lt_sigma2, 
                                       0.5 * sigma2 * tf.square(abs_diff),
                                       abs_diff - 0.5 / sigma2), [1]  )
    return bbox_loss
    
def BBox_Overlap(bboxes1, bboxes2):

    with tf.name_scope('BBox_Overlap'):
        x11, y11, x12, y12 = tf.split(bboxes1, 4, axis=1)
        x21, y21, x22, y22 = tf.split(bboxes2, 4, axis=1)

        xI1 = tf.maximum(x11, tf.transpose(x21))
        yI1 = tf.maximum(y11, tf.transpose(y21))

        xI2 = tf.minimum(x12, tf.transpose(x22))
        yI2 = tf.minimum(y12, tf.transpose(y22))

        intersection = (
                        tf.maximum(xI2 - xI1 + 1., 0.) *
                        tf.maximum(yI2 - yI1 + 1., 0.)
                       )

        bboxes1_area = (x12 - x11 + 1) * (y12 - y11 + 1)
        bboxes2_area = (x22 - x21 + 1) * (y22 - y21 + 1)

        union = (bboxes1_area + tf.transpose(bboxes2_area)) - intersection
        iou = tf.maximum(intersection / union, 0)

        return iou
    
    ''' Returns tensor with shape (total_bboxes1, total_bboxes2)
        with the IoU (intersection over union) of bboxes1[i] and bboxes2[j]
        in [i, j].
    '''   
if __name__=='__main__':
    bboxes1 = tf.constant([[1,2,3,4],[4,3,2,1],[6,7,8,9],[9,8,7,6]], dtype=tf.float32)
    bboxes2 = tf.constant([[1,2,3,4],[4,3,2,1],[6,7,8,9],[9,8,7,6]], dtype=tf.float32)
    print(tf.Session().run(BBox_Overlap(bboxes1, bboxes2)))
    
    
    
    