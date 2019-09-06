#Region Proposal Network - RPN
import tensorflow as tf
import numpy as np
import cv2
import xml.etree.ElementTree as ET
import os

#Importing self-defined functions
from Utils import Encode, Decode, Filter, Clip_Boxes, Change_Order, Smooth_L1_Loss, BBox_Overlap
from Anchors import Generate_Anchors
from Draw_bboxes import Draw_bboxes

'''
RPN performs the following tasks -
Input  - Image
Output - Proposals containing Objectness score + B-Boxes 
'''

class RPN:
    
    def __init__(self, image_pl, weight_file=None, train=False, gt_boxes_pl=None):
        
        self.prediction_dict = {}    
        self.parameters      = []                  
        
        self.image    = image_pl
        self.gt_boxes = gt_boxes_pl
        self.train    = train
               
        '''-----------------graph building starts from here-----------------'''
        self.Feature_Map()
        self.Cls_Reg()
        self.all_anchors = Generate_Anchors(self.f_map_shape, self.im_shape)      
        
        if not self.train:
            self.Proposals()  
            self.Load_trained_wts()
        else:
            self.Target_Anchors()    
            self.Loss()
            self.Optimize_Loss()
            self.Load_pretrained_wts(weight_file)                 
                  
    def Feature_Map(self):                   
        
        self.im_shape = tf.cast(tf.shape(self.image), dtype=tf.float64)
        
        # zero-mean input
        with tf.name_scope('preprocess') as scope:
            mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, 
                               shape=[1, 1, 1, 3], name='img_mean')
            image = self.image-mean
            
        # conv1_1
        with tf.name_scope('conv1_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 3, 64], dtype=tf.float32,
                                                     stddev=1e-1), name='weights', trainable=False)
            conv = tf.nn.conv2d(image, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv1_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv1_2
        with tf.name_scope('conv1_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 64], dtype=tf.float32,
                                                     stddev=1e-1), name='weights', trainable=False)
            conv = tf.nn.conv2d(self.conv1_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv1_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool1
        self.pool1 = tf.nn.max_pool(self.conv1_2,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool1')

        # conv2_1
        with tf.name_scope('conv2_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 128], dtype=tf.float32,
                                                     stddev=1e-1), name='weights', trainable=False)
            conv = tf.nn.conv2d(self.pool1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv2_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv2_2
        with tf.name_scope('conv2_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 128], dtype=tf.float32,
                                                     stddev=1e-1), name='weights', trainable=False)
            conv = tf.nn.conv2d(self.conv2_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv2_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool2
        self.pool2 = tf.nn.max_pool(self.conv2_2,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool2')

        # conv3_1
        with tf.name_scope('conv3_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 256], dtype=tf.float32,
                                                     stddev=1e-1), name='weights', trainable=True)
            conv = tf.nn.conv2d(self.pool2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv3_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv3_2
        with tf.name_scope('conv3_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32,
                                                     stddev=1e-1), name='weights', trainable=True)
            conv = tf.nn.conv2d(self.conv3_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv3_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv3_3
        with tf.name_scope('conv3_3') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32,
                                                     stddev=1e-1), name='weights', trainable=True)
            conv = tf.nn.conv2d(self.conv3_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv3_3 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool3
        self.pool3 = tf.nn.max_pool(self.conv3_3,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool3')

        # conv4_1
        with tf.name_scope('conv4_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights', trainable=True)
            conv = tf.nn.conv2d(self.pool3, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv4_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv4_2
        with tf.name_scope('conv4_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights', trainable=True)
            conv = tf.nn.conv2d(self.conv4_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv4_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv4_3
        with tf.name_scope('conv4_3') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights', trainable=True)
            conv = tf.nn.conv2d(self.conv4_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv4_3 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool4
        self.pool4 = tf.nn.max_pool(self.conv4_3,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool4')

        # conv5_1
        with tf.name_scope('conv5_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights', trainable=True)
            conv = tf.nn.conv2d(self.pool4, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv5_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]
        
        # conv5_2
        with tf.name_scope('conv5_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights', trainable=True)
            conv = tf.nn.conv2d(self.conv5_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv5_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]
        
        # conv5_3
        with tf.name_scope('conv5_3') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights', trainable=True)
            conv = tf.nn.conv2d(self.conv5_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv5_3 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]
                    
        self.f_map_shape = tf.cast(tf.shape(self.conv5_3), dtype=tf.float64)
        
    def Cls_Reg(self):
        
        # RPN_feature vector 512D
        with tf.name_scope('rpn_f_vector') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-2, name='weights'), trainable=True)
            conv = tf.nn.conv2d(self.conv5_3, kernel, [1, 1, 1, 1], padding='SAME')         #WHAT ABOUT EDGES OF FEATURE MAP????
            
            self.rpn_1 = tf.nn.relu(conv, name=scope)
            self.parameters += [kernel]
            
        # RPN_regression for delta coordinates
        with tf.name_scope('rpn_reg') as scope:
            kernel = tf.Variable(tf.truncated_normal([1, 1, 512, 9*4], dtype=tf.float32,
                                                     stddev=1e-2, name='weights'), trainable=True)
            conv = tf.nn.conv2d(self.rpn_1, kernel, [1, 1, 1, 1], padding='VALID')
            
            self.rpn_reg = tf.nn.relu(conv, name=scope)
            self.parameters += [kernel]
            
        #RPN_classifier object or non-objet
        with tf.name_scope('rpn_cls') as scope:
            kernel = tf.Variable(tf.truncated_normal([1, 1, 512, 9*2], dtype=tf.float32,
                                                     stddev=1e-2, name='weights'), trainable=True)
            conv = tf.nn.conv2d(self.rpn_1, kernel, [1, 1, 1, 1], padding='VALID')
            
            self.rpn_cls = tf.nn.relu(conv, name=scope)
            self.parameters += [kernel]
            
        #Flattening & Storing results in prediction_dict
        with tf.name_scope('Flat_n_Store') as scope:
            rpn_bbox_pred = tf.reshape(self.rpn_reg, [-1,4])
            
            rpn_cls_score = tf.reshape(self.rpn_cls, [-1,2])
            rpn_cls_prob  = tf.nn.softmax(rpn_cls_score)
            
            self.prediction_dict['rpn_bbox_pred'] = rpn_bbox_pred
            self.prediction_dict['rpn_cls_score'] = rpn_cls_score
            self.prediction_dict['rpn_cls_prob']  = rpn_cls_prob
        
    def Proposals(self):                #threshold probability 
        
        rpn_bbox_pred = self.prediction_dict['rpn_bbox_pred']
        rpn_cls_prob  = self.prediction_dict['rpn_cls_prob']
        
        '''x_min, y_min, x_max, y_max = tf.unstack(self.all_anchors, axis=1)
        
        anchor_filter = tf.logical_and( tf.logical_and ( tf.greater_equal(x_min, 0), 
                                                         tf.greater_equal(y_min, 0) ), 
                                        tf.logical_and ( tf.less_equal(x_max, self.im_shape[1]), 
                                                         tf.less_equal(y_max, self.im_shape[0]) ) )'''
        anchor_filter = Filter(self.all_anchors, self.im_shape)
        
        all_scores = rpn_cls_prob[ : , 1 ]
        all_scores = tf.reshape(all_scores, [-1])    
    
        #Anchors, bbox, scores--->>> filtered 
        anchors     = tf.boolean_mask(self.all_anchors, anchor_filter, axis=0)  #C
        bbox_pred   = tf.boolean_mask(rpn_bbox_pred, anchor_filter, axis=0)
        scores      = tf.boolean_mask(all_scores, anchor_filter, axis=0)
        
        #Decoding anchors and b-box predictions to give predicted points on image
        proposals   = Decode(anchors, bbox_pred)
        
        #Filter with < than threshold probability ~ 0.0
        min_prob_filter = tf.greater_equal(scores, 0.0)
    
        #Filter with -ve or zero area
        x1, y1, x2, y2 = tf.unstack(proposals, axis=1)
        width = x2 - x1 + 1
        height = y2 - y1 + 1
        area = width*height
        area_filter = tf.greater_equal(area, 0)
        
        #combining both filters and applying
        net_filter = tf.logical_and(min_prob_filter, area_filter)
        unsorted_proposals = tf.boolean_mask(proposals, net_filter)
        unsorted_scores    = tf.boolean_mask(scores,    net_filter)
    
        #Separating top-K proposals by score 
        k = 2000
        k = tf.minimum(k, tf.shape(unsorted_scores)[0])
        top_k = tf.nn.top_k(unsorted_scores, k=k)    
        
        top_k_proposals = tf.gather(unsorted_proposals, top_k.indices)
        top_k_scores = top_k.values
    
        #Clipping proposals to image size 
        top_k_proposals = Clip_Boxes(top_k_proposals, self.im_shape)
        
        #Non-Maximum supression 
        ordered_tf_proposal = Change_Order(top_k_proposals)
        selected_indices    = tf.image.non_max_suppression(ordered_tf_proposal, 
                                                    tf.reshape(top_k_scores, [-1]), 
                                                    max_output_size = 200,
                                                    iou_threshold = 0.5)
        nms_proposal_tf_order = tf.gather(ordered_tf_proposal, selected_indices)
    
        proposals = Change_Order(nms_proposal_tf_order)
        scores  = tf.gather(top_k_scores, selected_indices)
        
        self.prediction_dict['proposals'] = proposals
        self.prediction_dict['scores'] = scores
    
    def Target_Anchors(self):
        
        # Keep only the coordinates of gt_boxes
        gt_boxes    = self.gt_boxes[:, :4]
        all_anchors = self.all_anchors[:, :4] 
        
        # Only keep anchors inside the image
        '''(x_min_anchor, y_min_anchor,
         x_max_anchor, y_max_anchor) = tf.unstack(all_anchors, axis=1)
        
        anchor_filter = tf.logical_and(
                            tf.logical_and(
                                tf.greater_equal(x_min_anchor, 0),
                                tf.greater_equal(y_min_anchor, 0)
                            ),
                            tf.logical_and(
                                tf.less(x_max_anchor, im_shape[1]),
                                tf.less(y_max_anchor, im_shape[0])
                                          )   )'''
        anchor_filter = Filter(all_anchors, self.im_shape)
        
        # Filter anchors.
        anchors = tf.boolean_mask(all_anchors, anchor_filter, name='filter_anchors')    
            
        # Generate array with the labels for all_anchors.
        labels = tf.fill((tf.gather(tf.shape(all_anchors), [0])), -1)
        labels = tf.boolean_mask(labels, anchor_filter, name='filter_labels')    
            
        # Intersection over union (IoU) overlap
        overlaps = BBox_Overlap(tf.to_float(anchors), tf.to_float(gt_boxes))    
        
        # Generate array with the IoU value of the closest GT box for each anchor
        max_overlaps = tf.reduce_max(overlaps, axis=1)
        
        #Assigning background labels
        negative_overlap_nonzero = tf.less(max_overlaps, 0.3)
        
        labels = tf.where(condition=negative_overlap_nonzero,
                          x=tf.zeros(tf.shape(labels)), y=tf.to_float(labels))
       
        # Get the value of the max IoU for the closest anchor for each gt
        gt_max_overlaps = tf.reduce_max(overlaps, axis=0)

        # Find all the indices that match (at least one, but could be more)
        gt_argmax_overlaps = tf.squeeze(tf.equal(overlaps, gt_max_overlaps))
        gt_argmax_overlaps = tf.where(gt_argmax_overlaps)[:, 0]
        
        # Eliminate duplicates indices.
        gt_argmax_overlaps, _ = tf.unique(gt_argmax_overlaps)
        
        # Order the indices for sparse_to_dense compatibility
        gt_argmax_overlaps, _ = tf.nn.top_k(gt_argmax_overlaps, 
                                            k=tf.shape(gt_argmax_overlaps)[-1])
        gt_argmax_overlaps = tf.reverse(gt_argmax_overlaps, [0])

        # Foreground label: We set 1 at gt_argmax_overlaps_cond indices
        gt_argmax_overlaps_cond = tf.sparse_to_dense(gt_argmax_overlaps, 
                                                     tf.shape(labels, out_type=tf.int64),
                                                     True, default_value=False)
        
        labels = tf.where(condition=gt_argmax_overlaps_cond,
                          x=tf.ones(tf.shape(labels)), y=tf.to_float(labels))
        
        # Foreground label: above threshold Intersection over Union (IoU)
        positive_overlap_inds = tf.greater_equal(max_overlaps, 0.7)
        
        labels = tf.where(condition=positive_overlap_inds,
                          x=tf.ones(tf.shape(labels)), y=labels)
        
        # Subsample positive labels if we have too many
        def subsample_positive():
            # Shuffle the foreground indices
            disable_fg_inds = tf.random_shuffle(fg_inds)
           
            disable_place   = (tf.shape(fg_inds)[0] - num_fg)
            disable_fg_inds = disable_fg_inds[:disable_place]
            
            disable_fg_inds, _ = tf.nn.top_k(disable_fg_inds, 
                                             k=tf.shape(disable_fg_inds)[-1])
            disable_fg_inds = tf.reverse(disable_fg_inds, [0])
            disable_fg_inds = tf.sparse_to_dense(disable_fg_inds, 
                                                 tf.shape(labels, out_type=tf.int64),
                                                 True, default_value=False)
            
            # Put -1 to ignore the anchors in the selected indices
            return tf.where(
                            condition=tf.squeeze(disable_fg_inds),
                            x=tf.to_float(tf.fill(tf.shape(labels), -1)), y=labels
                           )
        
        num_fg = 128                                                            #128 +ve anchors
        # Get foreground indices, get True in the indices where we have a one
        fg_inds = tf.equal(labels, 1)
        # We get only the indices where we have True
        fg_inds = tf.squeeze(tf.where(fg_inds), axis=1)
        fg_inds_size = tf.size(fg_inds)
        # Condition for check if we have too many positive labels
        subsample_positive_cond = fg_inds_size > num_fg
        # Check the condition and subsample positive labels
        labels = tf.cond(subsample_positive_cond,
                         true_fn=subsample_positive, false_fn=lambda: labels)
        
        # Subsample negative labels if we have too many
        def subsample_negative():
            
            disable_bg_inds = tf.random_shuffle(bg_inds)

            disable_place = (tf.shape(bg_inds)[0] - num_bg)
            disable_bg_inds = disable_bg_inds[:disable_place]
            # Order the indices for sparse_to_dense compatibility
            disable_bg_inds, _ = tf.nn.top_k(disable_bg_inds, 
                                             k=tf.shape(disable_bg_inds)[-1])
            disable_bg_inds = tf.reverse(disable_bg_inds, [0])
            disable_bg_inds = tf.sparse_to_dense(disable_bg_inds, 
                                                 tf.shape(labels, out_type=tf.int64),
                                                 True, default_value=False)
            # Put -1 to ignore the anchors in the selected indices
            return tf.where(condition=tf.squeeze(disable_bg_inds),
                            x=tf.to_float(tf.fill(tf.shape(labels), -1)), 
                            y=labels)
        
        # Recalculate the foreground indices after (maybe) disable some of them

        # Get foreground indices, get True in the indices where we have a one.
        fg_inds = tf.equal(labels, 1)
        # We get only the indices where we have True.
        fg_inds = tf.squeeze(tf.where(fg_inds), axis=1)
        fg_inds_size = tf.size(fg_inds)

        num_bg = tf.to_int32(256 - fg_inds_size)
        # Get background indices, get True in the indices where we have a zero.
        bg_inds = tf.equal(labels, 0)
        # We get only the indices where we have True.
        bg_inds = tf.squeeze(tf.where(bg_inds), axis=1)
        bg_inds_size = tf.size(bg_inds)
        # Condition for check if we have too many positive labels.
        subsample_negative_cond = bg_inds_size > num_bg
        # Check the condition and subsample positive labels.
        labels = tf.cond(subsample_negative_cond,
                         true_fn=subsample_negative, false_fn=lambda: labels)
        
        # Return bbox targets with shape (anchors.shape[0], 4)
        # Find the closest gt box for each anchor
        argmax_overlaps = tf.argmax(overlaps, axis=1)
        argmax_overlaps_unique, _ = tf.unique(argmax_overlaps)
        # Filter the gt_boxes
        # We get only the indices where we have "inside anchors"
        anchor_filter_inds = tf.where(anchor_filter)
        gt_boxes = tf.gather(gt_boxes, argmax_overlaps)

        bbox_targets = Encode(anchors, gt_boxes)

        # For the anchors that arent foreground, we ignore the bbox_targets
        anchor_foreground_filter = tf.equal(labels, 1)
        bbox_targets = tf.where(condition=anchor_foreground_filter,
                                x=bbox_targets, y=tf.zeros_like(bbox_targets))

        # We unroll "inside anchors" value for all anchors (for shape compatibility)
        bbox_targets = tf.scatter_nd(indices=tf.to_int32(anchor_filter_inds),
                                     updates=bbox_targets, shape=tf.shape(all_anchors))
        labels_scatter = tf.scatter_nd(indices=tf.to_int32(anchor_filter_inds),
                                       updates=labels, shape=[tf.shape(all_anchors)[0]])
        # Put -1 to ignore the indices with 0 generated by scatter_nd
        labels = tf.where(condition=anchor_filter, x=labels_scatter,
                          y=tf.to_float(tf.fill(tf.shape(labels_scatter), -1)))

        max_overlaps = tf.scatter_nd(indices=tf.to_int32(anchor_filter_inds),
                                     updates=max_overlaps, shape=[tf.shape(all_anchors)[0]])

        ## return labels, bbox_targets, max_overlaps
        self.prediction_dict['rpn_bbox_target'] = bbox_targets
        self.prediction_dict['rpn_cls_target']  = labels
        self.prediction_dict['max_overlaps']    = max_overlaps
             
    def Loss(self):
             
        rpn_cls_score   = self.prediction_dict['rpn_cls_score']
        rpn_cls_target  = self.prediction_dict['rpn_cls_target']
    
        rpn_bbox_pred   = self.prediction_dict['rpn_bbox_pred']
        rpn_bbox_target = self.prediction_dict['rpn_bbox_target']
            
        with tf.variable_scope('RPN_Loss'):
             # Flatten already flat Tensor for usage as boolean mask filter.
             rpn_cls_target = tf.cast(tf.reshape(rpn_cls_target, [-1]),
                                      tf.int32, name='rpn_cls_target')
             # Transform to boolean tensor mask for not ignored.
             labels_not_ignored = tf.not_equal(rpn_cls_target, -1,
                                               name='labels_not_ignored')
             # Now we only have the labels we are going to compare with the
             # cls probability.
             labels = tf.boolean_mask(rpn_cls_target, labels_not_ignored)
             cls_score = tf.boolean_mask(rpn_cls_score, labels_not_ignored)
    
             # We need to transform `labels` to `cls_score` shape.
             # convert [1, 0] to [[0, 1], [1, 0]] for ce with logits.
             cls_target = tf.one_hot(labels, depth=2)
    
             # Equivalent to log loss
             ce_per_anchor = tf.nn.softmax_cross_entropy_with_logits_v2(
                                    labels=cls_target, logits=cls_score)
             self.prediction_dict['cross_entropy_per_anchor'] = ce_per_anchor          #LOSS 1
    
             # Finally, we need to calculate the regression loss over
             # `rpn_bbox_target` and `rpn_bbox_pred`.
             # We use SmoothL1Loss.
             rpn_bbox_target = tf.reshape(rpn_bbox_target, [-1, 4])
             rpn_bbox_pred = tf.reshape(rpn_bbox_pred, [-1, 4])
    
             # We only care for positive labels (we ignore backgrounds since
             # we don't have any bounding box information for it).
             positive_labels = tf.equal(rpn_cls_target, 1)
             rpn_bbox_target = tf.boolean_mask(rpn_bbox_target, positive_labels)
             rpn_bbox_pred   = tf.boolean_mask(rpn_bbox_pred, positive_labels)
    
             # We apply smooth l1 loss as described by the Fast R-CNN paper.
             reg_loss_per_anchor = Smooth_L1_Loss(rpn_bbox_pred, rpn_bbox_target)
             self.prediction_dict['reg_loss_per_anchor'] = reg_loss_per_anchor        #LOSS 2
    
             self.prediction_dict['rpn_reg_loss'] = tf.math.reduce_sum(reg_loss_per_anchor)
             self.prediction_dict['rpn_cls_loss'] = tf.math.reduce_sum(ce_per_anchor)
            
             '''# Loss summaries.
                tf.summary.scalar('batch_size', tf.shape(labels)[0], ['rpn'])
                foreground_cls_loss = tf.boolean_mask(
                        ce_per_anchor, tf.equal(labels, 1))
                    background_cls_loss = tf.boolean_mask(
                        ce_per_anchor, tf.equal(labels, 0))
                    tf.summary.scalar(
                        'foreground_cls_loss',
                        tf.reduce_mean(foreground_cls_loss), ['rpn'])
                    tf.summary.histogram(
                        'foreground_cls_loss', foreground_cls_loss, ['rpn'])
                    tf.summary.scalar(
                        'background_cls_loss',
                        tf.reduce_mean(background_cls_loss), ['rpn'])
                    tf.summary.histogram(
                        'background_cls_loss', background_cls_loss, ['rpn'])
                    tf.summary.scalar(
                        'foreground_samples', tf.shape(rpn_bbox_target)[0], ['rpn'])'''
    
    def Optimize_Loss(self, N_cls=256.0, Lambda=10.0, N_reg=2400.0):
        
        loss_reg_norm = self.prediction_dict['rpn_reg_loss']/N_cls
        loss_cls_norm = self.prediction_dict['rpn_cls_loss']*Lambda/N_reg
        
        net_loss = tf.add(loss_cls_norm,loss_reg_norm)       
        
        Optimizer = tf.contrib.opt.AdamWOptimizer(weight_decay=0.005, learning_rate=0.001,
                                                  name='Adam_W_Optimizer')
        self.Update_Params = Optimizer.minimize(net_loss)
        
    def Load_pretrained_wts(self, weight_file):
        
        keys = sorted(weight_file.keys())
        for i, k in enumerate(keys):
            if i<26:
                self.assign_wts = self.parameters[i].assign(weight_file[k])
     
if __name__ == '__main__':         
    
    def Test(img_dir):                                     #ADD WEIGHTS
       '''Takes input image                                #ADD hyper
          Resize it s.t. smaller side = 600 pixels         #PARAMETERS
          Put in placeholder and then feed dictionary'''
       image = cv2.imread(img_dir, 1)
       shape = np.shape(image)
       width, height = shape[1], shape[0]
       if width >= height:
           scale = 600.0/height
           height= 600.0
           width = scale*width
       else:
           scale = 600.0/width
           width = 600.0
           height= scale*width
       #Resize image shorter side ~ 600 pixels    
       image = cv2.resize(image,(int(width),int(height)))
       #Placeholder to contain test image
       image_pl = tf.placeholder(tf.float32, list(np.shape(image)))
       #Declaring object of RPN class OR
       #Building computation graph
       Rpn = RPN(image_pl=image_pl)             #weights=weights
       #Session
       with tf.Session() as sess:
          #Adding
           writer = tf.summary.FileWriter("Test_Graph", sess.graph)
          #Adding
           sess.run(tf.global_variables_initializer())
           proposals, scores = sess.run([Rpn.prediction_dict['proposals'], Rpn.prediction_dict['scores']], 
                                        feed_dict = {image_pl: image}) 
           Draw_bboxes(proposals, image)
           writer.close()
          
    def Train():
        
       i=0
       
       # Load images and their GT-Boxes    #
       # image_list, bbox_list = Dataset() #
       
       # Load pre-trained weights
       weights = np.load('vgg16_weights.npz')
                    
       #Declaring placeholders
       image_pl    = tf.placeholder(tf.float32)
       gt_boxes_pl = tf.placeholder(tf.int32, [None, 4])
       
       #Building Computation-graph
       Rpn   = RPN(image_pl=image_pl, weight_file=weights, train=True, gt_boxes_pl=gt_boxes_pl)                                                                               
       Saver = tf.train.Saver(Rpn.parameters)
       
       #Session
       with tf.Session() as sess:
           sess.run(tf.global_variables_initializer())
           sess.run(Rpn.assign_wts)
           writer = tf.summary.FileWriter("feature_graph", sess.graph)
           #Lopping through entire training data and updating parameters
           #for i in range(15000):
           #    image = image_list[i]
           #    gt_box_np = np.array(bbox_list[i])
           #    gt_box_np = np.reshape(gt_box_np, [-1,4])
           #    
           #    sess.run(Rpn.Update_Params, feed_dict = {image_pl: image, gt_boxes_pl: gt_box_np})  
           #    
           #    if i % 500 == 0:
           #        Saver.save(sess, 'RPN_weights', global_step=i)
           Img_dir = 'VOC_2012_DS\JPEGImages'
           ann_dir = 'VOC_2012_DS\Annotations'
           count = 0
           for xml_file in os.listdir(ann_dir):
                try :
                    bboxes = []
                    Image = None
                    scale = 1
                    xml_file_dir = os.path.join(ann_dir, xml_file)
                    #Reading xml files one-by-one
                    tree = ET.parse(xml_file_dir)
                    root = tree.getroot()
                    for sroot in root:
                        # Reading image file
                        if sroot.tag == 'filename':
                            img_name = sroot.text
                            img_dir  = os.path.join(Img_dir, img_name)
                            Image    = cv2.imread(img_dir, 1)
                            print('Reading image file')
                        # Getting scaled image with 600 as 
                        # length of smaller side                                            
                        if sroot.tag == 'size':
                            width = float(sroot[0].text)
                            height= float(sroot[1].text)
                            
                            if width >= height:
                                scale = 600.0/height
                                height= 600
                                width = scale*width
                            else:
                                scale = 600.0/width
                                width = 600
                                height= scale*width
                                
                            Image = cv2.resize(Image,(int(width),int(height)))
                            print('Getting scaled image')
                        # Getting bounding-boxes 
                        # co-ordinates
                        if sroot.tag == 'object':
                            for ssroot in sroot:
                                if ssroot.tag == 'bndbox':
                                    bbox = [int(float(ssroot[0].text)*scale),int(float(ssroot[1].text)*scale),
                                            int(float(ssroot[2].text)*scale),int(float(ssroot[3].text)*scale)]
                                    bboxes += [bbox]
                    gt_box_np = np.array(bboxes)
                    gt_box_np = np.reshape(gt_box_np, [-1,4])
                    
                    sess.run(Rpn.Update_Params, feed_dict = {image_pl: Image, gt_boxes_pl: gt_box_np}) 
                    count += 1
                    
                    if count % 5 == 0:
                        Saver.save(sess, 'RPN_weights', global_step=i)
                        print(' Training Image '+str(count)+' Finished')
                        print('DATA SAVED')
                    
                except:
                  print('Loading Error :-(')  
                  pass
                print('Image passed')
           print('Training Finished')     
           writer.close()
       print('Session Closed')        
       
       
       

       
       
       
       
    Train()       
       
            
            