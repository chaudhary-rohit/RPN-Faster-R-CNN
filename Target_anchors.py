import tensorflow as tf

def Target_Anchors(self, gt_boxes, all_anchors, im_shape):
    
        # Keep only the coordinates of gt_boxes
        gt_boxes = gt_boxes[:, :4]
        all_anchors = all_anchors[:, :4]

        # Only keep anchors inside the image
        (x_min_anchor, y_min_anchor,
         x_max_anchor, y_max_anchor) = tf.unstack(all_anchors, axis=1)
        
        anchor_filter = tf.logical_and(
            tf.logical_and(
                tf.greater_equal(x_min_anchor, 0),
                tf.greater_equal(y_min_anchor, 0)
            ),
            tf.logical_and(
                tf.less(x_max_anchor, im_shape[1]),
                tf.less(y_max_anchor, im_shape[0])
            )
        )
        # We (force) reshape the filter so that we can use it as a boolean mask
        anchor_filter = tf.reshape(anchor_filter, [-1])
        # Filter anchors.
        anchors = tf.boolean_mask(all_anchors, anchor_filter, name='filter_anchors')

        # Generate array with the labels for all_anchors.
        labels = tf.fill((tf.gather(tf.shape(all_anchors), [0])), -1)
        labels = tf.boolean_mask(labels, anchor_filter, name='filter_labels')   
        
        # Intersection over union (IoU) overlap between the anchors and the
        # ground truth boxes.
        overlaps = bbox_overlap_tf(tf.to_float(anchors), tf.to_float(gt_boxes))

        # Generate array with the IoU value of the closest GT box for each
        # anchor.
        max_overlaps = tf.reduce_max(overlaps, axis=1)
        
        # Assign bg labels first so that positive labels can clobber them.
        # First we get an array with True where IoU is less than
        # self._negative_overlap
        negative_overlap_nonzero = tf.less(max_overlaps, 0.3)

        # Finally we set 0 at True indices
        labels = tf.where(condition=negative_overlap_nonzero,
                          x=tf.zeros(tf.shape(labels)), y=tf.to_float(labels))
        # Get the value of the max IoU for the closest anchor for each gt.
        gt_max_overlaps = tf.reduce_max(overlaps, axis=0)

        # Find all the indices that match (at least one, but could be more).
        gt_argmax_overlaps = tf.squeeze(tf.equal(overlaps, gt_max_overlaps))
        gt_argmax_overlaps = tf.where(gt_argmax_overlaps)[:, 0]
        # Eliminate duplicates indices.
        gt_argmax_overlaps, _ = tf.unique(gt_argmax_overlaps)
        # Order the indices for sparse_to_dense compatibility
        gt_argmax_overlaps, _ = tf.nn.top_k(
            gt_argmax_overlaps, k=tf.shape(gt_argmax_overlaps)[-1])
        gt_argmax_overlaps = tf.reverse(gt_argmax_overlaps, [0])

        # Foreground label: for each ground-truth, anchor with highest overlap.
        # When the argmax is many items we use all of them (for consistency).
        # We set 1 at gt_argmax_overlaps_cond indices
        gt_argmax_overlaps_cond = tf.sparse_to_dense(
            gt_argmax_overlaps, tf.shape(labels, out_type=tf.int64),
            True, default_value=False
        )

        labels = tf.where(
            condition=gt_argmax_overlaps_cond,
            x=tf.ones(tf.shape(labels)), y=tf.to_float(labels)
        )

        # Foreground label: above threshold Intersection over Union (IoU)
        # First we get an array with True where IoU is greater or equal than
        # self._positive_overlap
        positive_overlap_inds = tf.greater_equal(
            max_overlaps, self._positive_overlap)
        # Finally we set 1 at True indices
        labels = tf.where(
            condition=positive_overlap_inds,
            x=tf.ones(tf.shape(labels)), y=labels
        )

        if self._clobber_positives:
            # Assign background labels last so that negative labels can clobber
            # positives. First we get an array with True where IoU is less than
            # self._negative_overlap
            negative_overlap_nonzero = tf.less(
                max_overlaps, self._negative_overlap)
            # Finally we set 0 at True indices
            labels = tf.where(
                condition=negative_overlap_nonzero,
                x=tf.zeros(tf.shape(labels)), y=labels
            )