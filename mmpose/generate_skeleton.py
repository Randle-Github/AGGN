# Copyright (c) OpenMMLab. All rights reserved.
import os
from reshape_base_algos.slim_utils import resize_on_long_side, enlarge_box_tblr, gen_skeleton_map, joint_to_body_box
import cv2
import numpy as np
from tqdm import tqdm
from mmpose.apis import inference_topdown
from mmpose.apis import init_model as init_pose_estimator
from mmpose.evaluation.functional import nms
from mmpose.utils import adapt_mmdet_pipeline

try:
    from mmdet.apis import inference_detector, init_detector
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False

det_config = './mmpose/rtmdet_m_640-8xb32_coco-person.py'
det_checkpoint = './mmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth'
pose_config = './mmpose/rtmpose-x_8xb256-700e_coco-384x288.py'
pose_checkpoint = './mmpose/rtmpose-x_simcc-body7_pt-body7_700e-384x288-71d7b7e9_20230629.pth'
device = 'cuda:0'
draw_heatmap = False

detector = init_detector(
        det_config, det_checkpoint, device=device)
detector.cfg = adapt_mmdet_pipeline(detector.cfg)

pose_estimator = init_pose_estimator(
        pose_config,
        pose_checkpoint,
        device=device,
        cfg_options=dict(
            model=dict(test_cfg=dict(output_heatmaps=draw_heatmap))))

data_path = os.path.join(os.getcwd(), 'data')
src_path = os.path.join(data_path, 'src')
gt_path = os.path.join(data_path, 'gt')
img_list = os.listdir(src_path)
out_path = os.path.join(data_path, 'collected_data')

pairs = [[9, 7], [7, 5], [5, 6], [6, 8], [8, 10], [6, 12], [12, 11], [11, 5], [12, 14], [14, 16], [11, 13], [13, 15]]

resize_h = 384
resize_w = 384

for img_name in tqdm(img_list):
    
    src_img_path = os.path.join(src_path, img_name)
    gt_img_path = os.path.join(gt_path, img_name)

    img_src = cv2.imread(src_img_path)
    img_gt = cv2.imread(gt_img_path)

    if np.all(img_src == img_gt):continue

    img_id = img_name.split('.')[0]
    out_id_path = os.path.join(out_path, img_id) 
    if not os.path.exists(out_id_path):
        os.makedirs(out_id_path)

    det_result = inference_detector(detector, src_img_path)
    pred_instance = det_result.pred_instances.cpu().numpy()
    bboxes = np.concatenate(
            (pred_instance.bboxes, pred_instance.scores[:, None]), axis=1)
    bboxes = bboxes[np.logical_and(pred_instance.labels == 0,
                                       pred_instance.scores > 0.3)]

    bboxes = bboxes[nms(bboxes, 0.3), :4]
    
    pose_results = inference_topdown(pose_estimator, src_img_path, bboxes)
    sample = pose_results[0].pred_instances

    scores = sample.keypoint_scores[0]
    keypoints = sample.keypoints[0]

    scores_reshaped = scores.reshape((-1,1))

    if len(bboxes) > 0 :
        dx1, dy1, dx2, dy2 = np.uint32(bboxes[0])

    small_size = 1200

    if img_src.shape[0] > small_size or img_src.shape[1] > small_size:
        _img_src, _scale = resize_on_long_side(img_src, small_size)
        _img_gt, scale = resize_on_long_side(img_gt, small_size)
        keypoints = keypoints * _scale
    else:
        _img_src = img_src
        _img_gt = img_gt

    reshaped_height = _img_src.shape[0]
    reshaped_width = _img_src.shape[1]

    all_key_info = np.concatenate((keypoints, scores_reshaped), axis = 1)

    human_joint_box = joint_to_body_box(all_key_info, 0.3)
    if human_joint_box == None:
        human_joint_box = joint_to_body_box(all_key_info, 0.2)
    if human_joint_box == None:
        human_joint_box = joint_to_body_box(all_key_info, 0.1)
    
    human_box = enlarge_box_tblr(human_joint_box, _img_src, ratio=0.3)
    human_box_height = human_box[1] - human_box[0]
    human_box_width = human_box[3] - human_box[2]
    roi_bbox = human_box

    skel_map, roi_bbox = gen_skeleton_map(all_key_info, "depth", input_roi_box=roi_bbox)
    skel_map[skel_map > 0] = 255
    skel_map = cv2.resize(skel_map, (resize_w,  resize_h),interpolation=cv2.INTER_LINEAR)

    roi_src_npy = _img_src[roi_bbox[0]:roi_bbox[1], roi_bbox[2]:roi_bbox[3], :].copy()
    roi_gt_npy = _img_gt[roi_bbox[0]:roi_bbox[1], roi_bbox[2]:roi_bbox[3], :].copy()

    cv2.imwrite(os.path.join(out_id_path, img_id + '_src' + '.jpg'), roi_src_npy)
    cv2.imwrite(os.path.join(out_id_path, img_id + '_gt' + '.jpg'), roi_gt_npy)
    cv2.imwrite(os.path.join(out_id_path, img_id + '_skel' + '.png'), skel_map[..., -1])
    np.save(os.path.join(out_id_path, img_id + '_skel' + '.npy'), skel_map)
