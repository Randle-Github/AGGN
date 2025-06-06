# coding=utf-8
# summary:
# author: Jianqiang Ren
# date:
import cv2
import numpy as np
from reshape_base_algos.slim_utils import enlarge_box_tblr,get_mask_bbox,gen_skeleton_map, get_map_fusion_map_cuda,visualize_flow,vis_joints,resize_on_long_side,plot_one_box
import torch
import time
from reshape_base_algos.image_warp import image_warp_grid1
from torch.nn import functional as F
import copy
from config.test_config import TESTCONFIG

class PersonInfo(object):
    def __init__(self, joints, logger):
        self._logger = logger

        self.joints = joints

        self.flow = None
        
        self.pad_boder = False
        self.height_expand = 0
        self.width_expand = 0
        self.coeff = 0.2
        self.network_input_W = 256
        self.network_input_H = 256


    def update_attribute(self, pad_boder, height_expand, width_expand):
        self.pad_boder = pad_boder
        self.height_expand = height_expand
        self.width_expand = width_expand
        if pad_boder:
            self.joints[:, 0] += width_expand
            self.joints[:, 1] += height_expand
     
    def pred_flow(self, img, flow_net, device):
        '''
        
        :param img: ndarray
        :param flow_net: pytorch network object
        :return:
        '''
        with torch.no_grad():
            if img is None:
                print("image is none")
                self.flow = None
            
            if len(img.shape)==2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
            if self.pad_boder:
                height_expand = self.height_expand
                width_expand = self.width_expand
                pad_img = cv2.copyMakeBorder(img, height_expand, height_expand, width_expand, width_expand,
                                                cv2.BORDER_CONSTANT, value=(127, 127, 127))
    
            else:
                height_expand = 0
                width_expand = 0
                pad_img = img.copy()
    
            canvas = np.zeros(shape=(pad_img.shape[0], pad_img.shape[1]), dtype=np.float32)
    

            self.human_joint_box = self.__joint_to_body_box()

            self.human_box = enlarge_box_tblr(self.human_joint_box, pad_img, ratio=0.25)
            human_box_height = self.human_box[1] - self.human_box[0]
            human_box_width = self.human_box[3] - self.human_box[2]

            self.leg_joint_box = self.__joint_to_leg_box()
            self.leg_box = enlarge_box_tblr(self.leg_joint_box, pad_img, ratio=0.25)

            self.arm_joint_box = self.__joint_to_arm_box()
            self.arm_box = enlarge_box_tblr(self.arm_joint_box, pad_img, ratio=0.1)

            x_flows = []
            y_flows = []
            multi_bbox = []

            for scale in TESTCONFIG.flow_scales: # better for metric
                self._logger.logger.info("---scale:{}---".format(scale))
                t_loop_start = time.time()
                scale_value = float(scale.split('_')[-1])

                arm_box = copy.deepcopy(self.arm_box)

                if arm_box[0] is None:
                    self._logger.logger.warn('arm_box is None')
                    arm_box = self.human_box
            
                arm_box_height = arm_box[1] - arm_box[0]
                arm_box_width = arm_box[3] - arm_box[2]

                t1 = time.time()
                roi_bbox = None

                if arm_box_width < human_box_width * 0.1 or arm_box_height < human_box_height * 0.1:
                    roi_bbox = self.human_box
                else:
                    arm_box = enlarge_box_tblr(arm_box, pad_img, ratio=scale_value)
                    if scale == 'upper_0.2':
                        arm_box[0] = min(arm_box[0], int(self.joints[0][1]))
                    if scale.startswith('upper'):
                        self._logger.logger.info( 'human_box:{}'.format(self.human_box))
                        roi_bbox = [max(self.human_box[0], arm_box[0]), min(self.human_box[1], arm_box[1]),
                                    max(self.human_box[2], arm_box[2]), min(self.human_box[3], arm_box[3])]
                        if roi_bbox[1] - roi_bbox[0] < 1 or roi_bbox[3] - roi_bbox[2] < 1:
                            continue
                           
                    elif scale.startswith('lower'):
                        roi_bbox = [max(self.human_box[0], self.leg_box[0]), min(self.human_box[1], self.leg_box[1]),
                                    max(self.human_box[2], self.leg_box[2]), min(self.human_box[3], self.leg_box[3])]
    
                        if roi_bbox[1] - roi_bbox[0] < 1 or roi_bbox[3] - roi_bbox[2] < 1:
                            continue
                self._logger.logger.info('time of gen roi_bbox: {}ms'.format(int((time.time() - t1) * 1000)))
                
                self._logger.logger.info('scale:{},roi_bbox:{}'.format(scale,roi_bbox))

                t1 = time.time()
                skel_map, roi_bbox = gen_skeleton_map(self.joints, "depth",  input_roi_box=roi_bbox)
                self._logger.logger.info('time of link_joints: {}ms'.format(int((time.time() - t1) * 1000)))

                if roi_bbox is None:
                    self._logger.logger.info("ROIisNone")
                    continue

                t1 = time.time()
                if skel_map.dtype != np.float32:
                    skel_map = skel_map.astype(np.float32)

                skel_map -=1.0   # [0,2] ->[-1,1]
                self._logger.logger.info('skel_map.shape:{}'.format(skel_map.shape))
                self._logger.logger.info('time of skel_map process: {}ms'.format(int((time.time() - t1) * 1000)))

                multi_bbox.append(roi_bbox)

                
                self._logger.logger.info("roi_bbox:{}".format(roi_bbox))
                roi_bbox_height = roi_bbox[1] - roi_bbox[0]
                roi_bbox_width = roi_bbox[3] - roi_bbox[2]
       
                self._logger.logger.info('roi_bbox_width:{},roi_bbox_height:{}'.format(roi_bbox_width, roi_bbox_height))

                divider = TESTCONFIG.divider
                self._logger.logger.debug("divider:{}".format(TESTCONFIG.divider))
                assert skel_map.shape[0] == roi_bbox_height
                assert skel_map.shape[1] == roi_bbox_width
                roi_height_pad = roi_bbox_height // divider
                roi_width_pad = roi_bbox_width // divider
                paded_roi_h = roi_bbox_height + 2 * roi_height_pad
                paded_roi_w = roi_bbox_width + 2 * roi_width_pad

                t1 = time.time()

                roi_height_pad_joint = skel_map.shape[0] // divider
                roi_width_pad_joint = skel_map.shape[1] // divider
                skel_map = np.pad(skel_map, ((roi_height_pad_joint, roi_height_pad_joint), (roi_width_pad_joint, roi_width_pad_joint), (0, 0)),
                                   'constant', constant_values=-1)
    
                skel_map_resized = cv2.resize(skel_map, (self.network_input_W, self.network_input_H))

                skel_map_resized[skel_map_resized <0] = -1.0
                skel_map_resized[skel_map_resized >-0.5] = 1.0
                skel_map_transformed = torch.from_numpy(skel_map_resized.transpose((2, 0, 1)))
    
                roi_npy = pad_img[roi_bbox[0]:roi_bbox[1], roi_bbox[2]:roi_bbox[3], :].copy()
                if roi_npy.dtype != np.float32:
                    roi_npy = roi_npy.astype(np.float32)

                roi_npy = np.pad(roi_npy, ((roi_height_pad, roi_height_pad), (roi_width_pad, roi_width_pad), (0, 0)),
                                 'edge')

                roi_npy = roi_npy[:, :, ::-1]
    
                roi_npy = cv2.resize(roi_npy, (self.network_input_W, self.network_input_H))

                roi_npy *= 1.0 / 255
                roi_npy -= 0.5
                roi_npy *= 2
                
                rgb_tensor = torch.from_numpy(roi_npy.transpose((2, 0, 1)))

                self._logger.logger.info('time of create tensor: {}ms'.format(int((time.time() - t1) * 1000)))

                t1 = time.time()
                rgb_tensor = rgb_tensor.unsqueeze(0).to(device)
                skel_map_tensor = skel_map_transformed.unsqueeze(0).to(device)

                self._logger.logger.info('time of toGPU: {}ms'.format(int((time.time() - t1) * 1000)))


                t1 = time.time()
                warped_img_val, flow_field_val = flow_net(rgb_tensor, skel_map_tensor) # inference, connectivity_mask [1,12,16,16]

                t2 = time.time()
                self._logger.logger.info('time of flow_field_val: {}ms'.format(int((t2 - t1) * 1000)))
    
                t1 = time.time()
                flow_field_val = flow_field_val.detach().squeeze().cpu().numpy()

                flow_field_val = cv2.resize(flow_field_val, (paded_roi_w, paded_roi_h), interpolation=cv2.INTER_LINEAR)
                flow_field_val[...,0] = flow_field_val[...,0] * paded_roi_w * 0.5 * 2 * self.coeff
                flow_field_val[...,1] = flow_field_val[...,1] * paded_roi_h * 0.5 * 2 * self.coeff

                # remove pad areas
                flow_field_val = flow_field_val[roi_height_pad:flow_field_val.shape[0] - roi_height_pad,
                      roi_width_pad:flow_field_val.shape[1] - roi_width_pad,:]

                self._logger.logger.info('time of postprocess_0: {}ms'.format(int((time.time() - t1) * 1000)))

                diffuse_width = max(roi_bbox_width // 3, 1)
                diffuse_height = max(roi_bbox_height // 3, 1)
                assert roi_bbox_width == flow_field_val.shape[1]
                assert roi_bbox_height == flow_field_val.shape[0]

                self._logger.logger.debug('diffuse_width:{},diffuse_height:{}'.format(diffuse_width, diffuse_height))
                self._logger.logger.debug('roi_bbox_width:{},roi_bbox_height:{}'.format(roi_bbox_width, roi_bbox_height))
                self._logger.logger.debug('flow_field_val width:{}, height:{}'.format(flow_field_val.shape[1], flow_field_val.shape[0]))
                t1 = time.time()
                origin_flow = np.zeros((pad_img.shape[0] + 2 * diffuse_height, pad_img.shape[1] + 2 * diffuse_width, 2),
                                      dtype=np.float32)

                self._logger.logger.info('time of postprocess_1: {}ms'.format(int((time.time() - t1) * 1000)))

                t1 = time.time()
                flow_field_val = np.pad(flow_field_val, ((diffuse_height, diffuse_height), (diffuse_width, diffuse_width), (0, 0)),
                                 'linear_ramp')
                self._logger.logger.info('time of postprocess_2: {}ms'.format(int((time.time() - t1) * 1000)))

                origin_flow[roi_bbox[0]:roi_bbox[1] + 2 * diffuse_height, roi_bbox[2]:roi_bbox[3] + 2 * diffuse_width] = flow_field_val

                origin_flow = origin_flow[diffuse_height:-diffuse_height, diffuse_width:-diffuse_width,:]

                x_flows.append(origin_flow[...,0])
                y_flows.append(origin_flow[...,1])
                self._logger.logger.info('time of loop {}: {}ms'.format(scale, int((time.time() - t_loop_start) * 1000)))

            if len(x_flows) == 0:
                return {'rDx': np.zeros(canvas.shape[:2],dtype=np.float32),
                        'rDy': np.zeros(canvas.shape[:2],dtype=np.float32),
                        "multi_bbox":multi_bbox,
                        'x_fusion_map':np.ones(canvas.shape[:2],dtype=np.float32),
                        'y_fusion_map':np.ones(canvas.shape[:2],dtype=np.float32)}
            else:
                origin_rDx, origin_rDy, x_fusion_map, y_fusion_map = self.blend_multiscale_flow(x_flows, y_flows, device=device)

            return {'rDx': origin_rDx,'rDy': origin_rDy,"multi_bbox":multi_bbox, 'x_fusion_map':x_fusion_map, 'y_fusion_map':y_fusion_map}
 
    def visualize(self, img, origin_rDx, origin_rDy,width_expand,height_expand,multi_bbox,  transRatio = 1.0):
        flow_field_val_vis = visualize_flow(np.dstack((origin_rDx, origin_rDy))) * 255

        pad_img = cv2.copyMakeBorder(img, height_expand, height_expand, width_expand, width_expand,
                                     cv2.BORDER_CONSTANT, value=(127, 127, 127))
        
        pred  = image_warp_grid1( origin_rDx, origin_rDy, pad_img, transRatio, width_expand, height_expand)

        flow_field_val_vis = flow_field_val_vis * 0.45 + pad_img * 0.55

        plot_one_box([self.human_box[2], self.human_box[0],self.human_box[3], self.human_box[1]], flow_field_val_vis,  (0, 0, 255),
                     label="human_box", line_thickness=3)
  
        for i, roi_bbox in enumerate(multi_bbox):
            plot_one_box([roi_bbox[2], roi_bbox[0],roi_bbox[3], roi_bbox[1] ],flow_field_val_vis,(255, 0, 0),label=f"{i}",line_thickness=1)

        color = np.random.randint(0, 255, size=3)
        color = [int(i) for i in color]
        pred = vis_joints(pred, self.joints, color, show_text=True)

      
        return {'pred': pred, 'flow': flow_field_val_vis}

    @staticmethod
    def blend_multiscale_flow(x_flows, y_flows, device=None):
        scale_num = len(x_flows)
        if scale_num == 1:
            return x_flows[0], y_flows[0], np.ones_like(x_flows[0]), np.ones_like(x_flows[0])
        
        origin_rDx = np.zeros((x_flows[0].shape[0], x_flows[0].shape[1]), dtype=np.float32)
        origin_rDy = np.zeros((y_flows[0].shape[0], y_flows[0].shape[1]), dtype=np.float32)

        x_fusion_map, x_acc_map = get_map_fusion_map_cuda(x_flows, 1, device=device)
        y_fusion_map, y_acc_map = get_map_fusion_map_cuda(y_flows, 1, device=device)

        x_flow_map = 1.0 / x_fusion_map
        y_flow_map = 1.0 / y_fusion_map
    
        all_acc_map = x_acc_map + y_acc_map
        all_acc_map = all_acc_map.astype(np.uint8)
        roi_box = get_mask_bbox(all_acc_map, threshold=1)

        
        if roi_box[0] is None or roi_box[1]-roi_box[0] <= 0 or roi_box[3]-roi_box[2] <= 0:
            roi_box = [0, x_flow_map.shape[0], 0, x_flow_map.shape[1]]
        
        roi_x_flow_map = x_flow_map[roi_box[0]:roi_box[1], roi_box[2]:roi_box[3]]
        roi_y_flow_map = y_flow_map[roi_box[0]:roi_box[1], roi_box[2]:roi_box[3]]
        
        roi_width = roi_x_flow_map.shape[1]
        roi_height = roi_x_flow_map.shape[0]

        roi_x_flow_map, scale = resize_on_long_side(roi_x_flow_map, 320)
        roi_y_flow_map, scale = resize_on_long_side(roi_y_flow_map, 320)
    
        roi_x_flow_map = cv2.blur(roi_x_flow_map, (55, 55))
        roi_y_flow_map = cv2.blur(roi_y_flow_map, (55, 55))
    
        roi_x_flow_map = cv2.resize(roi_x_flow_map, (roi_width, roi_height))
        roi_y_flow_map = cv2.resize(roi_y_flow_map, (roi_width, roi_height))
        
        x_flow_map[roi_box[0]:roi_box[1], roi_box[2]:roi_box[3]] = roi_x_flow_map
        y_flow_map[roi_box[0]:roi_box[1], roi_box[2]:roi_box[3]] = roi_y_flow_map
    
        for i in range(scale_num):
            origin_rDx += x_flows[i]
            origin_rDy += y_flows[i]
            
        origin_rDx *= x_flow_map
        origin_rDy *= y_flow_map
        
        return origin_rDx, origin_rDy, x_flow_map, y_flow_map
    
    def __joint_to_body_box(self):
        joint_left = int(np.min(self.joints,axis=0)[0])
        joint_right = int(np.max(self.joints, axis=0)[0])
        joint_top = int(np.min(self.joints,axis=0)[1])
        joint_bottom = int(np.max(self.joints, axis=0)[1])
        return [joint_top, joint_bottom, joint_left, joint_right]
    
    def __joint_to_leg_box(self):
        leg_joints = self.joints[8:,:]
        if np.max(leg_joints,axis=0)[2] < 0.05:
            return [0,0,0,0]
        joint_left = int(np.min(leg_joints,axis=0)[0])
        joint_right = int(np.max(leg_joints, axis=0)[0])
        joint_top = int(np.min(leg_joints,axis=0)[1])
        joint_bottom = int(np.max(leg_joints, axis=0)[1])
        return [joint_top, joint_bottom, joint_left, joint_right]

    def __joint_to_arm_box(self):
        arm_joints = self.joints[2:8,:]
        if np.max(arm_joints,axis=0)[2] < 0.05:
            return [0,0,0,0]
        joint_left = int(np.min(arm_joints,axis=0)[0])
        joint_right = int(np.max(arm_joints, axis=0)[0])
        joint_top = int(np.min(arm_joints,axis=0)[1])
        joint_bottom = int(np.max(arm_joints, axis=0)[1])
        return [joint_top, joint_bottom, joint_left, joint_right]