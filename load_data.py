import numpy as np
import os
import torch
import cv2
from torchvision import transforms
from torch.utils.data import Dataset

train_dir = os.path.join('./dataset')
train_transform = transforms.Compose([transforms.ToTensor()])

class Flowdataset(Dataset):
    def __init__(self, parent_path, dir_list, inputsize):

        self.data_info = self.get_img_info(parent_path, dir_list)
        self.divider = 20
        self.img_h = inputsize
        self.img_w = inputsize

    def __getitem__(self, index):

        src_img_path,gt_img_path, skel_path, flow_path = self.data_info[index]
        src_img = cv2.imread(src_img_path).astype(np.float32)
        gt_img = cv2.imread(gt_img_path).astype(np.float32)
        skel_map = np.load(skel_path)
        flow = np.load(flow_path)

        intWidth = np.shape(src_img)[1]
        intHeight = np.shape(src_img)[0]

        if flow.shape[0] != intHeight or flow.shape[1] != intWidth:
            flow = cv2.resize(flow, (intWidth, intHeight), interpolation=cv2.INTER_LINEAR)

        if skel_map.shape[0] != intHeight or skel_map.shape[1] != intWidth:
            skel_map = cv2.resize(skel_map, (intWidth, intHeight), interpolation=cv2.INTER_LINEAR)

        #skel_map = cv2.resize(skel_map, (intWidth, intHeight), interpolation=cv2.INTER_LINEAR)

        roi_height_pad = intHeight // self.divider
        roi_width_pad = intWidth // self.divider
        paded_roi_h = intHeight + 2 * roi_height_pad
        paded_roi_w = intWidth + 2 * roi_width_pad

        src_img = src_img[:, :, ::-1]
        gt_img = gt_img[:, :, ::-1]

        skel_map[skel_map == 0] = -1
        skel_map[skel_map > 0] = 1
        skel_map = np.pad(skel_map, ((roi_height_pad, roi_height_pad), (roi_width_pad, roi_width_pad), (0, 0)),
                                   'constant', constant_values=-1)
        skel_map_resized = cv2.resize(skel_map, (self.img_w, self.img_h))
        skel_map_resized[skel_map_resized <0] = -1.0
        skel_map_resized[skel_map_resized >-0.5] = 1.0
        skel_map_transformed = torch.from_numpy(skel_map_resized.transpose((2, 0, 1)))

        ###########################
        src_npy = np.pad(src_img, ((roi_height_pad, roi_height_pad), (roi_width_pad, roi_width_pad), (0, 0)),
                                 'edge')
        src_npy = cv2.resize(src_npy, (self.img_w, self.img_h))

        src_npy_loss = np.pad(src_img, ((roi_height_pad, roi_height_pad), (roi_width_pad, roi_width_pad), (0, 0)),
                                 'constant', constant_values=0)
        src_npy_loss = cv2.resize(src_npy_loss, (self.img_w, self.img_h))

        src_npy /= 255
        src_npy -= 0.5
        src_npy *= 2.0

        src_npy_loss /= 255
        src_npy_loss -= 0.5
        src_npy_loss *= 2.0
        
        src_tensor = torch.from_numpy(src_npy.transpose((2, 0, 1)))
        src_loss_tensor = torch.from_numpy(src_npy_loss.transpose((2, 0, 1)))

        gt_npy = np.pad(gt_img, ((roi_height_pad, roi_height_pad), (roi_width_pad, roi_width_pad), (0, 0)),
                                 'constant', constant_values=0)
        gt_npy = cv2.resize(gt_npy, (self.img_w, self.img_h))

        gt_npy /= 255
        gt_npy -= 0.5
        gt_npy *= 2.0
        
        gt_tensor = torch.from_numpy(gt_npy.transpose((2, 0, 1)))

        ############################
        flow = np.pad(flow, ((roi_height_pad, roi_height_pad), (roi_width_pad, roi_width_pad), (0, 0)),
                                 'constant', constant_values=0)
        flow[...,0] = flow[...,0] / paded_roi_w * 2
        flow[...,1] = flow[...,1] / paded_roi_h * 2

        resized_flow = cv2.resize(flow, (self.img_w, self.img_h))
        flow_tensor = torch.from_numpy(resized_flow)
        

        return src_tensor,src_loss_tensor,gt_tensor,flow_tensor, skel_map_transformed     #,intHeight,intWidth
    
    def __len__(self):
        return len(self.data_info)

    def get_img_info(self,parent_path, dir_list):
        data_info = list()
        for filename in dir_list:
            src_img_path = os.path.join(parent_path, filename, filename + '_src.jpg')
            gt_img_path = os.path.join(parent_path, filename, filename + '_gt.jpg')
            skel_path = os.path.join(parent_path, filename, filename + '_skel.npy')
            flow_path = os.path.join(parent_path, filename, filename + '_flow.npy')

            data_info.append((src_img_path,gt_img_path, skel_path, flow_path))
        return data_info




class Flowdatatestset(Dataset):
    def __init__(self, parent_path, origin_path, dir_list, inputsize):

        self.data_info = self.get_img_info(parent_path, origin_path, dir_list)
        self.divider = 20
        self.img_h = inputsize
        self.img_w = inputsize

    def __getitem__(self, index):

        origin_src_img_path, src_img_path, origin_gt_img_path, gt_img_path, skel_path = self.data_info[index]
        origin_src_img = cv2.imread(origin_src_img_path).astype(np.float32)
        src_img = cv2.imread(src_img_path).astype(np.float32)
        origin_gt_img = cv2.imread(origin_gt_img_path).astype(np.float32)
        #gt_img = cv2.imread(gt_img_path).astype(np.float32)
        skel_map = np.load(skel_path)

        intWidth = np.shape(src_img)[1]
        intHeight = np.shape(src_img)[0]

        if skel_map.shape[0] != intHeight or skel_map.shape[1] != intWidth:
            skel_map = cv2.resize(skel_map, (intWidth, intHeight), interpolation=cv2.INTER_LINEAR)

        roi_height_pad = intHeight // self.divider
        roi_width_pad = intWidth // self.divider
        paded_roi_h = intHeight + 2 * roi_height_pad
        paded_roi_w = intWidth + 2 * roi_width_pad

        src_img = src_img[:, :, ::-1]
        src_img_for_return = src_img.copy()

        #gt_img_for_return = gt_img.copy()
        origin_src_img_for_return = origin_src_img.copy()
        origin_gt_img_for_return = origin_gt_img.copy()

        skel_map[skel_map == 0] = -1
        skel_map[skel_map > 0] = 1
        skel_map = np.pad(skel_map, ((roi_height_pad, roi_height_pad), (roi_width_pad, roi_width_pad), (0, 0)),
                                   'constant', constant_values=-1)
        skel_map_resized = cv2.resize(skel_map, (self.img_w, self.img_h))
        skel_map_resized[skel_map_resized <0] = -1.0
        skel_map_resized[skel_map_resized >-0.5] = 1.0
        skel_map_transformed = torch.from_numpy(skel_map_resized.transpose((2, 0, 1)))

        ###########################
        src_npy = np.pad(src_img, ((roi_height_pad, roi_height_pad), (roi_width_pad, roi_width_pad), (0, 0)),
                                 'edge')

        src_npy = cv2.resize(src_npy, (self.img_w, self.img_h))

        src_npy /= 255
        src_npy -= 0.5
        src_npy *= 2.0
        
        src_tensor = torch.from_numpy(src_npy.transpose((2, 0, 1)))

        return src_tensor, skel_map_transformed, paded_roi_h, paded_roi_w, src_img_for_return, roi_height_pad, roi_width_pad, intHeight, intWidth, origin_src_img_for_return, origin_gt_img_for_return
    
    def __len__(self):
        return len(self.data_info)

    def get_img_info(self,parent_path, origin_path, dir_list):
        data_info = list()
        for filename in dir_list:
            origin_src_img_path = os.path.join(origin_path, filename, filename + '-src.jpg')
            src_img_path = os.path.join(parent_path, filename, filename + '-src.jpg')
            origin_gt_img_path = os.path.join(origin_path, filename, filename + '-gt.jpg')
            gt_img_path = os.path.join(parent_path, filename, filename + '-gt.jpg')
            skel_path = os.path.join(parent_path, filename, filename + '-skel.npy')

            data_info.append((origin_src_img_path, src_img_path, origin_gt_img_path, gt_img_path, skel_path))
        return data_info