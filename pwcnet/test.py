#!/usr/bin/env python
import numpy
#import tifffile as tiff

from reshape_base_algos.image_warp import image_warp_grid1
import cv2

def warp(src_img, flow):
    assert src_img.shape[:2] == flow.shape[:2]
    X_flow = flow[..., 0]
    Y_flow = flow[..., 1]

    X_flow = numpy.ascontiguousarray(X_flow)
    Y_flow = numpy.ascontiguousarray(Y_flow)

    pred = image_warp_grid1(X_flow, Y_flow, src_img, 1.0, 0, 0)

    return pred


if __name__ == '__main__':

    img1 = './data/data/09084aeda464bfc66/09084aeda464bfc66_src.jpg'
    img2 = './data/data/09084aeda464bfc66/09084aeda464bfc66_gt.jpg'
    flow = './data/data/09084aeda464bfc66/09084aeda464bfc66_flow.npy'
    img1_np = cv2.imread(img1)
    img2_np = cv2.imread(img2)
    flow_np = numpy.load(flow)
    np_output = cv2.resize(flow_np, (img1_np.shape[1], img1_np.shape[0]))

    img_warp = warp(img1_np, np_output)

    cv2.imwrite('img_ori.jpg', img1_np)
    cv2.imwrite('img_gt.jpg', img2_np)
    cv2.imwrite('img_warped.jpg', img_warp)
