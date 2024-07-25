# 自作ファイル
# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import time

import cv2
import mmcv
import torch
import cupy as cp
import numpy as np

from mmdet.apis import inference_detector, init_detector
from mmdet.registry import VISUALIZERS
# from self_make_visualizer import add_datasample

import os
from image_edit import image_Edit

def parse_args():
    parser = argparse.ArgumentParser(description='MMDetection webcam demo')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--device', type=str, default='cuda:0', help='CPU/CUDA device option')
    parser.add_argument(
        '--camera-id', type=int, default=0, help='camera device id')
    parser.add_argument(
        '--score-thr', type=float, default=0.5, help='bbox score threshold')
    args = parser.parse_args()
    return args

def pred_gpu_memory(height, wedth, channel):
    torch.cuda.synchronize()
    start = time.time()
    gpu_before_img = torch.zeros((height, wedth, channel+1), device='cuda')
    gpu_edit_img = torch.empty((height, wedth, channel+1), device='cuda')

    print(gpu_before_img.device)
    print(gpu_before_img.size())
    print(gpu_edit_img.device)
    print(gpu_edit_img.size())

    torch.cuda.synchronize()
    elapsed_time = time.time() - start
    print(elapsed_time, 'sec.')

    return gpu_before_img, gpu_edit_img

def cupy_pred_gpu_memory(height, wedth, channel):
    size = height*wedth*(channel+1)
    # メモリプールの作成
    memory_pool = cp.cuda.MemoryPool()
    memory_pool.set_limit(fraction=0.4)
    cp.cuda.set_allocator(memory_pool.malloc)

    # 事前にメモリを確保
    gpu_before_img = memory_pool.malloc(size* 2*np.dtype(np.uint8).itemsize)
    gpu_edit_img = memory_pool.malloc(size* 2*np.dtype(np.uint8).itemsize)
    gpu_mask_img = memory_pool.malloc(size* 2*np.dtype(np.bool_).itemsize)

    return gpu_before_img, gpu_edit_img, gpu_mask_img

def main():
    inf_sum = 0.0  # inference time
    visualizer_sum = 0.0  # visualizer time
    cutout_sum = 0.0  # cutout time
    loop_sum = 0.0  # loop time
    edit = True
    re_scale = 2
    loop = 1
    size = 1280
    gpu = True

    # concert2.jpg edit_image.jpg
    # file_name = 'Input image_sice.png'
    file_name = 'cars.jpg'
    # file_name = 'concert2.jpg'
    # file_name = 'human2.jpg'
    file_path = 'C:/Users/kokis/Segmentation-Model/rtmdet/demo/inputs/' + file_name
    try:
        open(file_path)
        img = cv2.imread(file_path)
    except FileNotFoundError as e:
            print(f"ファイルが見つかりませんでした: {e}")
            return
    args = parse_args()

    img = cv2.resize(img, (size,size))

    # # 画像を加工
    # if edit is True:
    #     edit_img = image_Edit(img, re_scale, gaussian=0.3, sobel=0.3, laplacian_num=0.3)
    # else:
    #     edit_img = cv2.resize(img, (img.shape[1]*re_scale, img.shape[0]*re_scale))
    # cv2.imshow('edit_image', edit_img)

    device = torch.device(args.device)
    model = init_detector(args.config, args.checkpoint, device=device)

    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    visualizer.dataset_meta = model.dataset_meta
    original_image = img
    print("gpu is {gpu}".format(gpu=gpu))
    print('Press "Esc", "q" or "Q" to exit.')
    for i in range(5):
        print("--------------------------------")
        time_start = time.time()

        det_img = original_image
        cv2.imshow("before", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print("before=", det_img.nbytes)

        # ret_val, img = camera.read()
        # if not ret_val:
        #     print("break")
        #     return

        # inference
        time_result = time.time()  
        result = inference_detector(model, det_img)  # edit_img
        inf_time = time.time()-time_result
        inf_sum = inf_sum + inf_time #推論時間の合計
        print("inferencce={:.4f}ms" .format((time.time()-time_result)*1000))  ####60~85ms

        # det_img = mmcv.imconvert(img, 'bgr', 'rgb')
        
        if loop == 1:
            t = time.time()
            height, wedth, channel = det_img.shape
            # gpu_before_img, gpu_edit_img = pred_gpu_memory(height, wedth, channel)
            gpu_before_img, gpu_edit_img, gpu_mask_img = cupy_pred_gpu_memory(height, wedth, channel)
            loop += 1
            print("preparetion gpu memory = {:.4f}ms".format((time.time()-t)*1000))
        
        # image1 = cv2.cvtColor(det_img, cv2.COLOR_BGR2BGRA)
        # # # image_torch = torch.from_numpy(image).to(torch.uint8)
        # # # gpu_before_img.copy_(image_torch, True)
        # t = time.time()
        # cupy_before_img1 = cp.ndarray(image1.shape, dtype=cp.uint8, memptr=gpu_before_img)
        # cupy_before_img1.set(image1)
        # # print("2", gpu_before_img.device)
        # print("t1",(time.time()-t))
        # print(det_img)

        det_img = cv2.cvtColor(det_img, cv2.COLOR_BGR2BGRA)
        # visualizer
        visualizer_time = time.time()
        img_result = visualizer.add_datasample(
            name='result',
            image=det_img,
            data_sample=result,
            draw_gt=False,
            pred_score_thr=args.score_thr,
            show=False,
            gpu=gpu,
            gpu_before_img=gpu_before_img,
            gpu_edit_img=gpu_edit_img,
            gpu_mask_img=gpu_mask_img)
        visualizer_sum = visualizer_sum + (time.time()-visualizer_time)
        print("visualizer = {:.4f}ms" .format((time.time()-visualizer_time)*1000))
        # img_result = mmcv.imconvert(img_result,'rgb', 'bgr')
        loop_sum = loop_sum + (time.time()-time_start)
        print("loop time = {:.4f}ms".format((time.time()-time_start)*1000))
        # t = time.time()
        img_result = img_result.astype(np.uint8)
        img_result = cv2.bitwise_and(det_img, img_result)
        # print(img_result)
        cv2.imshow('result', img_result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print("after=", img_result.nbytes)
        # print("imshow = {:.4f}ms".format((time.time()-t)*1000))

        # print("inference平均時間={:.4f}ms" .format(1000*inf_sum))
        # inf_sum  = 0
        # print("visualizer平均時間={:.4f}ms" .format(1000*visualizer_sum))
        # visualizer_sum  = 0
        # # print("send time平均時間={:.4f}ms" .format(1000*bytes_time_sum/(count-1)))
        # print("cutout平均時間={:.4f}ms" .format(1000*cutout_sum))
        # cutout_sum = 0
        # print("loop平均時間={:.4f}ms" .format(1000*loop_sum))
        # loop_sum = 0

        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    # # メモリの解放
    # gpu_before_img.free_all_blocks()
    # gpu_edit_img.free_all_blocks()
    # gpu_mask_img.free_all_blocks()

    if not os.path.exists('./outputs'):
        print("make directory outputs")
        os.makedirs('outputs')
    # cv2.imwrite('./outputs/cutout_tiny_'+file_name, img)
    # cv2.imwrite('./outputs/mask_tiny_640.jpg', black_image)



if __name__ == '__main__':
    main()
