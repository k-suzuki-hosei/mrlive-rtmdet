# 自作ファイル
# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import time

import cv2
import mmcv
import torch
import socket
import struct
import cupy as cp
import numpy as np

from mmdet.apis import inference_detector, init_detector
from mmdet.registry import VISUALIZERS
# from self_make_visualizer import add_datasample

import Socket as sc
import pyzed.sl as sl
import Setzed as sz

import matplotlib.pyplot as plt


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

def settingSockets2():
    global server_address_left, server_address_right
    global client_socket_left, client_socket_right
    server_address_left = ('localhost', 1500)
    server_address_right = ('localhost', 2000)
    client_socket_left = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    client_socket_right = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

def sendZed(image):
    # print(len(image))
    # 左画像を送信
    # udp_image_left = cv.resize(image[:,:image.shape[1]//2,:], (128, 128))  # 左サイズの変更
    # udp_image_left = cv.resize(image[:,:image.shape[1]//2,:], (256, 256))  # 左サイズの変更
    # udp_image_left = cv.resize(image[:,:image.shape[1]//2,:], (512, 512))  # 左サイズの変更
    # udp_image_left = cv.resize(image[:,:image.shape[1]//2,:], (1024, 1024))  # 左サイズの変更

    udp_image_left = image[:, :image.shape[1]//2, :]
    # height, width, channels = udp_image_left.shape
    # print(f"画像の高さ: {height} ピクセル")
    # print(f"画像の幅: {width} ピクセル")
    # print(f"チャンネル数: {channels}")

    img_bytes_left = cv2.imencode('.png', udp_image_left)[1].tobytes()

    # if(len(img_bytes_left) >= 250000):
    #     udp_image_left = cv.resize(image[:,image.shape[1]//2:,:], (512, 512))
    #     img_bytes_left = cv.imencode('.png', udp_image_left)[1].tobytes()

        # if(len(img_bytes_left) >=250000):
        #     udp_image_left = cv.resize(image[:,image.shape[1]//2:,:], (256, 256)) 
        #     img_bytes_left = cv.imencode('.png', udp_image_left)[1].tobytes()

    chunk_size = 5000  # 最大UDPパケットサイズ
    num =0
    loopnum = len(img_bytes_left)//chunk_size + 1
    for i in range(0, len(img_bytes_left), chunk_size):
        chunk = img_bytes_left[i:i+chunk_size]
        num += 1
        # header = struct.pack('!HH', loopnum, num) #2バイトづつ確保
        header = struct.pack('!BB', loopnum, num) #1バイトづつ確保
        # print(loopnum, num)
        # print(len(img_bytes_left))
        data_to_send = header + chunk
        client_socket_left.sendto(data_to_send, server_address_left)
    
    # 右画像を送信
    # udp_image_right = cv.resize(image[:,image.shape[1]//2:,:], (128, 128))  # 右サイズの変更
    # udp_image_right = cv.resize(image[:,image.shape[1]//2:,:], (256, 256))  # 右サイズの変更
    # udp_image_right = cv.resize(image[:,image.shape[1]//2:,:], (512, 512))  # 左サイズの変更
    # udp_image_right = cv.resize(image[:,image.shape[1]//2:,:], (1024, 1024))  # 左サイズの変更
    udp_image_right = image[:, image.shape[1] // 2:, :]
    img_bytes_right = cv2.imencode('.png', udp_image_right)[1].tobytes()

    # if(len(img_bytes_right) >= 250000):
    #     udp_image_right = cv.resize(image[:,image.shape[1]//2:,:], (512, 512))
    #     img_bytes_right = cv.imencode('.png', udp_image_right)[1].tobytes()

        # if(len(img_bytes_right) >=250000):
        #     udp_image_right = cv.resize(image[:,image.shape[1]//2:,:], (256, 256)) 
        #     img_bytes_right = cv.imencode('.png', udp_image_right)[1].tobytes()
    

    chunk_size = 5000  # 最大UDPパケットサイズ
    # chunk_size = 65505  # 最大UDPパケットサイズ
    num =0
    loopnum = len(img_bytes_right)//chunk_size + 1
    for i in range(0, len(img_bytes_right), chunk_size):
        chunk = img_bytes_right[i:i+chunk_size]
        num += 1
        # header = struct.pack('!HH', loopnum, num) #2バイトづつ確保
        header = struct.pack('!BB', loopnum, num) #1バイトづつ確保
        # print(loopnum, num)
        # print(len(img_bytes_right))
        data_to_send = header + chunk
        
        # client_socket_left.sendto(data_to_send, server_address_left)
        client_socket_right.sendto(data_to_send, server_address_right)

def closeSocket(zed):
    if zed == True:
        client_socket_left.close()
        client_socket_right.close()

def pred_gpu_memory(height, wedth, channel):
    torch.cuda.synchronize()
    start = time.time()
    gpu_before_img = torch.empty((height, wedth, channel), device='cuda')
    gpu_edit_img = torch.empty((height, wedth, channel+1), device='cuda')

    print(gpu_before_img.device)
    print(gpu_edit_img.device)

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
    gpu_before_img = memory_pool.malloc(size* np.dtype(np.uint8).itemsize)
    gpu_edit_img = memory_pool.malloc(size* np.dtype(np.uint8).itemsize)
    gpu_mask_img = memory_pool.malloc(size*3* np.dtype(np.bool_).itemsize)

    return gpu_before_img, gpu_edit_img, gpu_mask_img

def main():
    count = 0  # カウント
    inf_sum = 0.0  # inference time
    visualizer_sum = 0.0  # visualizer time
    cutout_sum = 0.0  # cutout time
    loop_sum = 0.0  # loop time
    loop = 1  # 最初のループ

    video_file = "C:/Users/kokis/Segmentation-Model/rtmdet/demo/inputs/cars.mp4"
    args = parse_args()
    cutout :bool = True  #### 切り抜くかデフォルトか
    size = 640
    gpu = False
    
    # sz.zed_setting()
    # sc.settingSocket()
    # sc.settingSockets2()  ####
    # settingSockets2()

    # build the model from a config file and a checkpoint file
    device = torch.device(args.device)
    model = init_detector(args.config, args.checkpoint, device=device)

    # init visualizer
    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    # the dataset_meta is loaded from the checkpoint and
    # then pass to the model in init_detector
    visualizer.dataset_meta = model.dataset_meta

    # camera = cv2.VideoCapture(args.camera_id)
    # camera = cv2.VideoCapture(0)
    # camera = cv2.VideoCapture(video_file)
    camera = cv2.VideoCapture(video_file)
    if not camera.isOpened():
        print("Error: Could not open video source.")
    else:
        print("suuccess")

    print("gpu is {gpu}".format(gpu=gpu))
    print('Press "Esc", "q" or "Q" to exit.')
    while True:
        # print("----------------------------loot%d--------------------------------" %count)
        time_start = time.time()

        t = time.time()
        ret_val, img = camera.read()
        if not ret_val:
            print("can't read image")
            break
        # if sz.zed.grab() == sl.ERROR_CODE.SUCCESS:
        #     # A new image is available if grab() returns SUCCESS
        #     sz.zed.retrieve_image(sz.zed_right, sl.VIEW.RIGHT)
        #     sz.zed.retrieve_image(sz.zed_left, sl.VIEW.LEFT)
        #     # frame = cv.hconcat([zed_right,zed_left])
        #     img = cv2.hconcat([sz.zed_left.get_data()[:,:,:3], sz.zed_right.get_data()[:,:,:3]])
        img = cv2.resize(img, (size*2, size))
        # print("zed=",(time.time()-t)*1000)
        
        # inference
        time_result = time.time()  
        result = inference_detector(model, img)
        inf_time = time.time()-time_result
        if count>2:
            inf_sum = inf_sum + inf_time #推論時間の合計
        # print("inference={:.4f}ms" .format((time.time()-time_result)*1000))  ####60~85ms

        if loop == 1:
            height, wedth, channel = img.shape
            # gpu_before_img, gpu_edit_img = pred_gpu_memory(height, wedth, channel)
            gpu_before_img, gpu_edit_img, gpu_mask_img = cupy_pred_gpu_memory(height, wedth, channel)
            loop += 1

        # img = mmcv.imconvert(img, 'bgr', 'rgb')

        # img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        # visualizer_gpu
        visualizer_time = time.time()
        img_result = visualizer.add_datasample(
            name='result',
            image=img,
            data_sample=result,
            draw_gt=False,
            pred_score_thr=args.score_thr,
            show=False,
            gpu=gpu,
            gpu_before_img=gpu_before_img,
            gpu_edit_img=gpu_edit_img,
            gpu_mask_img=gpu_mask_img)
        # img_result = cv2.bitwise_and(img, img_result)
        if count>2:
            visualizer_sum = visualizer_sum + (time.time()-visualizer_time)
        # print("visualizer={:.4f}ms" .format((time.time()-visualizer_time)*1000))  ####500~650ms

        # img_result = mmcv.imconvert(img_result,'rgb', 'bgr')

        if count>2:
            loop_sum = loop_sum + (time.time()-time_start)
            # print((time.time()-time_start)*1000)
        cv2.imshow('result', img_result)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # bytes_start = time.time()
        # # left_bytes, right_bytes = sc.sendZed(img) #ZED
        # # sc.sendZed(img) #ZED
        # sendZed(img)
        # bytes_time = time.time()-bytes_start
        # if count>2:
            # bytes_time_sum = bytes_time_sum + bytes_time #送信時間の合計

        ch = cv2.waitKey(1)
        if ch == 27 or ch == ord('q') or ch == ord('Q'):
            print("pushed q")
            break

        count += 1
    
    print("ループ回数=%d回" % count)
    print("inference平均時間={:.6f}ms" .format(1000*inf_sum/(count-2)))
    print("visualizer平均時間={:.6f}ms" .format(1000*visualizer_sum/(count-2)))
    # print("send time平均時間={:.6f}ms" .format(1000*bytes_time_sum/(count-2)))
    print("loop平均時間={:.6f}ms" .format(1000*loop_sum/(count-2)))
    camera.release()


if __name__ == '__main__':
    main()
