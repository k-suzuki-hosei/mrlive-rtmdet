# send_image_code
import cv2 as cv
import socket
import struct
# from Variable import *

socket_num = 2000
image_size = 1024

# ソケットの作成
def settingSocket():
    global server_address
    global client_socket
    server_address = ('localhost', socket_num)
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# 左右２つのソケットを作成
def settingSockets2():
    global server_address_left, server_address_right
    global client_socket_left, client_socket_right
    server_address_left = ('localhost', 1500)
    server_address_right = ('localhost', 2001)
    client_socket_left = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    client_socket_right = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# zedの画像を送信
def sendZed(image):
    # 右画像を送信
    # udp_image_right = cv.resize(image[:,image.shape[1]//2:,:], (256, 256))  # 右サイズの変更
    udp_image_right = image[:, image.shape[1] // 2:, :]
    cv.imshow("right", udp_image_right)
    img_bytes_right = cv.imencode('.png', udp_image_right)[1].tobytes()

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
    # 左画像を送信
    # udp_image_left = cv.resize(image[:,:image.shape[1]//2,:], (256, 256))  # 左サイズの変更
    udp_image_left = image[:, :image.shape[1]//2, :]
    # height, width, channels = udp_image_left.shape
    # print(f"画像の高さ: {height} ピクセル")
    # print(f"画像の幅: {width} ピクセル")
    # print(f"チャンネル数: {channels}")

    img_bytes_left = cv.imencode('.png', udp_image_left)[1].tobytes()

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
    

    # return len(img_bytes_left), len(img_bytes_right)

# フォルダの画像・動画を送信
def sendNomal(image):
    udp_image = cv.resize(image, (image_size, image_size))  # サイズの変更
    img_bytes = cv.imencode('.png', udp_image)[1].tobytes()

    # print(len(img_bytes))
    # print('frame_rgba_resize', udp_image.shape)
    # print("bytes", len(img_bytes))
    
    chunk_size = 65500  # 最大UDPパケットサイズ
    num =0
    loopnum = len(img_bytes)//chunk_size + 1
    for i in range(0, len(img_bytes), chunk_size):
        chunk = img_bytes[i:i+chunk_size]
        num += 1
        # header = struct.pack('!HH', loopnum, num) #2バイトづつ確保
        header = struct.pack('!BB', loopnum, num) #1バイトづつ確保
        # print(loopnum, num)
        # print('chunk_size', len(chunk))
        data_to_send = header + chunk
        client_socket.sendto(data_to_send, server_address)

    return len(img_bytes)


def closeSocket(zed):
    if zed == True:
        client_socket_left.close()
        client_socket_right.close()
    else:
        client_socket.close()