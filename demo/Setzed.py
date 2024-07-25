# setting_zed
import pyzed
import pyzed.sl as sl

"""
class SetZED:
    def __init__(self) :
        # ZED Miniの初期化
        zed = pyzed.sl.Camera()
        # # カメラの設定
        init_params = sl.InitParameters()
        init_params.camera_resolution = sl.RESOLUTION.HD720
        init_params.camera_fps = 30
        # カメラを開く
        global zed_err
        zed_err = zed.open(init_params)
        if zed_err != sl.ERROR_CODE.SUCCESS:
            exit(1)
"""

# zed の初期設定
def zed_setting():
    # ZED Miniの初期化
    global zed
    zed = pyzed.sl.Camera()
    # # カメラの設定
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720
    init_params.camera_fps = 30
    # カメラを開く
    global zed_err
    zed_err = zed.open(init_params)
    if zed_err != sl.ERROR_CODE.SUCCESS:
        exit(1)
        # ビューの作成
    global zed_right
    global zed_left
    zed_right = sl.Mat()
    zed_left = sl.Mat()