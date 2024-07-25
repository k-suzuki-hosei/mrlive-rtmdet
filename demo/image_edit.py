# bylinar
import numpy as np
import cv2
import os

def image_Edit(img, re_scale=2, gaussian: float=0.3, sobel: float=0.3, laplacian_num: float=0.3):
    height, width = img.shape[:2]
    re_img = cv2.resize(img, (width*re_scale, height*re_scale))

    # バイリニア補間
    bylinea_img = cv2.resize(img, (width*re_scale, height*re_scale), interpolation=cv2.INTER_LINEAR)

    # ガウシアンフィルタリング
    gaussian_blurred = cv2.GaussianBlur(bylinea_img, (5, 5), 0)
    # edit_img = cv2.addWeighted(gaussian_blurred, gaussian, re_img, 1-gaussian, 0, dtype=cv2.CV_64F).astype(np.uint8)  # gaussian_blurred と re_img のdtypeが異なってたら嫌。dtypeを指定し、astypeで8bitを指定

    # Sobelフィルタによるエッジ強調
    sobelx = cv2.Sobel(gaussian_blurred, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gaussian_blurred, cv2.CV_64F, 0, 1, ksize=3)
    sobel_combined = cv2.magnitude(sobelx, sobely)
    sobel_combined = cv2.convertScaleAbs(sobel_combined)
    # edit_img = cv2.addWeighted(sobel_combined, sobel, edit_img, 1-sobel, 0, dtype=cv2.CV_64F).astype(np.uint8)

    # ラプラシアンフィルタによるシャープニング
    laplacian = cv2.Laplacian(gaussian_blurred, cv2.CV_64F)
    sharpened = cv2.convertScaleAbs(laplacian)
    # edit_img = cv2.addWeighted(sharpened, laplacian_num, edit_img, 1-laplacian_num, 0, dtype=cv2.CV_64F).astype(np.uint8)
    
    bylinea_img[:,:,2] = np.clip(bylinea_img[:,:,2] + gaussian_blurred[:,:,2], 0, 255)
    bylinea_img[:,:,0] = np.clip(bylinea_img[:,:,0] + sobel_combined[:,:,0], 0, 255)
    bylinea_img[:,:,1] = np.clip(bylinea_img[:,:,1] + sharpened[:,:,1], 0, 255)
    edit_img = cv2.addWeighted(bylinea_img, 0.3, re_img, 0.7, 0)
    return edit_img

# re_scale = 2
# file_name = 'human2.jpg'
# try:
#     open("./images/"+file_name)
#     img = cv2.imread("./images/"+file_name)
# except FileNotFoundError as e:
#         print(f"ファイルが見つかりませんでした: {e}")

# height, width = img.shape[:2]
# cv2.imshow('before', img)

# img2 = cv2.resize(img, (width*re_scale, height*re_scale))
# cv2.imshow('only resize', img2)

# # バイリニア補間
# img = cv2.resize(img, (width*re_scale, height*re_scale), interpolation=cv2.INTER_LINEAR)
# cv2.imshow('byliner', img)

# # ガウシアンフィルタリング
# gaussian_blurred = cv2.GaussianBlur(img, (5, 5), 0)
# cv2.imshow('gaussioan', gaussian_blurred)

# # Sobelフィルタによるエッジ強調
# sobelx = cv2.Sobel(gaussian_blurred, cv2.CV_64F, 1, 0, ksize=3)
# sobely = cv2.Sobel(gaussian_blurred, cv2.CV_64F, 0, 1, ksize=3)
# sobel_combined = cv2.magnitude(sobelx, sobely)
# sobel_combined = cv2.convertScaleAbs(sobel_combined)
# cv2.imshow('sobel', sobel_combined)

# # ラプラシアンフィルタによるシャープニング
# laplacian = cv2.Laplacian(gaussian_blurred, cv2.CV_64F)
# sharpened = cv2.convertScaleAbs(laplacian)
# cv2.imshow('laplacian', sharpened)

# img[:,:,0] = np.clip(img[:,:,0] + sobel_combined[:,:,0], 0, 255)
# img[:,:,1] = np.clip(img[:,:,1] + sharpened[:,:,1], 0, 255)
# img[:,:,2] = np.clip(img[:,:,2] + gaussian_blurred[:,:,2], 0, 255)
# img = cv2.addWeighted(img, 0.3, img2, 0.7, 0)

# if os.path.exists('./outputs'):
#     cv2.imwrite('./outputs/edit_'+file_name, img)

# cv2.imshow('after', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()