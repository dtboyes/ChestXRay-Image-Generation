import cv2
import os
IMAGES_DIR = "VinDRCXR/jpg/train"

for image in os.listdir(IMAGES_DIR):
    print(image)
    img = cv2.imread(f"{IMAGES_DIR}/{image}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    denoised = cv2.medianBlur(src=img, ksize=7)
  
    edge = cv2.Canny(denoised, 0, 255)
    cv2.imwrite(f"VinDRCXR/jpg/edges/train/{image}", edge)
    cv2.imwrite(f"VinDRCXR/jpg/binary/train/{image}", denoised)
    

IMAGES_DIR = "VinDRCXR/jpg/test"

for image in os.listdir(IMAGES_DIR):
    print(image)
    img = cv2.imread(f"{IMAGES_DIR}/{image}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    denoised = cv2.medianBlur(src=img, ksize=7)
  
    edge = cv2.Canny(denoised, 0, 255)
    cv2.imwrite(f"VinDRCXR/jpg/edges/test/{image}", edge)
    cv2.imwrite(f"VinDRCXR/jpg/binary/test/{image}", denoised)