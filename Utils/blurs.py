import cv2
import numpy as np


def CLAHE_blur(src_img,t):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(t, t))
    dst = clahe.apply(src_img)
    return dst


def Sobel_blur(src_img):
    x = cv2.Sobel(src_img, cv2.CV_16S, 1, 0)
    y = cv2.Sobel(src_img, cv2.CV_16S, 0, 1)
    absX = cv2.convertScaleAbs(x)  # 转回uint8
    absY = cv2.convertScaleAbs(y)
    dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    return dst


def hist_blur(src_img):
    equalhist = cv2.equalizeHist(src_img)
    return equalhist


def whitening(src_img):
    mean = src_img.mean()
    std = src_img.std()
    whitening_img = (src_img - mean) / std
    img = (((whitening_img - whitening_img.min()) / (whitening_img.max() - whitening_img.min())) * 255).astype(
        np.uint8)
    return img


def bi_blur(src_img):
    bi_img = cv2.bilateralFilter(src_img, 15, 100, 2)
    return bi_img
