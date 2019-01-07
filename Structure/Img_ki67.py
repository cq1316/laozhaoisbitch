import cv2
import numpy as np

import multi_line_ki67.Utils.blurs as blurs
import torch


# 1:normalize_data or init_adjust
# 2:get_crop_img
class Img_ki67():
    def __init__(self, infor_dict):
        self.ki67_class = infor_dict["class"]
        self.infor = infor_dict
        src_img = np.array(infor_dict["src_img"])
        self.t = 255
        self.src_img = (((src_img - src_img.min()) / (src_img.max() - src_img.min())) * self.t).astype(
            np.uint8)
        self.id = infor_dict["id"]
        self.roi_img = self.src_img[self.infor["min_y"] - 3:self.infor["max_y"] + 4,
                       self.infor["min_x"] - 3:self.infor["max_x"] + 4]

        self.mrmr_feature = []

    def get_class(self):
        return self.ki67_class

    def get_id(self):
        return self.id

    def make_channel3_resize(self, resize_length):
        blur_img = blurs.Sobel_blur(self.src_img)[self.infor["min_y"] - 3:self.infor["max_y"] + 4,
                   self.infor["min_x"] - 3:self.infor["max_x"] + 4]
        clahe_img = blurs.CLAHE_blur(self.roi_img, t=8)
        roi_resize_img = cv2.resize(self.roi_img, (resize_length, resize_length))
        blur_resize_img = cv2.resize(blur_img, (resize_length, resize_length))
        clahe_resize_img = cv2.resize(clahe_img, (resize_length, resize_length))
        self.transform_tensor = torch.FloatTensor([roi_resize_img, blur_resize_img, clahe_resize_img])

    def get_blur_img(self, img):
        blur_img = blurs.Sobel_blur(img)
        CLAHE_img = blurs.hist_blur(img)
        return blur_img, CLAHE_img

    def set_vgg_pool_feature(self, b1, b2, b3, b4, b5):
        self.vgg_feature = [b1, b2, b3, b4, b5]

    def get_vgg_pool_feature(self):
        return self.vgg_feature

    def set_mrmr_feature(self, mrmr_feature, b):
        self.mrmr_feature = self.mrmr_feature + mrmr_feature
        if b == 4:
            self.mrmr_feature = np.array(self.mrmr_feature)

    def get_mrmr_feature(self):
        return self.mrmr_feature

    def get_3channel_tensor(self):
        return self.transform_tensor
