import numpy as np
import os
import json
import matplotlib.pyplot as plt

def get_samples(path, resize_length):
    samples = []
    for root, dirs, files in os.walk(path):
        for file in files:
            f = open(root + "/" + file)
            infor_dict = json.load(f)
            f.close()
            src_img = np.array(infor_dict["src_img"])
            if src_img.shape == (512,512):
                samples.append(src_img)
    return samples


def get_data_set(path, resize_length):
    train_path = path + "/train"
    test_path = path + "/test"
    train_neg_set = get_samples(path=train_path + "/neg",
                                resize_length=resize_length)
    train_pos_set = get_samples(train_path + "/pos",
                                resize_length=resize_length)
    test_pos_set = get_samples(test_path + "/pos",
                               resize_length=resize_length)
    test_neg_set = get_samples(test_path + "/neg",
                               resize_length=resize_length)
    return train_neg_set, train_pos_set, test_pos_set, test_neg_set


def train_norm(train_set):
    dataset = np.array(train_set)
    mean = dataset.mean(axis=0)
    std = dataset.std(axis=0)
    return mean, std


def make_same_dis(mean, std, dataset):
    new_set = []
    for src_img in dataset:
        norm_img = (src_img - mean) / (std+0.00001)
        new_set.append(norm_img)
    return new_set


def img_process(train_set, test_set):
    mean, std = train_norm(train_set)
    new_train = make_same_dis(mean, std, train_set)
    for src_img,norm_img in zip(train_set,new_train):
        plt.figure(1)
        plt.imshow(src_img,'gray')
        plt.figure(2)
        plt.imshow(norm_img,'gray')
        plt.show()
    new_test = make_same_dis(mean, std, test_set)


path = "E:\\Medical\\切分数据集\\4SeriesNosmall\\T2"
# get_data_set 正确
train_neg_set, train_pos_set, test_pos_set, test_neg_set = get_data_set(path=path,
                                                                        resize_length=20)
total_train_set = train_neg_set + train_pos_set
total_test_set = test_neg_set + test_pos_set
img_process(total_train_set,total_test_set)
