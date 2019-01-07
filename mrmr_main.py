# -*- coding: utf-8 -*
import math
import random
import torch
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import roc_curve, auc
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm
from ki67_mrmr_v2.Utils.make_batch import get_data_set
from ki67_mrmr_v2.Models.feature_extracter import Vgg16_pool
from ki67_mrmr_v2.Models.mrmr_classifier import Mrmr_classifier
from ki67_mrmr_v2.Utils.mrmr import get_mrmr
import sys
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)


def train_batch(epoch, train_set, optimizer, model, batchsize):
    batch_num = math.ceil(len(train_set) / batchsize)
    train_loss = 0
    model.train(mode=True)
    for n in range(0, batch_num):
        targets = []
        mrmr = []
        for img_ki67 in train_set[n * batchsize:(n + 1) * batchsize]:
            # 取出mrmr特征
            mrmr_feature = img_ki67.get_mrmr_feature()
            mrmr_feature = Variable(torch.FloatTensor(mrmr_feature).unsqueeze(0))
            mrmr.append(mrmr_feature)
            target = img_ki67.get_class()
            targets.append(target)
        mrmr = torch.cat(mrmr).cuda()
        targets = Variable(torch.LongTensor(targets)).cuda()
        optimizer.zero_grad()
        output = model(mrmr)
        loss = F.cross_entropy(input=output, target=targets)
        clip_grad_norm(model.parameters(), 10)
        train_loss += loss.data[0]
        loss.backward()
        optimizer.step()
    logging.info('Train Epoch: %d loss : %f' % (epoch, train_loss / len(train_set)))


def test(test_set, model, print_id=False):
    y_t = []
    y_p = []
    model.eval()
    for img_ki67 in test_set:
        # 取出mrmr特征
        mrmr_feature = img_ki67.get_mrmr_feature()
        mrmr_feature = Variable(torch.FloatTensor(mrmr_feature).unsqueeze(0)).cuda()
        target = img_ki67.get_class()
        y_t.append(target)
        output = model(mrmr_feature)
        y_p.append(output.data[0][1])
    return y_t, y_p


def evaluate_on_trainset(y_t, y_p):
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    for t, p in zip(y_t, y_p):
        if p >= 0.5:
            if t == 1:
                TP = TP + 1
            else:
                FP = FP + 1
        else:
            if t == 1:
                FN = FN + 1
            else:
                TN = TN + 1
    acc = (TP + TN) / (FP + FN + TP + TN)
    sen = TP / (TP + FN + 0.001)
    spe = TN / (TN + FP + 0.001)
    ppv = TP / (TP + FP + 0.001)
    npv = TN / (TN + FN + 0.001)
    logging.info("tp:%d tn:%d fp:%d fn:%d" % (TP, TN, FP, FN))
    logging.info('SEN:%f SPE:%f PPV:%f NPV:%f Acc:%f' % (sen, spe, ppv, npv, acc))
    return acc


def get_batch(i, train_pos_set, length):
    if length + i <= len(train_pos_set):
        pos_set = train_pos_set[i:i + length]
        i = i + length
    else:
        pos_set_tail = train_pos_set[i:]
        i = i + length - len(train_pos_set)
        pos_set_head = train_pos_set[:i]
        pos_set = pos_set_tail + pos_set_head
    return pos_set, i


if __name__ == "__main__":
    path = sys.argv[1]
    # path = "E:\\Medical\\split_dataset\\4SeriesNosmall\\DWI"
    logging.info("get data set")
    # get_data_set 正确
    train_neg_set, train_pos_set, test_pos_set, test_neg_set = get_data_set(path=path,
                                                                            resize_length=80)
    total_train_set = train_neg_set + train_pos_set
    total_test_set = test_neg_set + test_pos_set
    length = len(train_neg_set)
    logging.info("get data set completed")
    logging.info("get mrmr feature")
    # vgg_16_bn 模型正确，获取vgg池化特征
    vgg_feature_selector = Vgg16_pool().cuda()
    # 获取mrmr 并展开 正确
    num_feature = [32, 64, 128, 256, 256]
    get_mrmr(vgg_model=vgg_feature_selector,
             train_set=total_train_set,
             test_set=total_test_set,
             num_features=num_feature,
             batch_size=16)
    logging.info("mrmr completed")
    # pool_feature_model 分类模型正确
    logging.info("init model")
    model = Mrmr_classifier(num_class=2, num_mrmr=sum(num_feature), num_hidden=512).cuda()
    classifier_optimizer = optim.SGD(model.classifier.parameters(),
                                     lr=0.04,
                                     momentum=0.9,
                                     weight_decay=0.01)
    i = 0
    best_sen = 0
    best_spe = 0
    times = 0
    accs = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for epoch in range(1, 3000):
        if epoch % 2 == 0:
            for param_group in classifier_optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * 0.94
        logging.info("training")
        pos_set, i = get_batch(i=i, train_pos_set=train_pos_set, length=length)
        train_set = train_neg_set + pos_set
        random.shuffle(train_set)

        train_batch(epoch=epoch,
                    train_set=train_set,
                    optimizer=classifier_optimizer,
                    model=model,
                    batchsize=16)
        logging.info("Testing trainset")
        y_t1, y_p1 = test(test_set=total_train_set, model=model)

        fpr, tpr, threshold = roc_curve(y_t1, y_p1, pos_label=1)
        roc_auc_tr = auc(fpr, tpr)
        logging.info("auc: %f" % roc_auc_tr)
        acc = evaluate_on_trainset(y_t1, y_p1)
        acc_mean = sum(accs) / 10
        if acc > acc_mean:
            logging.info("Testing testset")
            y_t2, y_p2 = test(test_set=total_test_set, model=model)
            fpr, tpr, threshold = roc_curve(y_t2, y_p2, pos_label=1)
            roc_auc_tr = auc(fpr, tpr)
            logging.info("auc: %f" % roc_auc_tr)
            evaluate_on_trainset(y_t2, y_p2)
            times = 0
        else:
            times = times + 1
            if times > 10:
                break
        accs.pop(0)
        accs.append(acc)
