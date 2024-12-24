import torch
import numpy as np
import os
from sklearn import metrics
import torch.nn.functional as F

import sys
sys.path.append('..')
from utils import save_to_excel

binary = True

class Predicter():
    def __init__(self, predict_path):
        print('预测死大头')
        self.path = predict_path
        if not os.path.exists(self.path):
            os.mkdir(self.path)
        self.pred = []
        self.label = []
        self.save_data = []
        self.prob = []
        header = ['number', 'name', 'pred', 'label']
        self.save_data.append(header)
        
    def predict_once(self, number, net, image, ground_truth, name):
        pred = net(image)
        pred = pred.cpu()
        ground_truth = ground_truth.cpu()
        name = name[0]
        pred, mask = pred[0], ground_truth[0]
        # np.save(f'{self.path}/{name}.npy', pred)
        self.prob.append(F.softmax(pred, dim=0))
        pred = torch.argmax(pred, dim=0)
        mask = torch.argmax(mask, dim=0)
        self.pred.append(pred.item())
        self.label.append(mask.item())
        row_data = [number+1, name, pred.item(), mask.item()]
        self.save_data.append(row_data)
        print(f'{number+1}已完成')

    def end_predict(self, number):
        row_data = ['accuracy', 'precision', 'recall', 'f1_score', 'auc']
        self.save_data.append(row_data)
        accuracy = metrics.accuracy_score(self.label, self.pred)
        precision = metrics.precision_score(self.label, self.pred, average='macro')
        recall = metrics.recall_score(self.label, self.pred, average='macro')
        f1_score = metrics.f1_score(self.label, self.pred, average='macro')
        self.prob = np.stack(self.prob, axis=0)
        if binary:
            auc = metrics.roc_auc_score(self.label, self.prob[:, 1])
        else:
            auc = metrics.roc_auc_score(self.label, self.prob, average='macro', sample_weight=None, max_fpr=None, multi_class='ovo', labels=None)
        self.save_data.append([accuracy, precision, recall, f1_score, auc])
        save_to_excel(self.save_data, f'{self.path}/result.xlsx')
        print('预测恩德')
