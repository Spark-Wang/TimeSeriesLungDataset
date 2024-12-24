import torch

from config import config_predict as config
import factory
import numpy as np


model_path = config['model_path']  # 模型保存路径
dataset_name = config['dataset_name']  # 数据集名称
predict_mode = config['predict_mode']  # 预测方式
cdevice = config['device']  # 选择预测位置
result_path = config['result_path']  # 预测结果保存路径
predict_number = config['predict_number']  # 预测个数

def predict():
    predicter = factory.get_predicter(predict_mode,  result_path)
    net = torch.load(model_path, map_location='cpu')
    # net = None
    device = torch.device(cdevice)
    _, predict_loader = factory.get_dataloader(dataset_name, None, 1, with_name=True)
    net.to(device)
    net.eval()
    with torch.no_grad():
        for number, (name, image, _, ground_truth) in enumerate(predict_loader):
            if 'liai' not in name[0]:
                print(number + 1)
                continue
            if isinstance(image, list):
                for i in range(len(image)):
                    image[i] = image[i].to(device)
            else:
                image = image.to(device)
            ground_truth = ground_truth.to(device)
            predicter.predict_once(number, net, image, ground_truth, name)
            if number+1 >= predict_number:
                break
    predicter.end_predict(number)


def main():
    predict()


if __name__ == '__main__':
    main()
