import datetime
import torch
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import config_train as config
import factory


new_train = config['new_train']
epochs = config['epochs']
batch_size = config['batch_size']
val_batch_size = config['val_batch_size']
lr = config['lr']
coptimizer = config['optimizer']
closs_fn = config['loss_fn']
early_stop = config['early_stop']
decay_epochs = config['decay_epochs']
train_log = config['train_log']
val_log = config['val_log']
pretrain = config['pretrain']
model_name = config['model_name']
dataset_name = config['dataset_name']
train_name = config['train_name']
checkpoint_path = config['checkpoint_path']
cdevice = config['device']
model_save_path = config['model_save_path']
log_path = config['log_path']

if not os.path.exists(log_path):
    os.mkdir(log_path)
    new_train = 'new'

# 记录训练信息
class Logger():
    def __init__(self):
        if new_train == 'new':
            print('你TM看好了这是从头训练, 现在马上停还来得及')
            self.begin_epoch = 0  # 从第几轮开始训练
            self.early = 0  # 多少轮未下降
            self.best_epoch = 0  # 最好模型所在轮数
            self.best_train_number = 0  # 最好模型对应训练总训练数据量
            self.best_val_number = 0  # 最好模型对应训练总验证数据量
            self.best_loss = float('inf')  # 验证集最好损失值
            self.train_loss = np.array([])  # 记录训练损失
            self.epoch_train_loss = np.array([])  # 记录训练损失(每轮)
            self.val_loss = np.array([])  # 记录验证损失
            self.epoch_val_loss = np.array([])  # 记录验证损失(每轮)
            if not os.path.exists(checkpoint_path):
                os.mkdir(checkpoint_path)
        else:
            train_process = np.load(f'{checkpoint_path}/train_progress.npy')
            self.early = train_process[1]
            self.best_epoch = train_process[2]
            self.best_train_number = train_process[3]
            self.best_val_number = train_process[4]
            self.best_loss = np.load(f'{checkpoint_path}/best_loss.npy')
            self.train_loss = np.load(f'{checkpoint_path}/train_loss.npy')
            self.epoch_train_loss = np.load(f'{checkpoint_path}/epoch_train_loss.npy')
            self.val_loss = np.load(f'{checkpoint_path}/val_loss.npy')
            self.epoch_val_loss = np.load(f'{checkpoint_path}/epoch_val_loss.npy')
            if new_train == 'last':
                self.begin_epoch = train_process[0]
            elif new_train == 'best':
                self.early = 0
                self.begin_epoch = self.best_epoch
                self.train_loss = self.train_loss[:self.best_train_number]
                self.epoch_train_loss = self.epoch_train_loss[:self.best_epoch]
                self.val_loss = self.val_loss[:self.best_val_number]
                self.epoch_val_loss = self.epoch_val_loss[:self.best_epoch]
                

        if new_train == 'new':
            self.f = open(f'{log_path}/train_loss.log', 'w+', encoding='utf-8', buffering=1)
        else:
            self.f = open(f'{log_path}/train_loss.log', 'a+', encoding='utf-8', buffering=1)

    # 每轮开始时
    def begin_epoch_train(self, epoch):
        time = datetime.datetime.now().strftime('%y-%m-%d %H:%M:%S')
        print('--------第{}轮训练开始--------{}--------'.format(epoch + 1, time))
        self.f.write('--------第{}轮训练开始--------{}--------\n'.format(epoch + 1, time))
        self.train_epoch_loss = 0.  # 每轮平均训练损失
        self.val_epoch_loss = 0.  # 每轮平均验证损失

    # 一个数据训练之后
    def end_once_train(self, number, loss):
        self.train_loss = np.append(self.train_loss, loss)
        self.train_epoch_loss += loss
        if (number+1) % train_log == 0:
            print(f'训练次数:{number+1},Loss:{loss}')
            self.f.write(f'训练次数:{number+1},Loss:{loss}\n')

    # 一个数据验证之后
    def end_once_val(self, number, loss):
        self.val_loss = np.append(self.val_loss, loss)
        self.val_epoch_loss += loss
        if (number+1) % val_log == 0:
            print(f'验证次数:{number+1},Loss:{loss}')
            self.f.write(f'验证次数:{number+1},Loss:{loss}\n')

    # 每轮结束时
    def end_epoch(self, epoch, train_number, val_number):
        if train_number != 0:
            self.train_epoch_loss = self.train_epoch_loss/(train_number+1)
            self.epoch_train_loss = np.append(self.epoch_train_loss, self.train_epoch_loss)
            print(f'本轮训练损失平均值:{self.train_epoch_loss}, 训练个数{train_number+1}')
            self.f.write(f'本轮训练损失平均值:{self.train_epoch_loss}, 训练个数{train_number+1}\n')
        if val_number != 0:
            self.val_epoch_loss = self.val_epoch_loss/(val_number+1)
            self.epoch_val_loss = np.append(self.epoch_val_loss, self.val_epoch_loss)
            print(f'本轮验证损失平均值:{self.val_epoch_loss}, 验证个数{val_number+1}')
            self.f.write(f'本轮验证损失平均值:{self.val_epoch_loss}, 验证个数{val_number+1}\n')
        save_model, stopit = True, False
        if val_number != 0:
            if self.val_epoch_loss <= self.best_loss:
                self.best_loss = self.val_epoch_loss
                self.best_epoch = epoch
                self.best_train_number = len(self.train_loss)
                self.best_val_number = len(self.val_loss)
                self.early = 0
            else:
                save_model = False
                self.early += 1
                if self.early >= early_stop:
                    stopit = True
            print(f'目前验证集最好损失:{self.best_loss}')
            self.f.write(f'目前验证集最好损失:{self.best_loss}\n')
            if stopit:
                print(f'验证集loss经过{early_stop}轮仍未下降, 训练提前停止')
                self.f.write(f'验证集loss经过{early_stop}轮仍未下降, 训练提前停止\n')

        plt.plot([i for i in range(len(self.train_loss))], self.train_loss)
        plt.title('train_loss')
        plt.tight_layout()
        plt.savefig(f'{log_path}/train_loss.png')
        plt.close('all')
        plt.plot([i for i in range(len(self.epoch_train_loss))], self.epoch_train_loss)
        plt.title('epoch_train_loss')
        plt.tight_layout()
        plt.savefig(f'{log_path}/epoch_train_loss.png')
        plt.close('all')
        np.save(f'{checkpoint_path}/train_progress.npy', np.array([epoch + 1, self.early, self.best_epoch + 1, self.best_train_number, self.best_val_number]))
        np.save(f'{checkpoint_path}/epoch_train_loss.npy', self.epoch_train_loss)
        np.save(f'{checkpoint_path}/train_loss.npy', self.train_loss)
        
        if val_number != 0:
            plt.plot([i for i in range(len(self.val_loss))], self.val_loss)
            plt.title('val_loss')
            plt.tight_layout()
            plt.savefig(f'{log_path}/val_loss.png')
            plt.close('all')
            plt.plot([i for i in range(len(self.epoch_val_loss))], self.epoch_val_loss)
            plt.title('epoch_val_loss')
            plt.tight_layout()
            plt.savefig(f'{log_path}/epoch_val_loss.png')
            plt.close('all')
            np.save(f'{checkpoint_path}/best_loss.npy', self.best_loss)
            np.save(f'{checkpoint_path}/val_loss.npy', self.val_loss)
            np.save(f'{checkpoint_path}/epoch_val_loss.npy', self.epoch_val_loss)
        return save_model, stopit

    # 整个训练结束时
    def end_train(self):
        time = datetime.datetime.now().strftime('%y-%m-%d %H:%M:%S')
        print('--------{}--------'.format(time))
        self.f.write('--------{}--------'.format(time))
        self.f.close()

def train():
    logger = Logger()
    if new_train == 'new':
        if pretrain is not None:
            net = torch.load(pretrain)
        else:
            net = factory.get_net(model_name)
    elif new_train == 'best':
        net = torch.load(model_save_path)
    elif new_train == 'last':
        net = torch.load(f'{checkpoint_path}/model.pth')
    optimizer = factory.get_optimizer(coptimizer, net, lr)
    loss_fn = factory.get_loss_fn(closs_fn)
    if decay_epochs is not None:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=decay_epochs[1])
    train_loader, val_loader = factory.get_dataloader(dataset_name, batch_size, val_batch_size)
    device = torch.device(cdevice)
    net.to(device)
    loss_fn.to(device)
    loss_epochs = 0
    loss_number = 0
    for epoch in range(logger.begin_epoch, epochs):
        logger.begin_epoch_train(epoch)

        net.train()
        for train_number, data in enumerate(train_loader):
            image, time_diff, mask = data
            if isinstance(image, list):
                for i in range(len(image)):
                    image[i] = image[i].to(device)
            else:
                image = image.to(device)
            time_diff = time_diff.to(device)
            mask = mask.to(device)

            pred_image, pred_mask, flow = net(image, mask, True)
            # print(image.shape, mask.shape, pred_image.shape, pred_mask.shape, len(flow), time_diff.shape)
            loss = loss_fn(image, mask, pred_image, pred_mask, flow, time_diff)
            # print(loss.item())
            # exit()
            loss.requires_grad_(True)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            logger.end_once_train(train_number, loss.item())

        with torch.no_grad():
            net.eval()
            val_number = 0
            for val_number, data in enumerate(val_loader):
                image, time_diff, mask = data
                if isinstance(image, list):
                    for i in range(len(image)):
                        image[i] = image[i].to(device)
                else:
                    image = image.to(device)
                time_diff = time_diff.to(device)
                mask = mask.to(device)
                pred_image, pred_mask, flow = net(image, mask.type(torch.float32), True)
                loss = loss_fn(image, mask, pred_image, pred_mask, flow, time_diff)
                logger.end_once_val(val_number, loss.item())

        save_model, stopit = logger.end_epoch(epoch, train_number, val_number)
        torch.save(net, f'{checkpoint_path}/model.pth')
        if stopit:
            break
        elif save_model:
            torch.save(net, model_save_path)
            loss_epochs = 0
        elif (decay_epochs is not None) and (loss_number < decay_epochs[2]):
            loss_epochs += 1
            if loss_epochs >= decay_epochs[0]:
                loss_epochs = 0
                loss_number += 1
                scheduler.step()


def main():
    train()


if __name__ == '__main__':
    main()
