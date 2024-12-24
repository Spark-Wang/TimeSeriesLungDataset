config_train = {}  # 训练信息

# 指定训练相关参数, 根据实验需求设置
config_train['new_train'] = 'new'  # 训练从哪开始, new:从头开始, best:从最好模型开始, last:从上次开始
config_train['epochs'] = 1500  # 训练轮数
config_train['batch_size'] = 1  # 训练批数
config_train['val_batch_size'] = 1  # 验证批数
config_train['lr'] = 1e-3  # 学习率
config_train['optimizer'] = 'adam'  # 优化器
config_train['loss_fn'] = 'vm'  # 损失函数
config_train['early_stop'] = 300  # 验证损失连续未下降多少轮结束训练
config_train['decay_epochs'] = (10, 0.1, 2)  # 多少轮后loss未下降降低loss, loss乘数和下降次数上限, 不用就设None
config_train['train_log'] = 8  # 训练多少轮输出一次
config_train['val_log'] = 8  # 验证多少轮输出一次
config_train['pretrain'] = None  # 预训练模型路径

# 指定训练用的模型和数据集以及训练名称, 每次实验基本都要设置
config_train['model_name'] = 'seqreg'  # 模型名称
config_train['dataset_name'] = 'seq3imgreg'  # 数据集名称
config_train['train_name'] = 'k16seq3reg0sc_seqimgreg' # 本次训练名称

# 其他参数, 基本不需要修改
config_train['checkpoint_path'] = f'./checkpoints/{config_train["train_name"]}' # 断点信息路径
config_train['device'] = 'cuda'  # 选择训练位置
config_train['model_save_path'] = f'./models/{config_train["train_name"]}.pth'  # 模型保存路径
config_train['log_path'] = f'./logs/{config_train["train_name"]}'  # 训练日志路径


config_predict = {}  # 预测信息
config_predict['model_path'] = f'./models/{config_train["train_name"]}.pth'  # 模型保存路径
config_predict['dataset_name'] = 'imgreg'  # 数据集名称
config_predict['predict_mode'] = 'reg3d'  # 预测方式
config_predict['device'] = 'cuda'  # 选择预测位置
config_predict['result_path'] = f'./result/{config_train["train_name"]}newtest'  # 预测结果保存路径
config_predict['predict_number'] = 200  # 预测个数


def main():
    print(config_train)

if __name__ == '__main__':
    main()
