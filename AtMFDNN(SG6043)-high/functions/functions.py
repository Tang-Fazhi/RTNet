import numpy as np
from math import pi
from torch import autograd
import copy
from torch.nn import functional as F
import torch.nn as nn
import time
import matplotlib.pyplot as plt
import os
import pandas as pd
import torch


def trainLModel(model, optimizer, xl, yl, epochs, name):
    best_model = None
    best_result = None
    best_loss = float("inf")
    total_loss = []
    epoch = 0
    model.train()

    while best_loss >= 5e-5 and epoch <= epochs:

        epoch_loss = 0

        xl = xl.float().cuda()
        yl = yl.float().cuda()

        optimizer.zero_grad()
        pred_yl = model(xl)

        loss = F.mse_loss(pred_yl, yl)
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()

        total_loss.append(epoch_loss)

        if epoch_loss <= best_loss:
            best_loss = epoch_loss
            best_model = copy.deepcopy(model)
            best_result = f'{name} Epoch:{epoch}, epoch_loss:{epoch_loss}, best_loss:{best_loss}'

        if epoch % 100 == 0:
            print(f"{name} Epoch {epoch}: epoch_loss:{epoch_loss}, best_loss:{best_loss}")

        epoch += 1

    print(f"{name} Best Result:")
    print(best_result)
    return best_model, total_loss


def to_numpy(input):
    if isinstance(input, torch.Tensor):
        return input.detach().cpu().numpy()
    elif isinstance(input, np.ndarray):
        return input
    else:
        raise TypeError('Unknown type of input, expected torch.Tensor or ' \
                        'np.ndarray, but got {}'.format(type(input)))


def saveResults(data, path):
    np.save(path, data)
    # print('The results has been saved!')


def save_model(model, path):
    torch.save(model.state_dict(), path)
    # print("Save model successfully!")


def load_model(model, path):
    model.load_state_dict(torch.load(path))
    print("Load model successfully!")
    return model


def show_fig(train, test, label, re):
    plt.figure()
    plt.scatter(train[:, 1], train[:, 2], label='Train Points', facecolor='blue', edgecolor='blue', marker='o', s=150)
    plt.scatter(label[:, 1], label[:, 2], label='Exact', facecolor='orange', edgecolor='blue', marker='o', s=150)
    plt.scatter(test[:, 1], test[:, 2], label='Ours', color='red', marker='x', s=60)
    plt.xlabel('alpha')
    plt.ylabel('cl')
    plt.title(f'The result of Cl with Re of {re}')
    plt.legend()
    plt.savefig(f"results/images/re_{re}_cl.png", dpi=600)
    plt.show()

    plt.figure()
    plt.scatter(train[:, 1], train[:, 3], label='Train Points', facecolor='g', edgecolor='g', marker='o', s=150)
    plt.scatter(label[:, 1], label[:, 3], label='Exact', facecolor='orange', edgecolor='g', marker='o', s=150)
    plt.scatter(test[:, 1], test[:, 3], label='Ours', color='red', marker='x', s=60)
    plt.xlabel('alpha')
    plt.ylabel('cd')
    plt.title(f'The result of Cd with Re of {re}')
    plt.legend()
    plt.savefig(f"results/images/re_{re}_cd.png", dpi=600)
    plt.show()


def show_result(pred):

    yh_all = pd.read_excel('data/yh_train.xlsx', sheet_name='Sheet1').to_numpy()
    test_label = pd.read_excel('data/yh_test.xlsx', sheet_name='Sheet1').to_numpy()

    re_100000_train = []
    re_100000_test = []
    re_100000_label = []
    # re_150000_train = []
    # re_150000_test = []
    # re_150000_label = []
    re_200000_train = []
    re_200000_test = []
    re_200000_label = []
    # re_300000_train = []
    # re_300000_test = []
    # re_300000_label = []
    # re_400000_train = []
    # re_400000_test = []
    # re_400000_label = []
    re_500000_train = []
    re_500000_test = []
    re_500000_label = []

    for row in yh_all:
        if row[0] == 500000:
            re_500000_train.append(row)
        # elif row[0] == 400000:
        #     re_400000_train.append(row)
        # elif row[0] == 300000:
        #     re_300000_train.append(row)
        elif row[0] == 200000:
            re_200000_train.append(row)
        # elif row[0] == 150000:
        #     re_150000_train.append(row)
        else:
            re_100000_train.append(row)

    re_500000_train = np.array(re_500000_train)
    # re_400000_train = np.array(re_400000_train)
    # re_300000_train = np.array(re_300000_train)
    re_200000_train = np.array(re_200000_train)
    # re_150000_train = np.array(re_150000_train)
    re_100000_train = np.array(re_100000_train)

    for row in pred:
        if row[0] >= 490000:
            re_500000_test.append(row)
        # elif row[0] >= 390000:
        #     re_400000_test.append(row)
        # elif row[0] >= 290000:
        #     re_300000_test.append(row)
        elif row[0] >= 190000:
            re_200000_test.append(row)
        # elif row[0] >= 140000:
        #     re_150000_test.append(row)
        else:
            re_100000_test.append(row)

    re_500000_test = np.array(re_500000_test)
    # re_400000_test = np.array(re_400000_test)
    # re_300000_test = np.array(re_300000_test)
    re_200000_test = np.array(re_200000_test)
    # re_150000_test = np.array(re_150000_test)
    re_100000_test = np.array(re_100000_test)


    for row in test_label:
        if row[0] >= 490000:
            re_500000_label.append(row)
        # elif row[0] >= 390000:
        #     re_400000_label.append(row)
        # elif row[0] >= 290000:
        #     re_300000_label.append(row)
        elif row[0] >= 190000:
            re_200000_label.append(row)
        # elif row[0] >= 140000:
        #     re_150000_label.append(row)
        else:
            re_100000_label.append(row)

    re_500000_label = np.array(re_500000_label)
    # re_400000_label = np.array(re_400000_label)
    # re_300000_label = np.array(re_300000_label)
    re_200000_label = np.array(re_200000_label)
    # re_150000_label = np.array(re_150000_label)
    re_100000_label= np.array(re_100000_label)

    saveResults(re_500000_train, 'results/data/re_500000.npy')
    # saveResults(re_400000_train, 'results/data/re_400000.npy')
    # saveResults(re_300000_train, 'results/data/re_300000.npy')
    saveResults(re_200000_train, 'results/data/re_200000.npy')
    # saveResults(re_150000_train, 'results/data/re_150000.npy')
    saveResults(re_100000_train, 'results/data/re_100000.npy')

    saveResults(re_500000_test, 'results/data/re_500000_test.npy')
    # saveResults(re_400000_test, 'results/data/re_400000_test.npy')
    # saveResults(re_300000_test, 'results/data/re_300000_test.npy')
    saveResults(re_200000_test, 'results/data/re_200000_test.npy')
    # saveResults(re_150000_test, 'results/data/re_150000_test.npy')
    saveResults(re_100000_test, 'results/data/re_100000_test.npy')

    saveResults(re_500000_label, 'results/data/re_500000_label.npy')
    # saveResults(re_400000_label, 'results/data/re_400000_label.npy')
    # saveResults(re_300000_label, 'results/data/re_300000_label.npy')
    saveResults(re_200000_label, 'results/data/re_200000_label.npy')
    # saveResults(re_150000_label, 'results/data/re_150000_label.npy')
    saveResults(re_100000_label, 'results/data/re_100000_label.npy')

    show_fig(re_500000_train, re_500000_test, re_500000_label, 500000)
    # show_fig(re_400000_train, re_400000_test, re_400000_label, 400000)
    # show_fig(re_300000_train, re_300000_test, re_300000_label, 300000)
    show_fig(re_200000_train, re_200000_test, re_200000_label, 200000)
    # show_fig(re_150000_train, re_150000_test, re_150000_label, 150000)
    show_fig(re_100000_train, re_100000_test, re_100000_label, 100000)


def AtMFDNN_train_model(model, x_low_train, y_low_train, x_high_train, xh_yl, y_high_train, model_save_path):
    # parameters
    lr = 0.001
    # lr = 1e-4
    epochs = 500000
    tolerance = 0.001  # loss容忍度
    lambda_reg = 0.01  # 超参数alpha的正则化系数

    xh_yl = nn.Parameter(xh_yl)

    optimizer = torch.optim.Adam([
        {'params': model.parameters(), 'lr': lr, 'weight_decay': 1e-5},
        {'params': xh_yl, 'lr': lr, 'weight_decay': 1e-5},
    ])

    # 定义投影函数，用来约束alpha的值在0~1之间。
    def project_to_range(value, min_val, max_val):
        return torch.clamp(value, min_val, max_val)

    def train(epoch):
        model.train()

        def closure():
            optimizer.zero_grad()
            y_low_pred, y_high_pred = model(x_low_train, x_high_train)
            loss = (((y_low_pred - y_low_train) ** 2).mean()) + \
                   (((y_high_pred - xh_yl) ** 2).mean()) + \
                   (((xh_yl - y_high_train) ** 2).mean())

            loss.backward()
            return loss

        loss = optimizer.step(closure)
        loss_value = loss.item() if not isinstance(loss, float) else loss
        # 投影超参数到指定范围以实现约束
        model.alpha.data = project_to_range(model.alpha.data, 0.0, 1.0)
        if epoch % 100 == 0:
            print(f'epoch {epoch}: loss {loss_value:.6f}')
        return loss_value

    print('start training...')
    tic = time.time()
    best_epoch, best_loss = 0, 1000000.0
    for epoch in range(1, epochs + 1):
        loss = train(epoch)
        if loss < best_loss:
            save_model(model, os.path.join(model_save_path, 'AtMFDNN.pth'))
            best_loss = loss
        if loss <= tolerance:
            save_model(model, os.path.join(model_save_path, 'AtMFDNN.pth'))
            break
    toc = time.time()
    print(f'total training time: {toc - tic}')
    print("Trained model successfully!")


def train_model(model, x_low_train, y_low_train, x_high_train, y_high_train, model_save_path):
    ## parameters
    # lr = 0.001
    lr = 1e-4
    epochs = 500000
    tolerance = 1e-3  # loss容忍度
    lambda_reg = 0.01  # 超参数alpha的正则化系数

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 定义投影函数，用来约束alpha的值在0~1之间。
    def project_to_range(value, min_val, max_val):
        return torch.clamp(value, min_val, max_val)

    def train(epoch):
        model.train()

        def closure():
            optimizer.zero_grad()
            y_low_pred, y_high_pred = model(x_low_train, x_high_train)
            loss = (((y_low_pred - y_low_train) ** 2).mean()) + \
                   (((y_high_pred - y_high_train) ** 2).mean())
            loss.backward()
            return loss

        loss = optimizer.step(closure)
        loss_value = loss.item() if not isinstance(loss, float) else loss
        # 投影超参数到指定范围以实现约束
        model.alpha.data = project_to_range(model.alpha.data, 0.0, 1.0)
        if epoch % 100 == 0:
            print(f'epoch {epoch}: loss {loss_value:.6f}')
        return loss_value

    print('start training...')
    tic = time.time()
    best_epoch, best_loss = 0, 1000000.0
    for epoch in range(1, epochs + 1):
        loss = train(epoch)
        if loss < best_loss:
            save_model(model, os.path.join(model_save_path, 'AtMFDNN_final.pth'))
            best_loss = loss
        if loss <= tolerance:
            save_model(model, os.path.join(model_save_path, 'AtMFDNN_final.pth'))
            break
    toc = time.time()
    print(f'total training time: {toc - tic}')
    print("Trained model successfully!")


def test_model(model, x_tensor, y_low_tensor, y_high_tensor, x_low_train, y_low_train, x_high_train, y_high_train):
    # 1. 整个区间上的预测
    y_low_pred, y_high_pred = model(x_tensor, x_tensor)
    x_tensor = to_numpy(x_tensor)
    y_low_tensor = to_numpy(y_low_tensor)
    y_high_tensor = to_numpy(y_high_tensor)
    y_low_pred = to_numpy(y_low_pred)
    y_high_pred = to_numpy(y_high_pred)
    result = np.concatenate((x_tensor, y_low_tensor, y_high_tensor, y_low_pred, y_high_pred), axis=1)
    # header="x, y_low, y_high, y_low_pred, y_high_pred"
    # np.savetxt('MFDNN2_LinearCountiousFunc_TowFidelity.dat', result,header=header,comments='')
    print("Test finished and result saved successfully!")
    return result


if __name__ == '__main__':
    # functionName = 'LC'
    # x, x_low_star, x_high_star, y_low, y_low_star, y_high, y_high_star = get_data(functionName)

    import torch

    # 创建一个张量
    tensor = torch.tensor([1, 2, 3, 4, 5, 6])
    tensor2 = torch.tensor([1, 1.1, 10])
    for i in tensor2:
        pass

    # 获取特定值的索引
    index = torch.nonzero(torch.eq(tensor, 3))
    print(index)

