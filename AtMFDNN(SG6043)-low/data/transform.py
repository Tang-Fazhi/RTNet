import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

def standardize_norm(data):
    """
    将数据标准化，使每列具有零均值和单位方差
    """
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    standardized_data = (data - mean) / std
    return standardized_data, mean, std


def destandardize_norm(standardized_data, mean, std):
    """
    将标准化数据反归一化回原始数据
    """
    original_data = standardized_data * std + mean
    return original_data


def get_data():
    yh_data = pd.read_excel('data/yh_train.xlsx', sheet_name='Sheet1').to_numpy()
    yh_test_data = pd.read_excel('data/yh_test.xlsx', sheet_name='Sheet1').to_numpy()
    yl_data = pd.read_excel('data/yl.xlsx', sheet_name='Sheet1').to_numpy()

    standardized_data, mean, std = standardize_norm(np.concatenate((yl_data, yh_data, yh_test_data), axis=0))
    yl_size = yl_data.shape[0]
    yh_size = yh_data.shape[0]

    xl = standardized_data[:yl_size, :2]
    yl = standardized_data[:yl_size, 2:]

    xh = standardized_data[yl_size:yl_size+yh_size, 0:2]
    yh = standardized_data[yl_size:yl_size+yh_size, 2:]

    xh_test = standardized_data[yl_size+yh_size:, :2]
    yh_test = standardized_data[yl_size+yh_size:, 2:]

    return xl, yl, xh, yh, xh_test, yh_test, mean, std


def Totensor(xl, yl, xh, yh, xh_test, yh_test, device):
    xl = torch.tensor(xl, requires_grad=True, dtype=torch.float32).to(device)
    yl = torch.tensor(yl, dtype=torch.float32).to(device)
    xh = torch.tensor(xh, requires_grad=True, dtype=torch.float32).to(device)
    yh = torch.tensor(yh, dtype=torch.float32).to(device)

    xh_test = torch.tensor(xh_test, dtype=torch.float32).to(device)
    yh_test = torch.tensor(yh_test, dtype=torch.float32).to(device)

    return xl, yl, xh, yh, xh_test, yh_test


class MyDataset(Dataset):
    def __init__(self, xl, yl):
        self.xl = xl
        self.yl = yl

    def __len__(self):
        return len(self.xl)

    def __getitem__(self, idx):
        sample = self.xl[idx]
        label = self.yl[idx]
        return sample, label

def saveResults(data, path):
    np.save(path, data)

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
    re_150000_train = []
    re_150000_test = []
    re_150000_label = []
    re_200000_train = []
    re_200000_test = []
    re_200000_label = []
    re_300000_train = []
    re_300000_test = []
    re_300000_label = []
    re_400000_train = []
    re_400000_test = []
    re_400000_label = []
    re_500000_train = []
    re_500000_test = []
    re_500000_label = []

    for row in yh_all:
        if row[0] == 500000:
            re_500000_train.append(row)
        elif row[0] == 400000:
            re_400000_train.append(row)
        elif row[0] == 300000:
            re_300000_train.append(row)
        elif row[0] == 200000:
            re_200000_train.append(row)
        elif row[0] == 150000:
            re_150000_train.append(row)
        else:
            re_100000_train.append(row)

    re_500000_train = np.array(re_500000_train)
    re_400000_train = np.array(re_400000_train)
    re_300000_train = np.array(re_300000_train)
    re_200000_train = np.array(re_200000_train)
    re_150000_train = np.array(re_150000_train)
    re_100000_train = np.array(re_100000_train)

    for row in pred:
        if row[0] >= 490000:
            re_500000_test.append(row)
        elif row[0] >= 390000:
            re_400000_test.append(row)
        elif row[0] >= 290000:
            re_300000_test.append(row)
        elif row[0] >= 190000:
            re_200000_test.append(row)
        elif row[0] >= 140000:
            re_150000_test.append(row)
        else:
            re_100000_test.append(row)

    re_500000_test = np.array(re_500000_test)
    re_400000_test = np.array(re_400000_test)
    re_300000_test = np.array(re_300000_test)
    re_200000_test = np.array(re_200000_test)
    re_150000_test = np.array(re_150000_test)
    re_100000_test = np.array(re_100000_test)


    for row in test_label:
        if row[0] >= 490000:
            re_500000_label.append(row)
        elif row[0] >= 390000:
            re_400000_label.append(row)
        elif row[0] >= 290000:
            re_300000_label.append(row)
        elif row[0] >= 190000:
            re_200000_label.append(row)
        elif row[0] >= 140000:
            re_150000_label.append(row)
        else:
            re_100000_label.append(row)

    re_500000_label = np.array(re_500000_label)
    re_400000_label = np.array(re_400000_label)
    re_300000_label = np.array(re_300000_label)
    re_200000_label = np.array(re_200000_label)
    re_150000_label = np.array(re_150000_label)
    re_100000_label= np.array(re_100000_label)

    saveResults(re_500000_train, 'results/data/re_500000.npy')
    saveResults(re_400000_train, 'results/data/re_400000.npy')
    saveResults(re_300000_train, 'results/data/re_300000.npy')
    saveResults(re_200000_train, 'results/data/re_200000.npy')
    saveResults(re_150000_train, 'results/data/re_150000.npy')
    saveResults(re_100000_train, 'results/data/re_100000.npy')

    saveResults(re_500000_test, 'results/data/re_500000_test.npy')
    saveResults(re_400000_test, 'results/data/re_400000_test.npy')
    saveResults(re_300000_test, 'results/data/re_300000_test.npy')
    saveResults(re_200000_test, 'results/data/re_200000_test.npy')
    saveResults(re_150000_test, 'results/data/re_150000_test.npy')
    saveResults(re_100000_test, 'results/data/re_100000_test.npy')

    saveResults(re_500000_label, 'results/data/re_500000_label.npy')
    saveResults(re_400000_label, 'results/data/re_400000_label.npy')
    saveResults(re_300000_test, 'results/data/re_300000_label.npy')
    saveResults(re_200000_label, 'results/data/re_200000_label.npy')
    saveResults(re_150000_label, 'results/data/re_150000_label.npy')
    saveResults(re_100000_label, 'results/data/re_100000_label.npy')

    show_fig(re_500000_train, re_500000_test, re_500000_label, 500000)
    show_fig(re_400000_train, re_400000_test, re_400000_label, 400000)
    show_fig(re_300000_train, re_300000_test, re_300000_label, 300000)
    show_fig(re_200000_train, re_200000_test, re_200000_label, 200000)
    show_fig(re_150000_train, re_150000_test, re_150000_label, 150000)
    show_fig(re_100000_train, re_100000_test, re_100000_label, 100000)


if __name__ == '__main__':
    get_data()