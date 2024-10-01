import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


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


if __name__ == '__main__':
    get_data()