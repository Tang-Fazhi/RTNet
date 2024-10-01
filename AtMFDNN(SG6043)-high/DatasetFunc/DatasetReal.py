import torch
from torch.utils.data import DataLoader, Dataset, random_split

from utils.normalization import z_score,min_max_scaling
from utils.os_tools import *

torch.manual_seed(0)
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#train dataset,读取一个xlsx文件
class DatasetForReal(Dataset):
    def __init__(self, src_data_path,norm_paras):
        data=read_data_from_excel(src_data_path)
        self.data=data
        if norm_paras is not None:
            _, __, self.re_all = z_score(norm_paras['re_mean'], norm_paras['re_std'],data['re'].tolist())
            _, __, self.alpha_all = z_score(norm_paras['alpha_mean'], norm_paras['alpha_std'], data['alpha'].tolist())
            _, __, self.cl_all = z_score(norm_paras['cl_mean'], norm_paras['cl_std'], data['cl'].tolist())
            _, __, self.cd_all = z_score(norm_paras['cd_mean'], norm_paras['cd_std'], data['cd'].tolist())
        else:
            self.re_all=torch.tensor(data['re'].tolist())
            self.alpha_all=torch.tensor(data['alpha'].tolist())
            self.cl_all=torch.tensor(data['cl'].tolist())
            self.cd_all=torch.tensor(data['cd'].tolist())

        self.re_all=self.re_all.clone().detach().to(device).to(torch.float32).requires_grad_(True)
        self.alpha_all=self.alpha_all.clone().detach().to(device).to(torch.float32).requires_grad_(True)
        self.cl_all=self.cl_all.clone().detach().to(device).to(torch.float32)
        self.cd_all=self.cd_all.clone().detach().to(device).to(torch.float32)

    def __len__(self):
        return len(self.re_all)

    def __getitem__(self, idx):
        re=self.re_all[idx].unsqueeze(0)         #[bs]
        alpha=self.alpha_all[idx].unsqueeze(0)   #[bs]
        cl=self.cl_all[idx].unsqueeze(0)         #[bs]
        cd=self.cd_all[idx].unsqueeze(0)         #[bs]
        return re,alpha,cl,cd    #tuple,[bs,32],[bs],[bs],[bs],[bs]

def get_data_loader(merged_data_path,batch_size,shuffle,norm_paras=None):
    dataset=DatasetForReal(merged_data_path,norm_paras)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)
    return data_loader

def get_norm_paras(data_path):
    def get_z_score_paras(data):
        mean = torch.mean(data)
        std = torch.std(data)
        return mean,std
    def get_min_max_paras(data):
        min_val=torch.min(data)
        max_val=torch.max(data)
        return min_val,max_val

    data=read_data_from_excel(data_path)
    keys = ['re', 'alpha','cl','cd']
    norm_paras={}

    for i in range(4):
        key=keys[i]
        data_i = torch.tensor(data[key], dtype=torch.float)
        mean,std=get_z_score_paras(data_i)
        norm_paras[key + '_mean'] = mean
        norm_paras[key + '_std'] = std

    return norm_paras

class TestDataset(Dataset):
    def __init__(self,data_dir,norm_paras):
        names,paths=get_names_and_paths_in_dir(data_dir)
        re=[]
        for name in names:
            start_idx = name.find('_')
            end_idx = name.rfind('.')
            re_num = int(name[start_idx + 1:end_idx])
            re.append(re_num)
        c=list(zip(re,paths,names))
        sorted_c = sorted(c, key=lambda x: x[0])        # 对a进行顺序排序，并同时对b按照a的顺序排序
        re,paths,names = zip(*sorted_c)
        self.num_of_re=len(re)
        self.re=re
        self.paths=paths
        self.norm_paras=norm_paras
        self.names=names

    def __len__(self):
        return self.num_of_re

    def __getitem__(self,idx):
        # get re num
        name=self.names[idx]
        path=self.paths[idx]
        re_num=self.re[idx]
        data=read_data_from_excel(path)
        alpha=torch.tensor(data['alpha'].tolist(),dtype=torch.float32)
        cl=torch.tensor(data['cl'].tolist(),dtype=torch.float32)
        cd=torch.tensor(data['cd'].tolist(),dtype=torch.float32)
        data_length=len(data['cl'].tolist())
        re=[re_num for i in range(data_length)]
        re=torch.tensor(re,dtype=torch.float32).t()
        if self.norm_paras is not None:
            norm_paras=self.norm_paras
            _, __, re_norm = z_score(norm_paras['re_mean'], norm_paras['re_std'], re)
            _, __, alpha_norm = z_score(norm_paras['alpha_mean'], norm_paras['alpha_std'], alpha)
            _, __, cl_norm = z_score(norm_paras['cl_mean'], norm_paras['cl_std'], cl)
            _, __, cd_norm = z_score(norm_paras['cd_mean'], norm_paras['cd_std'], cd)
        else:
            re_norm,alpha_norm,cl_norm,cd_norm=re,alpha,cl,cd
        return name,re_num,re,alpha,cl,cd,re_norm,alpha_norm,cl_norm,cd_norm

def get_test_loader(test_data_dir,norm_paras=None):
    dataset=TestDataset(test_data_dir,norm_paras)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    return data_loader










    loader=get_data_loader(data_path,4,True,norm_paras)
    # loader=get_data_loader(data_path,4,True,norm_paras=None)

    for re,alpha,cl,cd in loader:
        print(re)
        exit()