
import time
import random
from DatasetFunc.DatasetReal import src_data_show_only_scatter

random.seed(111)
from utils.os_tools import *
from utils.model_operating import load_model

data_dir='../RealData/sg6043'
hf_dir=pthjoin(data_dir,'UIUC-HF')
hf_data=pthjoin(hf_dir,'uiuc.xlsx')
mf_dir=pthjoin(data_dir,'Xfoil-N9-MF')
lf_dir=pthjoin(data_dir,'Xfoil-N5-LF')

re_nums=[50000,100000,200000,500000,1000000]

def modify_keys():
    not_pred_hf_dir='../RealData/sg6043/not-pre-processed-data/UIUC-HF'
    not_pred_mf_dir='../RealData/sg6043/not-pre-processed-data/Xfoil-N9-MF'
    not_pred_lf_dir='../RealData/sg6043/not-pre-processed-data/Xfoil-N5-LF'
    pred_hf_dir='../RealData/sg6043/pre-processed-data/UIUC-HF'
    pred_mf_dir='../RealData/sg6043/pre-processed-data/Xfoil-N9-MF'
    pred_lf_dir='../RealData/sg6043/pre-processed-data/Xfoil-N5-LF'

    all_folders=[not_pred_hf_dir,not_pred_mf_dir,not_pred_lf_dir,pred_hf_dir,pred_mf_dir,pred_lf_dir]

    for folder in all_folders:
        names,paths=get_names_and_paths_in_dir(folder)
        for path in paths:
            data=read_data_from_excel(path)
            old_keys = list(data.keys())
            new_keys = {}
            for key in old_keys:
                new_keys[key] = key.lower()
            data = data.rename(columns=new_keys)
            data.to_excel(path, index=False)

def split_data_by_ratio(src_data_path,split_ratio,train_data_path,test_data_path):
    print(f"start split dataset...")
    data = read_data_from_excel(src_data_path)
    old_keys=list(data.keys())
    new_keys={}
    for key in old_keys:
        new_keys[key]=key.lower()
    data = data.rename(columns=new_keys)
    num_rows = len(data)
    start_index = 1
    end_index = num_rows - 2  # 倒数第1个数据的索引

    # 随机选择测试数据
    num_rows_ratio = int((1-split_ratio) * num_rows)
    random_indices = random.sample(range(start_index, end_index + 1), num_rows_ratio)
    test_rows = data.loc[random_indices]
    # 剩下的训练数据
    train_rows = data.drop(index=random_indices)

    train_rows.to_excel(train_data_path, index=False)
    test_rows.to_excel(test_data_path, index=False)

    return train_rows,test_rows


#合并低、中保真数据，并加上雷诺数
def merge_data(data_paths,save_path):
    new_data={'re':None,'alpha':None,'cl':None,'cd':None}
    for path in data_paths:
        data=read_data_from_excel(path)
        start_idx=path.find('_')
        end_idx=path.rfind('.')
        re_num=int(path[start_idx+1:end_idx])
        data_length=len(data['alpha'].to_list())
        re=[re_num for i in range(data_length)]
        if new_data['re']==None:
            new_data['re']=re
            new_data['alpha']=data['alpha'].tolist()
            new_data['cl']=data['cl'].tolist()
            new_data['cd']=data['cd'].tolist()
        else:
            new_data['re'] += re
            new_data['alpha'] += data['alpha'].tolist()
            new_data['cl'] += data['cl'].tolist()
            new_data['cd'] += data['cd'].tolist()
    write_data_to_excel(new_data,save_path)

def make_dataset(ratio,set_name,use_pre_processed_data=True):
    Data_Dir_Pred='../RealData/sg6043/pre-processed-data'
    Data_Dir_NOT_Pred='../RealData/sg6043/not-pre-processed-data'
    HF_KEY='UIUC-HF'
    MF_KEY='Xfoil-N9-MF'
    LF_KEY='Xfoil-N5-LF'

    dataset_dir = f'../RealData/sg6043/{set_name}'
    make_dir(dataset_dir)
    if use_pre_processed_data:
        src_hf_dir=pthjoin(Data_Dir_Pred,HF_KEY)
        src_mf_dir=pthjoin(Data_Dir_Pred,MF_KEY)
        src_lf_dir=pthjoin(Data_Dir_Pred,LF_KEY)
    else:
        src_hf_dir = pthjoin(Data_Dir_NOT_Pred, HF_KEY)
        src_mf_dir = pthjoin(Data_Dir_NOT_Pred, MF_KEY)
        src_lf_dir = pthjoin(Data_Dir_NOT_Pred, LF_KEY)
    copy_onefolder_to_anotherfolder(src_hf_dir,dataset_dir)
    copy_onefolder_to_anotherfolder(src_mf_dir,dataset_dir)
    copy_onefolder_to_anotherfolder(src_lf_dir,dataset_dir)

    dst_hf_dir = pthjoin(dataset_dir, 'UIUC-USE')
    sample_hf_train_dir = pthjoin(dataset_dir, 'UIUC_Sampled/Train/')
    sample_hf_test_dir = pthjoin(dataset_dir, 'UIUC_Sampled/Test/')
    make_dir(dst_hf_dir)
    make_dir(sample_hf_train_dir)
    make_dir(sample_hf_test_dir)

    #make lf、mf dataset
    lf_datas = []
    mf_datas = []
    re_nums = [50000, 100000, 200000, 500000, 1000000]
    for re in re_nums:
        lf_path = pthjoin(src_lf_dir, f'n5_{re}.xlsx')
        lf_datas.append(lf_path)
        mf_path = pthjoin(src_mf_dir, f'n9_{re}.xlsx')
        mf_datas.append(mf_path)
    lf_merged_data_path = pthjoin(dataset_dir, 'lf_merged_data.xlsx')
    mf_merged_data_path = pthjoin(dataset_dir, 'mf_merged_data.xlsx')
    merge_data(lf_datas, lf_merged_data_path)
    merge_data(mf_datas, mf_merged_data_path)

    #make hf dataset
    split_ratio = ratio
    hf_train_path = pthjoin(dst_hf_dir, f'hf_train.xlsx')
    hf_test_path = pthjoin(dst_hf_dir, f'hf_test.xlsx')
    src_names, src_paths = get_names_and_paths_in_dir(src_hf_dir)
    Selected_rows, Remained_rows = None, None
    for i, path in enumerate(src_paths):
        sample_train_path = pthjoin(sample_hf_train_dir, src_names[i])
        sample_test_path = pthjoin(sample_hf_test_dir, src_names[i])

        selected_rows, remained_rows = split_data_by_ratio(path, split_ratio, sample_train_path, sample_test_path)
        if Selected_rows is None:
            Selected_rows = selected_rows
            Remained_rows = remained_rows
        else:
            Selected_rows = pd.concat([Selected_rows, selected_rows], ignore_index=True)
            Remained_rows = pd.concat([Remained_rows, remained_rows], ignore_index=True)
    Selected_rows.to_excel(hf_train_path, index=False)
    Remained_rows.to_excel(hf_test_path, index=False)


if __name__ == '__main__':
    # modify_keys()
    # make_dataset(0.7,'plan1',True)
    # make_dataset(0.5,'plan2',True)
    # make_dataset(0.6,'plan3',True)
    make_dataset(0.5,'planx3',True)
    src_data_show_only_scatter('planx3')
