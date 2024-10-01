#-*- encoding='utf-8'-*-
import matplotlib.pyplot as plt
import os

from DatasetFunc.DatasetReal import get_test_loader

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

plt.switch_backend('TkAgg')  # 使用 TkAgg 作为交互式框架


def set_legend(ax,cl_or_cd):
    if cl_or_cd=='cl':
        loc='lower right'
    else:
        loc='upper right'

    # 获取当前子图的图例实例
    legend_handles, _ = ax.get_legend_handles_labels()
    num_legends = len(legend_handles)

    # 根据图例数量调整布局
    # 根据图例数量调整布局
    if num_legends == 2:
        ax.legend(handles=legend_handles, loc=loc)
    elif num_legends == 4:
        if cl_or_cd=='cl':
            ax.legend(handles=legend_handles, loc=loc)
        else:
            ax.legend(handles=legend_handles, loc=loc,ncol=2)

    else:
        ax.legend(handles=legend_handles, loc=loc)


#专利绘图 灰度图
def sub_plot_func(data_loader,cl_or_cd,ax,label,linestyle,marker=None,legend=False):
    name, re_num, re, alpha, cl, cd, re_norm, alpha_norm, cl_norm, cd_norm = next(data_loader)
    re_num=re_num[0]//100000
    color='black'
    xlabel="$\\alpha$"
    if cl_or_cd=='cl':
        y=cl
        ylabel = "$C_L$"
        title=f'$C_L$ - $\\alpha$ (Re:{re_num}.0$\\times10^5$)'
    else:
        y=cd
        ylabel = "$C_D$"
        title=f'$C_D$ - $\\alpha$ (Re:{re_num}.0$\\times10^5$)'

    y= y.squeeze(0)
    alpha=alpha.squeeze(0)
    linewidth=1.5
    ax.plot(alpha, y, label=label, linestyle=linestyle,color=color, linewidth=linewidth)
    # ax.set_ylim(0, 2.5)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if legend:
        set_legend(ax,cl_or_cd)


def sub_plot_func_sample_data(data_loader,cl_or_cd,ax,label,marker,legend=False):
    name, re_num, re, alpha, cl, cd, re_norm, alpha_norm, cl_norm, cd_norm = next(data_loader)
    re_num=re_num[0]//100000
    xlabel="$\\alpha$"
    color='black'
    if cl_or_cd=='cl':
        y=cl
        ylabel = "$C_L$"
        title=f'$C_L$ (Re:{re_num}.0$\\times10^5$)'
    else:
        y=cd
        ylabel = "$C_D$"
        title=f'$C_D$ (Re:{re_num}.0$\\times10^5$)'

    y = y.squeeze(0)
    alpha=alpha.squeeze(0)

    facecolor = 'none'
    edgecolor = color
    linewidth=1
    s=50
    if marker=='x':
        ax.scatter(alpha, y, label=label, marker=marker,color=color,  s=s,linewidth=linewidth)
    else:
        ax.scatter(alpha, y, label=label, marker=marker, facecolor=facecolor,edgecolor=edgecolor,s=s,linewidth=linewidth)
    # ax.set_ylim(0, 2.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if legend:
        set_legend(ax,cl_or_cd)


def src_data_show_only_scatter(dataset_name,cl_or_cd):
    set_name=dataset_name
    cl_or_cd=cl_or_cd
    src_all_save_path = f'../RealData/sg6043/{set_name}/all_scr_data_{cl_or_cd}_blackwhite.png'

    lf_test_data_dir = f'../RealData/sg6043/{set_name}/Xfoil-N5-LF'
    mf_test_data_dir = f'../RealData/sg6043/{set_name}/Xfoil-N9-MF'
    hf_test_data_dir = f'../RealData/sg6043/{set_name}/UIUC-HF'
    hf_sample_train_dir=f"../RealData/sg6043/{set_name}/UIUC_Sampled/Train"
    hf_sample_test_dir=f"../RealData/sg6043/{set_name}/UIUC_Sampled/Test"

    lf_test_loader = get_test_loader(lf_test_data_dir, norm_paras=None)
    mf_test_loader = get_test_loader(mf_test_data_dir, norm_paras=None)
    hf_test_loader = get_test_loader(hf_test_data_dir, norm_paras=None)
    hf_sample_train_loader = get_test_loader(hf_sample_train_dir, norm_paras=None)
    hf_sample_test_loader = get_test_loader(hf_sample_test_dir, norm_paras=None)

    mf_test_loader=iter(mf_test_loader)
    lf_test_loader=iter(lf_test_loader)
    hf_test_loader = get_test_loader(hf_test_data_dir, norm_paras=None)
    hf_test_loader=iter(hf_test_loader)
    hf_sample_test_loader = iter(hf_sample_test_loader)
    hf_sample_train_loader = iter(hf_sample_train_loader)

    #三类数据汇总
    re_nums=[5,10,15,20,30,40,50,100]
    re_lf=[5,10,20,50,100]
    re_hf=[10,15,20,30,40,50]
    num_of_data = len(re_nums)  # re的个数
    one_line_number=num_of_data//2  #一行有多少子图
    print(f"one line number:{one_line_number}")
    plt.rcParams['font.family'] = 'SimSun'  #宋体
    plt.rcParams.update({'font.size': 12})
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    fig, ax = plt.subplots( one_line_number,2)
    fig.set_size_inches(9, 18)
    row_number=0
    col_number=0
    legend=False
    for re_x in re_nums:
        if re_x in re_lf:
            sub_plot_func(lf_test_loader,cl_or_cd,ax[row_number,col_number],'低保真',':',legend=False)
            sub_plot_func(mf_test_loader,cl_or_cd,ax[row_number,col_number],'中保真','--',legend=True)
        if re_x in re_hf:
            # sub_plot_func(hf_test_loader,cl_or_cd,ax[row_number,col_number],'HF','',legend)
            sub_plot_func_sample_data(hf_sample_train_loader,cl_or_cd,ax[row_number,col_number],'高保真-训练','x',legend=False)
            sub_plot_func_sample_data(hf_sample_test_loader,cl_or_cd,ax[row_number,col_number],'高保真-测试','o',legend=True)
        if re_x in re_hf and re_x in re_lf:
            handles, labels = ax[row_number,col_number].get_legend_handles_labels()
        # if row_number==0 and col_number== one_line_number - 1:
        #     ax[row_number,col_number].legend(handles, labels, loc=2, bbox_to_anchor=(1.05, 1))
        if col_number==2-1:
            col_number=0
            row_number+=1
        else:
            col_number+=1
        print(re_x,row_number,col_number)
        print(f"one line number:{one_line_number}")

    if cl_or_cd=='cl':
        label='$C_L-\\alpha$'
    else:
        label='$C_D-\\alpha$'

    # plt.suptitle(label)
    # plt.subplots_adjust(top=0.95)

    # plt.legend(handles, labels, loc=2, bbox_to_anchor=(1.05, 1))
    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    # plt.tight_layout()
    plt.savefig(src_all_save_path,bbox_inches='tight',dpi=1000)

    # plt.show()
    plt.close()
if __name__ == '__main__':
    dataset_name='planx'
    for cl_or_cd in ['cl','cd']:
        src_data_show_only_scatter(dataset_name,cl_or_cd)
