import matplotlib.pyplot as plt
import os

from DatasetFunc.DatasetReal import get_test_loader

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

plt.switch_backend('TkAgg')  # 使用 TkAgg 作为交互式框架

SHOW=True
class PlotCfg():
    def __init__(self):

        self.styles()
        self.markers()
        self.colors()
        self.labels()

    def trans2inch(self, cm):
        # dpi=像素/英寸
        # 1 inch = 2.54 cm
        return cm / 2.54

    def trans2cm(self, inch):
        return 2.54 * inch

    # 绘图设置，字体，字大小，坐标轴
    def plt_pre_config(self):
        plt.rc('font', family=self.font)
        plt.rcParams['font.size'] = self.fontsize
        plt.rcParams['xtick.direction'] = 'in'
        plt.rcParams['ytick.direction'] = 'in'

    def cal_wh(self,col,row):
        width = row * self.fig_width
        height = col * self.fig_height
        return width,height

    def markers(self):
        self.traindata_marker='o'
        self.testdata_marker=self.traindata_marker
        self.predmarkers={'c32':'x','tlmfnn':'+','nargm':'v'}

    def colors(self):
        self.traindata_color='black'
        self.testdata_color='green'
        self.predcolors={"c32":'red','tlmfnn':'blue','nargp':'orange'}

    def labels(self):
        self.traindata_label='Train'
        self.testdata_label='Test'
        self.predlabels={'c32':"$C_3^2$",'tlmfnn':'MFDNN','nargp':'NARGP'}

    def styles(self):
        self.linewidth=1
        self.s_real=20  #真实数据marker的大小
        self.s=20
        self.markersize=2
        self.clcd_style={"cl":"$C_L$","cd":"$C_D$"}
        self.alpha_style='$\\alpha$'
        self.dpi = 600
        self.font = 'Times New Roman'  # Arial
        self.fontsize = 7
        self.save_type = 'png'  # 图片保存格式

        # 图片2*4
        self.fig_width = 4.5
        self.fig_height = 5

        self.fig_width = self.trans2inch(self.fig_width)
        self.fig_height = self.trans2inch(self.fig_height)
cfgs=PlotCfg()


def sub_plot_func(data_loader,cl_or_cd,ax,label,color,legend=False):
    name, re_num, re, alpha, cl, cd, re_norm, alpha_norm, cl_norm, cd_norm = next(data_loader)
    re_num=re_num[0]//10000
    xlabel="$\\alpha$"
    if cl_or_cd=='cl':
        y=cl
        ylabel = "$C_L$"
        title=f'$C_L$ (Re:{re_num}$\\times10^4$)'
    else:
        y=cd
        ylabel = "$C_D$"
        title=f'$C_D$ (Re:{re_num}$\\times10^4$)'

    y= y.squeeze(0)
    alpha=alpha.squeeze(0)

    ax.plot(alpha, y, label=label, color=color, linewidth=1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if legend:
        ax.legend(loc='upper right', bbox_to_anchor=(1.6, 1))


def sub_plot_func_sample_data(data_loader,cl_or_cd,ax,label,color,marker,legend=False):
    name, re_num, re, alpha, cl, cd, re_norm, alpha_norm, cl_norm, cd_norm = next(data_loader)
    re_num=re_num[0]//10000
    xlabel="$\\alpha$"
    if cl_or_cd=='cl':
        y=cl
        ylabel = "$C_L$"
        title=f'$C_L$ (Re:{re_num}$\\times10^4$)'
    else:
        y=cd
        ylabel = "$C_D$"
        title=f'$C_D$ (Re:{re_num}$\\times10^4$)'

    y = y.squeeze(0)
    alpha=alpha.squeeze(0)

    facecolor = 'none'
    edgecolor = color
    if marker=='x':
        ax.scatter(alpha, y, label=label, marker=marker, color=color, s=30,linewidth=1)
    else:
        ax.scatter(alpha, y, label=label, marker=marker,facecolor=facecolor,edgecolor=edgecolor, s=30,linewidth=1)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if legend:
        ax.legend(loc='upper right', bbox_to_anchor=(1.6, 1))


def src_data_show_only_scatter(dataset_name,cl_or_cd):
    set_name=dataset_name
    cl_or_cd=cl_or_cd
    src_all_save_path = f'../RealData/sg6043/{set_name}/all_scr_data_{cl_or_cd}.{cfgs.save_type}'

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
    cfgs.plt_pre_config()
    col,row=2,one_line_number
    w,h=cfgs.cal_wh(col,row)
    fig, ax = plt.subplots(col,row)
    fig.set_size_inches(w,h)
    row_number=0
    col_number=0
    legend=False
    label=cl_or_cd
    subfig_label = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l']
    for i,re_x in enumerate(re_nums):
        f_ax=ax[row_number,col_number]
        if re_x in re_lf:  #画出中、低保真数据
            sub_plot_func(lf_test_loader,cl_or_cd,f_ax,'LF','green',legend)
            sub_plot_func(mf_test_loader,cl_or_cd,f_ax,'MF','blue',legend)
        if re_x in re_hf:   #画出高保真训练和测试数据
            # sub_plot_func(hf_test_loader,cl_or_cd,ax[row_number,col_number],'HF','red',legend)
            sub_plot_func_sample_data(hf_sample_train_loader,cl_or_cd,f_ax,'HF Train','red','x',legend)
            sub_plot_func_sample_data(hf_sample_test_loader,cl_or_cd,f_ax,'HF Test','black','o',legend)
        if re_x in re_hf and re_x in re_lf:
            handles, labels = ax[row_number,col_number].get_legend_handles_labels()

        if col_number==one_line_number-1:
            col_number=0
            row_number+=1
        else:
            col_number+=1
        print(re_x,row_number,col_number)
        print(f"one line number:{one_line_number}")

        # 设置文本，标注
        f_ax.text(0.5, -0.3, f'({subfig_label[i]})', transform=f_ax.transAxes,
                  va='center', ha='center', fontsize=8)
        f_ax.set_xlabel(cfgs.alpha_style, labelpad=0.1)
        f_ax.set_ylabel(cfgs.clcd_style[label], labelpad=0.3)
        f_ax.set_title(f'{cfgs.clcd_style[label]} (Re:{re_x / 10}$\\times10^5$)')
        f_ax.set_xlim(-7,13)
    fig.legend(handles, labels,loc='lower center',ncol=4,frameon=False,borderaxespad=-0.5)
    # plt.tight_layout()
    fig.tight_layout()

    plt.savefig(src_all_save_path,dpi=cfgs.dpi)
    if SHOW:
        plt.show()
    plt.close()


if __name__ == '__main__':
    dataset_name='planx'
    for cl_or_cd in ['cl','cd']:
        src_data_show_only_scatter(dataset_name,cl_or_cd)