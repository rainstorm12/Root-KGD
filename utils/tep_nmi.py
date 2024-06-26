import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mutual_info_score
from scipy.stats import entropy
import sys
from pathlib import Path

# 保证绝对路径可以被检测到
current_file_path = Path(__file__).parent.absolute() # 获取当前文件的绝对路径
project_root_path = current_file_path.parent # 获取项目根目录的路径
sys.path.append(str(project_root_path)) # 将项目根目录添加到 sys.path
from utils.common_function import list_filenames

class tep_dataset_read:

    def __init__(self):
        pass

    def tep_dataset_read(self,data_path):
        """
            从路径中读取tep原始的时序数据
            考虑原始数据矩阵维度不同，最后统一转化为[52,x]的矩阵，x表示时序的长度，可能是480/500
            主要是考虑了d00的原始数据维度为[500,52],其他数据为[52,480](训练集)/[52,960](测试集)
        """
        df = pd.read_csv(data_path, sep='\t', header=None)
        dat_array = []
        for index,row in df.iterrows():
            array = np.fromstring(row[0], sep=' ')
            array=array.reshape((len(array), 1))
            dat_array.append(array)
        dat_matrix = np.concatenate(dat_array, axis=1)
        if dat_matrix.shape[0]!=52:
            dat_matrix = dat_matrix.transpose()
        return dat_matrix

class nmi_calculate:

    def __init__(self):
        pass

    def sturges_rule(self,n):
        '''计算基于 Sturges' Rule 的 bins 数量'''
        k = 1 + 3.322* np.log10(n)
        return int(np.ceil(k))
    
    def calculate_nmi(self,S, V, bins=10):
        '''
            归一化互信息熵计算(normalized mutual information entropy)
            H(S) = -∑i p(si)log(p(si))
            H(V) = -∑i p(vi) log p(vi))
            I(S; V) = ∑si ∑vj p(si, vj)log(p(si, vj)) / (p(si)p(vj))
            NMI(S; V) = 2 * I(S; V) / (H(S) + H(V))
            bins计算上默认TEP数据在480-960之间,那么根据Sturges' Rule,得到bins在10左右
        '''
        hist = np.histogram2d(S, V, bins=bins)[0]
        # 计算边缘概率分布
        marginal_S = np.sum(hist, axis=1)
        marginal_V = np.sum(hist, axis=0)
        # 边缘概率分布归一化
        marginal_S = marginal_S / np.sum(marginal_S)
        marginal_V = marginal_V / np.sum(marginal_V)
        # 计算熵H(S)和H(V)
        H_S = 0
        for si in range(len(marginal_S)):
            if marginal_S[si] > 0:
                H_S += -marginal_S[si] * np.log2(marginal_S[si])
        H_V = 0
        for sj in range(len(marginal_V)):
            if marginal_V[sj] > 0:
                H_V += -marginal_V[sj] * np.log2(marginal_V[sj])
        # 计算联合概率分布
        joint_prob = hist / np.sum(hist)
        #计算互信息I(S; V)
        I_SV = 0
        for si in range(len(marginal_S)):
            for sj in range(len(marginal_V)):
                if joint_prob[si][sj] > 0:
                    I_SV += joint_prob[si][sj] * np.log2(joint_prob[si][sj] / (marginal_S[si] * marginal_V[sj]))
        NMI = 2* I_SV / (H_S + H_V)
        return NMI

    def NMI_scipy(self,X,Y):
        """scipy库直接计算互信息熵,目前这个算的有点问题"""
        mi = mutual_info_score(X, Y)
        entropy_X = entropy(X,base=2)  # 使用 base=2 以便结果以比特为单位
        entropy_Y = entropy(Y,base=2)
        nmi = mi / ((entropy_X + entropy_Y) / 2)
        return nmi

    def calculate_nmi_matrix(self, dat_matrix, calculate='formula' ,bins=10):
        '''归一化互信息熵矩阵计算'''
        variable_num = len(dat_matrix)
        NMI_Matrix = np.zeros((variable_num,variable_num))
        for i in range(variable_num):
            for j in range(variable_num):
                if (calculate=='scipy'):
                    NMI_Matrix[i][j] = self.NMI_scipy(dat_matrix[i],dat_matrix[j])
                if (calculate=='formula'):
                    NMI_Matrix[i][j] = self.calculate_nmi(dat_matrix[i],dat_matrix[j],bins=bins)
        return NMI_Matrix
    
    def nmi_matrix_visual(self,nmi_matrix, fig_title = "NMI matix" ,fig_name = 'fig.png'):
        #调色函数说明： https://www.bookstack.cn/read/seaborn-0.9/docs-60.md
        # cmap = sns.diverging_palette(240,10,center='light',as_cmap=True)
        plt.figure(figsize=(10, 8))
        ax = sns.heatmap(nmi_matrix,square=True,linewidths=0.5,
                        vmin=np.min(nmi_matrix), vmax=np.max(nmi_matrix),cbar=True) # vmin和vmax是自定义显示颜色的范围
                        #  xticklabels=bands_wavelength, yticklabels=bands_wavelength[::-1])
        
        plt.title(fig_title) # 设置标题
        # plt.xlabel("值", fontsize=14) # 设置x标题
        # plt.ylabel("值的平方", fontsize=14) # 设置y轴标题
        plt.savefig(fig_name,bbox_inches='tight',dpi=300)
        plt.show()

class tep_dynamic_triples:
    
    def __init__(self):
        pass

    def thr_filter_with_nmi(self,nmi_matrix,thr):
        """
            根据阈值对nmi矩阵进行过滤,获取动态三元组
            Arguments:
                nmi_matrix(array[n,n]):NMI矩阵,n为维度
                thr(int):过滤阈值
            Returns:
                count(int):过滤出的变量组合个数
                nmi_matrix_copy(array[n,n]):过滤后的NMI矩阵,所有过滤出的数值组合全部置为1,便于可视化
                filter_relation(array[m,2]):过滤出的m个组合坐标
        """
        filter_relation = []
        count = 0
        nmi_matrix_copy = nmi_matrix.copy()
        for i in range(len(nmi_matrix)):
            for j in range(len(nmi_matrix[i])):
                #对角线上不考虑
                if i!=j:  
                    if nmi_matrix[i][j] > thr:
                        nmi_matrix_copy[i][j] = 1
                        filter_relation.append([i,j])
                        count += 1
        # nc.nmi_matrix_visual(nmi_matrix_copy) 
        return count,nmi_matrix_copy,filter_relation

    def construct_dynamic_triples_to_csv(self,thr,TEP_dataset_folder,TEP_Dynamic_Triples_folder,file_index):
        """
            根据阈值对nmi矩阵进行过滤,获取动态三元组,保存到csv文件中
            Arguments:
                thr(int):过滤阈值
                TEP_dataset_folder(str):读取文件夹名称,如'data/TEP-dataset/train/'
                TEP_Dynamic_Triples_folder(str):保存文件夹名称,如'data/TEP-Dynamic-Triples/train/'
                file_index(str):当前文件序号,如'd00.dat'
            Returns:
                nmi_matrix(array[n,n]):NMI矩阵,n为维度
                nmi_matrix_copy(array[n,n]):过滤后的NMI矩阵,所有过滤出的数值组合全部置为1,便于可视化
                save_csv_path(str):保存的路径名
        """
        tdr = tep_dataset_read()
        nc = nmi_calculate()
        filename = TEP_dataset_folder +file_index #当前的时序数据
        save_csv_path = TEP_Dynamic_Triples_folder+file_index.split('.')[0]+'thr_'+str(thr)+'.csv' #csv存储位置
        triple_design = ['Node1 Index','Node1 Description','Node1 Type','Relation','Node2 Index','Node2 Description','Node2 Type'] #csv数据列的命名

        dat_matrix = tdr.tep_dataset_read(filename) #预处理原始数据
        nmi_matrix = nc.calculate_nmi_matrix(dat_matrix) #计算变量NMI
        relation_count,nmi_matrix_copy,filter_relation = self.thr_filter_with_nmi(nmi_matrix,thr) #根据阈值过滤NMI

        # nc.nmi_matrix_visual(nmi_matrix) #NMI可视化
        # nc.nmi_matrix_visual(nmi_matrix_copy) #过滤后的NMI可视化
        print(f"{file_index}: there are {relation_count} relations after nmi filtering")

        #查询节点相关描述和种类需要调用专家知识
        from utils.preprocess_tep_csv import preprocess_tep_csv
        tep_df = pd.read_excel('data/TEP-Introduction.xlsx', engine='openpyxl') # 读取TEP-Introduction.xlsx文件
        kg_df = pd.read_excel('data/TEP-KG-raw.xlsx', engine='openpyxl') # 读取TEP-KG-raw.xlsx文件
        ppc = preprocess_tep_csv(tep_df=tep_df,kg_df=kg_df) 

        concatenated_data = []
        for relation_axis in filter_relation:

            #将坐标转化为变量序号字符串格式
            var_x = 'x' + str(relation_axis[0]+1) 
            var_y = 'x' + str(relation_axis[1]+1) 

            #根据专家知识搜索信息
            descpx,nodetypex = ppc.searchinfo_node(var_x)
            descpy,nodetypey = ppc.searchinfo_node(var_y)

            #关系定义
            relation_name = nodetypex+"_"+'Related_to'+"_"+nodetypey

            #以三元组形式存储
            concatenated_data.append([var_x,descpx,nodetypex,relation_name,var_y,descpy,nodetypey])

        triples_df = pd.DataFrame(concatenated_data, columns=triple_design)
        triples_df.to_csv(save_csv_path, index=False, encoding='utf-8-sig')

        print(f"{file_index}: save in {save_csv_path}")
        return nmi_matrix,nmi_matrix_copy,save_csv_path
    
    def construct_whole_dynamic_triples_to_csv(self,thr):
        """
            根据threshold将所有的数据集全部计算
        """
        from utils.common_function import list_filenames
        TEP_dataset_folder = 'data/TEP-dataset/train/' #读取文件夹名称
        TEP_Dynamic_Triples_folder = 'data/TEP-Dynamic-Triples/train/' #保存文件夹名称
        filenames = list_filenames(TEP_dataset_folder)
        for filename in filenames:
            tdt.construct_dynamic_triples_to_csv(thr,TEP_dataset_folder,TEP_Dynamic_Triples_folder,filename)

        TEP_dataset_folder = 'data/TEP-dataset/test/' #读取文件夹名称
        TEP_Dynamic_Triples_folder = 'data/TEP-Dynamic-Triples/test/' #保存文件夹名称
        filenames = list_filenames(TEP_dataset_folder)
        for filename in filenames:
            tdt.construct_dynamic_triples_to_csv(thr,TEP_dataset_folder,TEP_Dynamic_Triples_folder,filename)

class fit_filter_threshold(tep_dynamic_triples):

    def __init__(self,folder_path):
        # super().__init__(None)
        self.folder_path = folder_path #'data/TEP-dataset/train'
        self.cal_whole_nmi_matrix()

    def cal_whole_nmi_matrix(self):
        #计算所有的NMI矩阵
        from utils.common_function import list_filenames
        tdr = tep_dataset_read()
        nc = nmi_calculate()

        filenames = list_filenames(self.folder_path)
        self.data_list = []
        self.nmi_list = []
        for filename in filenames:
            dat_matrix = tdr.tep_dataset_read(self.folder_path+'/'+filename) #预处理原始数据
            nmi_matrix = nc.calculate_nmi_matrix(dat_matrix) #计算变量NMI
            self.data_list.append(dat_matrix)
            self.nmi_list.append(nmi_matrix)
    
    def fit_filter_thr(self,thr):
        self.nmi_copy_list = []
        self.count_list = []
        for i in range(len(self.nmi_list)):
            count,nmi_matrix_copy,_ = self.thr_filter_with_nmi(self.nmi_list[i],thr)
            self.count_list.append(count)
            self.nmi_copy_list.append(nmi_matrix_copy)
        for i in range(len(self.count_list)):
            print(f"the {i} matrix: there are {self.count_list[i]} relations after nmi filtering with threshold = {thr}")
    
if __name__ == '__main__':
    test = 0   
    if (test==1):
        # 单个数据集的NMI计算与可视化
        data_path = 'data/TEP-dataset/test/d00_te.dat'
        tdr = tep_dataset_read()
        dat_matrix = tdr.tep_dataset_read(data_path)
        nc = nmi_calculate()
        nmi_matrix = nc.calculate_nmi_matrix(dat_matrix)
        nc.nmi_matrix_visual(nmi_matrix)
    elif (test==2):
        # 过滤后的动态三元组保存
        tdt = tep_dynamic_triples()
        thr = 0.6 #NMI过滤阈值
        # 单个过滤
        # tdt.construct_dynamic_triples_to_csv(thr,'data/TEP-dataset/train/','data/TEP-Dynamic-Triples/train/','d00.dat')
        # 多个过滤
        tdt.construct_whole_dynamic_triples_to_csv(thr)
    elif (test==3):
        #测试不同数据对thr的性能以更好地选择thr
        fft = fit_filter_threshold('data/TEP-dataset/train')
        fft.fit_filter_thr(0.6)