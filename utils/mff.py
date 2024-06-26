import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mutual_info_score
from scipy.stats import entropy
import sys
from pathlib import Path
import json
import scipy.io as scio

class mff_dataset_read:

    def __init__(self):
        self.folder_path = 'data_mff/MFF-dataset/test/'
        self.mff_test_path = {
            'Set1_1':'FaultyCase1.mat',
            'Set1_2':'FaultyCase1.mat',
            'Set1_3':'FaultyCase1.mat',
            'Set2_1':'FaultyCase2.mat',
            'Set2_2':'FaultyCase2.mat',
            'Set2_3':'FaultyCase2.mat', 
            'Set3_1':'FaultyCase3.mat',
            'Set3_2':'FaultyCase3.mat',
            'Set3_3':'FaultyCase3.mat', 
            'Set4_1':'FaultyCase4.mat',
            'Set4_2':'FaultyCase4.mat',
            'Set4_3':'FaultyCase4.mat',  
            'Set5_1':'FaultyCase5.mat',
            'Set5_2':'FaultyCase5.mat',
            'Set6_1':'FaultyCase6.mat',      
            'Set6_2':'FaultyCase6.mat',      
        }
        self.mff_table = {
            'Set1_1':{'FaultStart':1566,'FaultEnd':5181},
            'Set1_2':{'FaultStart':657,'FaultEnd':3777},
            'Set1_3':{'FaultStart':691,'FaultEnd':3691},
            'Set2_1':{'FaultStart':2244,'FaultEnd':6616},
            'Set2_2':{'FaultStart':476,'FaultEnd':2656},
            'Set2_3':{'FaultStart':331,'FaultEnd':2467},
            'Set3_1':{'FaultStart':1136,'FaultEnd':8352},
            'Set3_2':{'FaultStart':333,'FaultEnd':5871},
            'Set3_3':{'FaultStart':596,'FaultEnd':9566},
            'Set4_1':{'FaultStart':953,'FaultEnd':6294},
            'Set4_2':{'FaultStart':851,'FaultEnd':3851},
            'Set4_3':{'FaultStart':241,'FaultEnd':3241},
            'Set5_1':{'FaultStart':686,'FaultEnd':1172,'FaultStart2':1772,'FaultEnd2':2253},
            'Set5_2':{'FaultStart':1633,'FaultEnd':2955,'FaultStart2':7031,'FaultEnd2':7553,'FaultStart3':8057,'FaultEnd3':10608},
            'Set6_1':{'FaultStart':1723,'FaultEnd':2800},
            'Set6_2':{'FaultStart':1037,'FaultEnd':4830},
        }

    def mff_dataset_read(self,test_set):
        mff_table = self.mff_table[test_set]
        mff_test_path = self.folder_path + self.mff_test_path[test_set]
        data = scio.loadmat(mff_test_path)
        dat_matrix = data[test_set]
        return dat_matrix,mff_table
    
class preprocess_mff_csv:
    def __init__(self, mff_df, kg_df, node_df):
        self.mff_df = mff_df
        self.kg_df = kg_df
        self.node_df = node_df
        map_node2type = {}
        for i,row in self.node_df.iterrows():
            map_node2type[row['Node']] = row['Type']
        self.map_node2type = map_node2type
        self.relation_reverse = {'Input':'Output','Output':'Input',
                                 'State':'State_of','State_of':'State',
                                 'Contain':'Contained_by','Contained_by':'Contain'}
    
    def searchinfo_node(self,node):
        nodetype = self.map_node2type[node]
        return nodetype
    
    def searchinfo_columns(self,row):
        nodetype1 = self.searchinfo_node(row['Node1'])
        nodetype2 = self.searchinfo_node(row['Node2'])
        relation = row['relation']
        raw_triple = [row['Node1'],nodetype1,relation,row['Node2'],nodetype2]
        reverse_triple = [row['Node2'],nodetype2,self.relation_reverse[relation],row['Node1'],nodetype1]
        return raw_triple,reverse_triple

    def concatenate_data(self,save_csv_path,triple_design):
        
        def push_triple_into_data(triple,data):
            #排除一些不被许可的关系
            if triple[2]!='Input':
                if triple not in data:
                    data.append(triple)

        concatenated_data = []
        for i,row in self.kg_df.iterrows():
            raw_triple,reverse_triple = self.searchinfo_columns(row)
            push_triple_into_data(raw_triple,concatenated_data)
            push_triple_into_data(reverse_triple,concatenated_data)
        
        self.triples_df = pd.DataFrame(concatenated_data, columns=triple_design)
        self.triples_df.to_csv(save_csv_path, index=False, encoding='utf-8-sig')
    
    def csv_to_graph_json(self,save_json_path,save_csv_path=None,ExcL=[]):
        if save_csv_path:
            self.triples_df = pd.read_csv(save_csv_path)
        hmp_graph = {}
        for i,row in self.triples_df.iterrows():
            node1 = row[0]
            relation = row[2]
            node2 = row[3]
            if node1 not in ExcL and node2 not in ExcL:
                if node1 not in hmp_graph.keys():
                    hmp_graph[node1] = {}
                if node2 not in hmp_graph.keys():
                    hmp_graph[node2] = {}
                hmp_graph[node1][node2]=relation
        # 将字典转换为JSON格式的字符串
        json_str = json.dumps(hmp_graph, indent=4)
        # 将JSON字符串存储到文件中
        with open(save_json_path, 'w', encoding='utf-8') as json_file:
            json_file.write(json_str)

if __name__ == '__main__':

    data_path = 'Set1_1'
    mdr = mff_dataset_read()
    dat_matrix,_ = mdr.mff_dataset_read(data_path)

    # 读取TEP-Introduction.xlsx文件
    mff_df = pd.read_excel('data_mff/MFF-Introduction.xlsx', engine='openpyxl')

    # 将DataFrame保存为CSV文件
    mff_df.to_csv('data_mff/MFF-Introduction.csv', index=False, encoding='utf-8-sig')

    # 读取TEP-KG-raw.xlsx文件
    # kg_df = pd.read_excel('data_mff/MFF-KG-raw.xlsx', engine='openpyxl')
    kg_df = pd.read_excel('data_mff/MFF-KG-raw - copy.xlsx', engine='openpyxl')

    # 将DataFrame保存为CSV文件
    kg_df.to_csv('data_mff/MFF-KG-raw.csv', index=False, encoding='utf-8-sig')

    # 读取TEP-KG-raw.xlsx文件
    # node_df = pd.read_excel('data_mff/MFF-KG-Node.xlsx', engine='openpyxl')
    node_df = pd.read_excel('data_mff/MFF-KG-Node - copy.xlsx', engine='openpyxl')

    # 将DataFrame保存为CSV文件
    node_df.to_csv('data_mff/MFF-KG-Node.csv', index=False, encoding='utf-8-sig')

    #三元组各项命名空间
    triple_design = ['Node1 Description','Node1 Type','Relation','Node2 Description','Node2 Type']
    
    #三元组存储地址
    triple_csv_path = 'data_mff/triples.csv'

    #json存储地址
    raw_save_json_path = 'data_mff/graph.json'

    #json存储地址
    simple_save_json_path = 'data_mff/simple_graph.json'

    #图谱预处理
    pmc = preprocess_mff_csv(mff_df=mff_df,node_df=node_df,kg_df=kg_df)

    pmc.concatenate_data(triple_csv_path,triple_design)

    pmc.csv_to_graph_json(raw_save_json_path)

    pmc.csv_to_graph_json(simple_save_json_path,ExcL=['x24'])