import csv
import pandas as pd
import json

class preprocess_tep_csv:

    def __init__(self, tep_df, kg_df):
        # 创建一个字典，将Index映射到有效信息上
        self.kg_df = kg_df
        self.descp_dict = dict(zip(tep_df['Index'], tep_df['Variable Description']))
        self.type1_dict = dict(zip(tep_df['Index'], tep_df['Variable Type1']))
        self.type2_dict = dict(zip(tep_df['Index'], tep_df['Variable Type2']))
        self.type2_to_abbreviation = {"Process measurement variable":"Measurement",
                                      "Component measurement variable":"Measurement",
                                      "Process operating variable":"Operating"}
        self.relation_reverse = {'Input':'Output','Output':'Input',
                                 'State':'State_of','State_of':'State',
                                 'Contain':'Contained_by','Contained_by':'Contain',
                                 'Generated_from':'Generate','Generate':'Generated_from'}

    def hashmap_vaild(self,key_name,hashmap):
        if key_name in hashmap.keys():
            return hashmap[key_name]
        else:
            return None
    
    def searchinfo_node(self,node):
        """查询节点描述和类型，输入为x1-x53以及Stream和Device"""
        #节点描述
        descp = self.hashmap_vaild(node,self.descp_dict)
        if not descp:
            descp = node
        
        #节点类型
        type1 = self.hashmap_vaild(node,self.type1_dict)
        type2 = self.hashmap_vaild(node,self.type2_dict)
        if type1 and type2:
            # nodetype = type1+'_'+self.type2_to_abbreviation[type2]
            nodetype = self.type2_to_abbreviation[type2]
            # nodetype = 'Variable'
        else:
            if 'Stream' in node:
                nodetype = 'Stream'
            elif len(node) == 1:
                nodetype = 'Substance'
            else:
                nodetype = 'Device'

        return descp,nodetype

    def searchinfo_columns(self,row):
        descp1,nodetype1 = self.searchinfo_node(row['Node1'])
        descp2,nodetype2 = self.searchinfo_node(row['Node2'])
        # relation = nodetype1+"_"+row['relation']+"_"+nodetype2
        relation = row['relation']

        #补全反向关系
        raw_triple = [row['Node1'],nodetype1,row['relation'],row['Node2'],nodetype2]
        reverse_triple = [row['Node2'],nodetype2,self.relation_reverse[row['relation']],row['Node1'],nodetype1]

        # return descp1,nodetype1,relation,descp2,nodetype2
        return raw_triple,reverse_triple
    
    def concatenate_data(self,save_csv_path,triple_design):

        def push_triple_into_data(triple,data):
            #排除一些不被许可的关系
            if triple[2]!='Input' and triple[2] != 'Generated_from':
                if triple not in data:
                    data.append(triple)

        # 应用函数到每一行
        # concatenated_data = self.kg_df.apply(self.searchinfo_columns, axis=1).tolist()
        concatenated_data = []
        for i,row in self.kg_df.iterrows():
            raw_triple,reverse_triple = self.searchinfo_columns(row)
            push_triple_into_data(raw_triple,concatenated_data)
            push_triple_into_data(reverse_triple,concatenated_data)

        # 创建一个新的DataFrame来存储三元组数据
        self.triples_df = pd.DataFrame(concatenated_data, columns=triple_design)

        # 将DataFrame保存为CSV文件
        self.triples_df.to_csv(save_csv_path, index=False, encoding='utf-8-sig')
    
    def csv_to_graph_json(self,save_json_path,save_csv_path=None,ExcL =['x53']):
        #考虑全部的节点，不排除component
        # ExcL = ['x53']
        #不考虑component(x23-x41)和x53
        # ExcL = ['x53']+['x'+str(i) for i in range(23,42)]
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
    
    def csv_to_kg_txt(self,save_txt_path,kg_word_table,save_csv_path=None,ExcL =['x53']):
        if save_csv_path:
            self.triples_df = pd.read_csv(save_csv_path)
        lines = []
        count = 0
        id2word = {}
        for i,row in self.triples_df.iterrows():
            node1 = row[0]
            relation = row[2]
            node2 = row[3]
            if node1 not in ExcL and node2 not in ExcL:
                if node1 not in id2word.values():
                    id2word[count] = node1
                    count+=1
                if node2 not in id2word.values():
                    id2word[count] = node2
                    count+=1
                descp1 = self.hashmap_vaild(node1,self.descp_dict)
                if not descp1:
                    descp1 = node1
                descp2 = self.hashmap_vaild(node2,self.descp_dict)
                if not descp2:
                    descp2 = node2
                lines.append(node1+'\t'+relation+'\t'+node2+'\n')
                # lines.append(node1+' '+descp1+'\t'+relation+'\t'+descp2+'\n')

        with open(save_txt_path, 'w', encoding='utf-8') as file:
            file.writelines(lines)

        # 将字典转换为JSON格式的字符串
        json_str = json.dumps(id2word , indent=4)

        # 将JSON字符串存储到文件中
        with open(kg_word_table , 'w', encoding='utf-8') as json_file:
            json_file.write(json_str)

if __name__ == '__main__':

    # 读取TEP-Introduction.xlsx文件
    tep_df = pd.read_excel('data_tep/TEP-Introduction.xlsx', engine='openpyxl')

    # 将DataFrame保存为CSV文件
    tep_df.to_csv('data_tep/TEP-Introduction.csv', index=False, encoding='utf-8-sig')

    # 读取TEP-KG-raw.xlsx文件
    kg_df = pd.read_excel('data_tep/TEP-KG-raw.xlsx', engine='openpyxl')

    # 将DataFrame保存为CSV文件
    kg_df.to_csv('data_tep/TEP-KG-raw.csv', index=False, encoding='utf-8-sig')

    #三元组各项命名空间
    triple_design = ['Node1 Description','Node1 Type','Relation','Node2 Description','Node2 Type']
    
    #三元组存储地址
    triple_csv_path = 'data_tep/triples.csv'

    #json存储地址
    raw_save_json_path = 'data_tep/graph.json'
    simple_save_json_path = 'data_tep/simple_graph.json'

    #txt存储地址
    simple_save_txt_path = 'data_tep/tep_simple.txt'
    kg_word_table = 'data_tep/tep_wordtable.json'

    #图谱预处理
    ppc = preprocess_tep_csv(tep_df=tep_df,kg_df=kg_df)

    ppc.concatenate_data(triple_csv_path,triple_design)

    ppc.csv_to_graph_json(raw_save_json_path)
    ppc.csv_to_graph_json(simple_save_json_path,ExcL = ['x53']+['x'+str(i) for i in range(23,42)])

    ppc.csv_to_kg_txt(simple_save_txt_path,kg_word_table,ExcL = ['x53']+['x'+str(i) for i in range(23,42)])