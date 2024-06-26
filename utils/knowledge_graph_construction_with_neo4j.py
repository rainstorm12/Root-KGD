from py2neo import Node, Relationship,Graph,NodeMatcher,RelationshipMatcher,Subgraph,Path
import pandas as pd
from tqdm import tqdm

class knowledge_graph_construction_with_neo4j:

    def __init__(self, neo4j_user='neo4j', neo4j_password='neo4j',delete_ever=True):
        self.neo4j_user = neo4j_user
        self.neo4j_password = neo4j_password
        self.delete_ever = delete_ever
        self.test_graph = Graph(
            "http://localhost:7474", 
            auth=(neo4j_user,neo4j_password)
        )
        if self.delete_ever:
            self.test_graph.delete_all()  # 删除已有的所有内容
        self.node_matcher = NodeMatcher(self.test_graph)
        self.relationship_matcher = RelationshipMatcher(self.test_graph)
    
    #依赖函数
    #node列表内查重函数
    def node_duplicate_checking(self,node,node_list):
        num_attribute = len(node)
        list_attribute = list(dict(node).keys())
        for nd in node_list:
            if len(nd)==num_attribute:#检查属性数目是否相同
                flag=True#只要有属性不相同就会置位
                for attribute in list_attribute:
                    if node[attribute]!=nd[attribute]:
                        flag=False
                        break
                if flag==True:#所有属性都相同说明有重复，返回重复节点
                    return nd
        #如果不存在重复则返回0
        return 0
    
    #relation列表内查重函数
    def relation_duplicate_checking(self,relation,relation_list):
        num_attribute = len(relation)
        list_attribute = list(dict(relation).keys())
        for nd in relation_list:
            if len(nd)==num_attribute:#检查属性数目是否相同
                flag=True#只要有属性不相同就会置位
                for attribute in list_attribute:
                    if relation[attribute]!=nd[attribute]:
                        flag=False
                        break
                if flag==True:#所有属性都相同说明有重复，返回重复节点
                    return nd
        #如果不存在重复则返回0
        return 0
    
    def search_graph(self,node_matcher,node,nodetype,nodename):
        anode = node_matcher.match(nodetype).where(name=nodename)
        return self.node_duplicate_checking(node,anode)
    
    def construction_single_triple(self, node1_name, node1_type, relation_name, node2_name, node2_type, relation_type = "Bidirectional", multiple_relation = True):
        count = 0
        node1 = Node (node1_type, name = node1_name)
        node2 = Node (node2_type, name = node2_name)
        
        nodegraph = self.search_graph(self.node_matcher,node1,node1_type,node1_name)
        if not nodegraph:
            self.test_graph.create(node1)
        else:
            node1 = nodegraph

        nodegraph = self.search_graph(self.node_matcher,node2,node2_type,node2_name)
        if not nodegraph:
            self.test_graph.create(node2)
        else:
            node2 = nodegraph

        if relation_type == "Bidirectional":
            if multiple_relation:#允许多个关系存在，即双向多个关系
                relation = Relationship(node1,relation_name,node2)
                self.test_graph.create(relation)
                count = count+1
            else:#不允许多个关系存在，即双向单个关系，即每个方向最多只有一个关系
                if not len(list(self.relationship_matcher.match((node1,node2), r_type=None))):
                    relation = Relationship(node1,relation_name,node2)
                    self.test_graph.create(relation)
                    count = count+1

        if relation_type == "Unidirectional":
            if multiple_relation:#允许多个关系存在，即单向多个关系
                if not len(list(self.relationship_matcher.match((node2,node1), r_type=None))):
                    relation = Relationship(node1,relation_name,node2)
                    self.test_graph.create(relation)
                    count = count+1
            else:#不允许多个关系存在，即单向单个关系，两个节点之间只存在一个关系
                if not len(list(self.relationship_matcher.match((node2,node1), r_type=None))):
                    if not len(list(self.relationship_matcher.match((node1,node2), r_type=None))):
                        relation = Relationship(node1,relation_name,node2)
                        self.test_graph.create(relation)
                        count = count+1
        return count
                        
    
    def construction_with_neo4j(self,file_name,file_type,triple_design:list, relation_type = "Bidirectional", multiple_relation = True):
        count = 0
        if file_type=='csv':
            triple = pd.read_csv(file_name)
            for index, row in tqdm(triple.iterrows(), desc="Creating Nodes and Relationships", unit="triple"):
                node1_name, node1_type, relation_name, node2_name, node2_type = (
                    row[i] for i in triple_design
                )
                count += self.construction_single_triple(node1_name, node1_type, relation_name, node2_name, node2_type, relation_type, multiple_relation)
        print(f"construct {count} triples in fact")

if __name__ == '__main__':

    kgcwn = knowledge_graph_construction_with_neo4j()

    triple_design = ['Node1 Description','Node1 Type','Relation','Node2 Description','Node2 Type']
    
    kgcwn.construction_with_neo4j('data/triples.csv','csv',triple_design) #专家经验静态建图
    # kgcwn.construction_with_neo4j('data/TEP-Dynamic-Triples/train/d00thr_0.6.csv','csv',triple_design) #互信息熵动态建图
