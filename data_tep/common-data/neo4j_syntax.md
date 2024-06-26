## neo4j 基本语法

### 查询

#### 查询节点

```cypher
MATCH (n) RETURN n
```

#### 查询路径

##### 指定步长
```cypher
match p = (a)-[r*..5]-(b)
where a.name = 'x1' and b.name='x44'
return extract(n in nodes(p)| n.name)
```
```cypher
match p = (a)-[r*..5]-(b)
where a.name = 'x1' and b.name='x44'
return p
```

##### 最短路径
```cypher
match p = shortestPath((a)-[*]-(b))
where a.name = 'x1' and b.name='x46'
return p
```

```cypher
match p=shortestpath((a)-[r*..8]->(b)) where a.name="x4" and b.name="x13" return p,extract(n in nodes(p)| n.name),extract(r in relationships(p)| type(r))
```