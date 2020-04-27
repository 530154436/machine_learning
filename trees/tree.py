# /usr/bin/env python3
# -*- coding:utf-8 -*-
from collections import deque
import pydotplus as pdp

'''
    定义树的一些结构、通用方法
'''
class BiNode(object):

    def __init__(self, id, left=None, right=None, **kwargs):
        self.id = id
        self.left = left
        self.right = right
        self.kwargs = kwargs

    def __repr__(self):
        return self.kwargs_to_str()

    def __str__(self):
        return self.kwargs_to_str()

    def update_kwargs(self, others:dict):
        if self.kwargs is None: self.kwargs = {}
        self.kwargs.update(others)

    def kwargs_to_str(self):
        return '\n'.join(['%s=%s' % (k,v) if v!=None else '%s' % k for k,v in self.kwargs.items()])

class BiTree(object):

    @classmethod
    def level_traversal(cls, root:BiNode, get_rel_pair=False) -> list:
        '''
        层次遍历
        :param root:           根节点
        :param get_rel_pair:   是否获取 父节点-节点 对
        :return:
        '''
        if not root:
            return []

        nodes = []
        pairs = []
        q = deque([root])
        while len(q)>0:
            root = q.popleft()
            nodes.append(root)

            if root.left:
                q.append(root.left)
                pairs.append((root, root.left))

            if root.right:
                q.append(root.right)
                pairs.append((root, root.right))

        return pairs if get_rel_pair else nodes

    @classmethod
    def depth(cls, root:BiNode):
        '''
        返回树的深度(后序遍历)
        :param root: 根节点
        :return:
        '''
        if not root:
            return 0
        l = cls.depth(root.left)
        r = cls.depth(root.right)
        return  max(l+1, r+1)

    @classmethod
    def plot(cls, root: BiNode, path):
        '''
        打印单颗树 (pydotplus + graphviz)

        brew install graphviz
        graphviz文档 https://graphviz.readthedocs.io/en/stable/manual.html
        graphviz官网 https://graphviz.gitlab.io/documentation/
        pydotplus   https://github.com/carlos-jenkins/pydotplus
        '''
        if not root: return

        dot = pdp.Dot(graph_type='digraph')
        # graph = pdp.Graph(graph_type='digraph')
        dot.set_node_defaults(shape='box')

        # 遍历获取决策树的父子节点关系
        nodes = cls.level_traversal(root)
        edges = cls.level_traversal(root, get_rel_pair=True)

        # 添加节点
        for tree_node in nodes:
            node = pdp.Node(name=tree_node.id, label=tree_node.kwargs_to_str())
            dot.add_node(node)

        # 添加边
        for pair in edges:
            label='Y' if pair[0].left == pair[1] else 'N'
            edge = pdp.Edge(src=pair[0].id, dst=pair[1].id, label=label)
            dot.add_edge(edge)

        if path:
            dot.write(path, format='png')
        return dot.to_string()

if __name__ == '__main__':
    root = BiNode(left=BiNode(xx=1, id=1, dd=1, yy=111),
                  right=BiNode(xx=2, id=2, dd=2, yy=222),
                  xx=0, id=0, dd=0.12, yy=333)
    BiTree.plot(root, path='imgs/test.png')