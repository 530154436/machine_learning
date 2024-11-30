#!/usr/bin/env python3
# -*- coding:utf-8 -*--
import os
import gc
import time
import numpy as np
from scipy import sparse
from functools import partial
from multiprocessing import Pool
'''
    Numpy =格式转换=> libfm、libffm
    参考: sklearn.datasets.dump_svmlight_file/load_svmlight_file
'''
# ------------------------------------------------------------------------------------
# 单线程版本
# ------------------------------------------------------------------------------------
class Converter():

    def __init__(self,  X, y, file_path, fields=None, threshold=1e-6):
        '''
        Write data to libsvm or libffm.
        =Ref=> https://github.com/scikit-learn/scikit-learn/blob/a24c8b46/sklearn/datasets/svmlight_format.py
               from xlearn import write_data_to_xlearn_format
        :param X: array-like
                 Feature matrix in numpy or sparse format
        :param y: array-like
                 Label in numpy or sparse format
        :param filepath: file location for writing data to
        :param fields: An array specifying fields in each columns of X. It should have same length
           as the number of columns in X. When set to None, convert data to libsvm format else
           libffm format.
        :param threshold: x = x if x>=threshold else 0
        '''
        self.X = X
        self.y = y
        self.file_path = file_path
        self.fields = fields
        self.threshold = threshold

        self.dtype_kind = X.dtype.kind # i:整型
        self.is_ffm_format = True if fields is not None else False
        self.X_is_sp = int(hasattr(X, "tocsr"))
        self.y_is_sp = int(hasattr(y, "tocsr"))

        assert len(X.shape)==2

    def label_format(self, value):
        '''
        标签格式
        :param value:       标签值
        :param dtype_kind:  数据类型
        '''
        return "%d" %value if  self.dtype_kind== 'i' else ("%.6g" % value)

    def fm_format(self, feature_id, value):
        '''
        FM 特征格式
        :param feature_id:  特征id (列索引)
        :param value:       特征值
        '''
        return "%d:%d"%(feature_id,value) if self.dtype_kind=='i' else ("%d:%.6g" % (feature_id, value))

    def ffm_format(self, field_id, feature_id, value):
        '''
        FFM 特征格式
        :param field_id     特征域id(对于多分类变量，包含多个列索引同属一个field) [0,0,0,1,2]
        :param feature_id:  特征id(列索引)
        :param value:       特征值
        '''
        return "%d:%s" % (field_id, self.fm_format(feature_id, value))

    def process_row(self, row_idx):
        '''
        根据行索引转换每一行
        :param row_idx: 行号
        :return:
        '''
        if self.X_is_sp:
            span = slice(self.X.indptr[row_idx], self.X.indptr[row_idx + 1])
            x_indices = self.X.indices[span]
            row = zip(self.fields[x_indices], x_indices, self.X.data[span]) if self.is_ffm_format \
                else zip(x_indices, self.X.data[span])
        else:
            nz = self.X[row_idx] != 0
            # print(nz)
            row = zip(self.fields[nz], np.where(nz)[0], self.X[row_idx, nz]) if self.is_ffm_format \
                else zip(np.where(nz)[0], self.X[row_idx, nz])

        if self.is_ffm_format:
            s = " ".join(self.ffm_format(f, j, x) for f, j, x in row)
        else:
            s = " ".join(self.fm_format(j, x) for j, x in row)

        if self.y_is_sp:
            labels_str = self.label_format(self.y.data[row_idx])
        else:
            labels_str = self.label_format(self.y[row_idx])

        return "%s %s" % (labels_str, s)

    def convert(self):
        with open(self.file_path, "w") as f_handle:
            start = time.time()
            for row_idx in range(self.X.shape[0]):
                f_handle.write(f'{self.process_row(row_idx)}\n')
            print(f'All Finished. cost: {time.time()-start}s')

# ------------------------------------------------------------------------------------
# 多进程版本
# ------------------------------------------------------------------------------------
def _label_format(value, dtype_kind):
    return "%d" %value if  dtype_kind== 'i' else ("%.6g" % value)

def _fm_format(feature_id, value, dtype_kind):
    return "%d:%d"%(feature_id,value) if dtype_kind=='i' else ("%d:%.6g" % (feature_id, value))

def _ffm_format(field_id, feature_id, value, dtype_kind):
    return "%d:%s" % (field_id, _fm_format(feature_id, value, dtype_kind))

def _process_row(X, y, fields, row_idx, X_is_sp, y_is_sp, is_ffm_format, dtype_kind):
    '''
    根据行索引转换每一行
    :param row_idx: 行号
    :return:
    '''
    if X_is_sp:
        span = slice(X.indptr[row_idx], X.indptr[row_idx + 1])
        x_indices = X.indices[span]
        row = zip(fields[x_indices], x_indices, X.data[span]) if is_ffm_format \
            else zip(x_indices, X.data[span])
    else:
        nz = X[row_idx] != 0
        # print(nz)
        row = zip(fields[nz], np.where(nz)[0], X[row_idx, nz]) if is_ffm_format \
            else zip(np.where(nz)[0], X[row_idx, nz])

    if is_ffm_format:
        s = " ".join(_ffm_format(f, j, x, dtype_kind) for f, j, x in row)
    else:
        s = " ".join(_fm_format(j, x, dtype_kind) for j, x in row)

    if y_is_sp:
        labels_str = _label_format(y.data[row_idx], dtype_kind)
    else:
        labels_str = _label_format(y[row_idx], dtype_kind)

    return "%s %s\n" % (labels_str, s)

def _process_chunk(X_is_sp, y_is_sp, is_ffm_format, dtype_kind, chunk):
    lines = []
    if is_ffm_format:
        x, y, fields = chunk[0], chunk[1], chunk[2]
        print(f'Process-{os.getpid()}, chunk_size={x.shape[0]}')
        for row_idx in range(x.shape[0]):
            lines.append(_process_row(x, y, fields, row_idx, X_is_sp, y_is_sp, is_ffm_format, dtype_kind))
    else:
        x, y = chunk[0], chunk[1]
        print(f'Process-{os.getpid()}, chunk_size={x.shape[0]}')
        for row_idx in range(x.shape[0]):
            lines.append(_process_row(x, y, None, row_idx, X_is_sp, y_is_sp, is_ffm_format, dtype_kind))
    return ''.join(lines)

def convert_by_parallel(X, y, file_path, fields=None, n_jobs=8, chunk_size=10):
    '''
    多进程版本，转换为 libsvm、libffm 数据格式
    :param X:           仅支持 scipy.csr_matrix 或 np.array
    :param y:           标签集 np.array
    :param file_path:   保存路径
    :param fields:      特征域 np.array
    :param n_jobs:      进程数
    :param chunk_size:  分块大小
    :return:
    '''
    # 根据索引分块
    chunks = []
    for i in range(0, X.shape[0], chunk_size):
        indices = np.arange(i, i+chunk_size if i+chunk_size<X.shape[0] else X.shape[0])
        # print(indices)
        if fields is not  None:
            chunks.append((X[indices], y[indices], fields))
        else:
            chunks.append((X[indices], y[indices]))

    is_ffm_format = True if fields is not  None else False
    dtype_kind = X.dtype.kind  # i:整型
    X_is_sp = int(hasattr(X, "tocsr"))
    y_is_sp = int(hasattr(y, "tocsr"))

    del X,y
    gc.collect()

    with open(file_path, "w") as f_handle, Pool(processes=n_jobs) as pool:
        start = time.time()
        # 多进程、保证顺序、chunksize(每个进程分配的元素个数)
        for res in pool.imap(func=partial(_process_chunk, X_is_sp, y_is_sp, is_ffm_format, dtype_kind),
                             iterable=chunks, chunksize=10):
            f_handle.write(res)
            f_handle.flush()
        print(f'All Finished. cost: {time.time()-start}s')

if __name__ == '__main__':
    X = np.random.randint(0, 11, (1000, 10))
    # X = sparse.csr_matrix(np.random.randint(0, 5, (100, 5)))
    y = np.random.randint(0, 2, 1000)
    print(X)
    print(y)
    # fields = np.random.randint(0, 11, 10)
    fields = None
    convert_by_parallel(X, y, 'tmp01.txt', fields=fields, n_jobs=8, chunk_size=100)

    ctr = Converter(X, y, 'tmp02.txt', fields=fields)
    ctr.convert()