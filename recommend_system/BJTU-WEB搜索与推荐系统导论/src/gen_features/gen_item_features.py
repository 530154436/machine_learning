#!/usr/bin/env python3
# -*- coding:utf-8 -*--
import gc
import jieba
# import sys
# sys.path.append("/root/zhengchubin/programing")
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from bjtu_programming.search_rs import DATA_DIR, BASE_DIR
from bjtu_programming.search_rs.src.dataset.loader_item import ItemLoader
from bjtu_programming.search_rs.src.dataset.loader_train import TrainLoader
from bjtu_programming.search_rs.src.utils.io.dictionary import DictionaryLoader
from bjtu_programming.search_rs.src.utils.pd_opt_util import MemoryOptimizer
from bjtu_programming.search_rs.src.utils.pickle_util import PickleUtil


ITEM_LABEL_ENCODER = {
    'i_category1': LabelEncoder(),
    'i_category2': LabelEncoder(),
}


def gen_item_static_feature():

    def safe_len(x: str):
        if not isinstance(x, list) and pd.isna(x):
            return 0
        return len(x)

    def get_keys(x: str):
        if pd.isna(x):
            return []
        d = []
        for item in str(x).split(','):
            item = item.replace('^', '').split(':')
            if len(item) < 2:
                continue
            d.append((''.join(item[:-1]), float(item[-1])))
        d = sorted(d, key=lambda w_prob: w_prob[1], reverse=True)
        d = list(map(lambda w: w[0], d))
        # print(d)
        return d

    def cut_words(x: str):
        if pd.isna(x):
            return []
        words = []
        for w in jieba.cut(str(x)):
            if w in stop_words:
                continue
            words.append(w)
        # print(words)
        return words

    print("生成Item静态属性...")
    stop_words = DictionaryLoader.load(BASE_DIR.joinpath('src', 'dataset', 'stop_words.txt'))
    df_item = ItemLoader().load_csv()
    print("处理标题...")
    df_item['i_title_char_len'] = df_item['i_title'].parallel_apply(safe_len)
    df_item['i_title'] = df_item['i_title'].parallel_map(cut_words)
    df_item['i_title_word_len'] = df_item['i_title'].parallel_apply(safe_len)

    print("处理关键词...")
    df_item['i_keys_prob'] = df_item['i_keys_prob'].parallel_map(get_keys)
    df_item['i_keys_len'] = df_item['i_keys_prob'].parallel_apply(safe_len)

    for cat_feat, encoder in ITEM_LABEL_ENCODER.items():
        df_item[cat_feat] = encoder.fit_transform(df_item[cat_feat])

    print("生成Item静态属性:", df_item.shape)
    print(df_item.head())
    df_item = MemoryOptimizer.reduce_mem_usage(df_item)

    # 保存Item-分词结果
    i_keys = df_item.set_index('item_id')['i_keys_prob']
    PickleUtil.save2pkl(i_keys, ItemLoader().get_dict_pkl_name('keys'))
    i_keys = df_item.set_index('item_id')['i_title']
    PickleUtil.save2pkl(i_keys, ItemLoader().get_dict_pkl_name('title'))

    df_item = df_item.drop(columns=['i_title', 'i_keys_prob'])
    PickleUtil.save2pkl(df_item, ItemLoader().get_pkl_name())
    PickleUtil.save2pkl(ITEM_LABEL_ENCODER, ItemLoader().get_label_encoder_pkl_name())
    del df_item
    gc.collect()


def gen_item_agg_feature():
    df_tra = TrainLoader().load_pkl_by_valid(valid=False)
    df_tra = df_tra[['user_id', 'item_id', 'is_click']]

    # 新闻曝光数量
    print("生成Item统计特征: 新闻被点击数量...")
    df_item = df_tra.groupby(by=['item_id']).agg(
        i_click_cnt=pd.NamedAgg(column="is_click", aggfunc='sum'),
    )
    df_item = df_item.reset_index()
    print("生成Item统计特征:", df_item.shape)
    print(df_item.head())
    print(df_item.describe())
    df_item = MemoryOptimizer.reduce_mem_usage(df_item)
    PickleUtil.save2pkl(df_item, ItemLoader().get_agg_pkl_name())

    del df_tra, df_item
    gc.collect()


def gen_item_click_seqs():
    # 生成物品点击序列
    df_tra: pd.DataFrame = TrainLoader().load_pkl_by_valid(valid=False)
    df_tra = df_tra[df_tra['is_click'] == 1]
    df_tra = df_tra[['user_id', 'item_id', 'ctx_timestamp_expose']]

    item_click_seq = dict()
    print('生成item点击序列..')
    for item_id, chunk in df_tra.groupby(by=['item_id']):

        chunk = chunk.sort_values(by='ctx_timestamp_expose', ascending=False)
        item_click_seq[item_id] = chunk['user_id'].tolist()

    PickleUtil.save2pkl(item_click_seq, ItemLoader().get_seq_pkl_name('click_seq'))


def gen_item_features():
    # gen_item_static_feature()
    gen_item_agg_feature()
    # gen_item_click_seqs()


if __name__ == '__main__':
    pd.set_option('display.width', 800)  # 设置打印宽度
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)

    from pandarallel import pandarallel
    pandarallel.initialize()
    gen_item_features()
