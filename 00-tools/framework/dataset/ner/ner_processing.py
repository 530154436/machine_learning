#!/usr/bin/env python3
# -*- coding:utf-8 -*--
from typing import List, Tuple, Iterable

import torch
from transformers import AutoTokenizer, PreTrainedTokenizer

from src.config_gpu import logger
from src.framework.dataset import BertFeat


def convert_text_to_features(sentences: List[str],
                             tokenizer: PreTrainedTokenizer,
                             max_seq_length: int,
                             add_special_tokens=True) -> Tuple:
    """
    将输入的文本转换为bert的输入特征（会进行分词、过滤的操作） => 和原始的文本信息有出入
    """
    feature = tokenizer.batch_encode_plus(sentences,
                                          return_tensors="pt",
                                          add_special_tokens=add_special_tokens,
                                          max_length=max_seq_length,
                                          padding=True,
                                          truncation=True)
    input_ids = feature.get("input_ids")
    input_mask = feature.get("attention_mask")
    segment_ids = feature.get("token_type_ids")
    return input_ids, input_mask, segment_ids


def convert_text_to_features_v2(sentences: List[str],
                                tokenizer: PreTrainedTokenizer,
                                max_seq_length: int,
                                add_special_tokens=True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    将输入的文本转换为bert的输入特征、和原始的文本信息一致
    """
    if not sentences:
        return None, None, None
    all_input_ids, all_input_mask, all_segment_ids, all_label_ids = [], [], [], []

    # 找到满足所有文本的最小最大长度
    min_max_seq_length = min(max([len(seq) for seq in sentences]), max_seq_length)

    for sentence in sentences:
        tokens = [tokenizer.cls_token] + list(sentence)[:min_max_seq_length-2] + [tokenizer.sep_token]
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        segment_ids = [0] * len(input_ids)

        # 补齐最大长度
        pad_length = min_max_seq_length - len(input_ids)
        input_ids += [tokenizer.pad_token_id] * pad_length
        input_mask += [0] * pad_length
        segment_ids += [0] * pad_length

        all_input_ids.append(input_ids)
        all_input_mask.append(input_mask)
        all_segment_ids.append(segment_ids)
    return torch.LongTensor(all_input_ids), torch.LongTensor(all_input_mask), torch.LongTensor(all_segment_ids)


def convert_example2features(examples: Iterable[Tuple[List[str], List[str]]],
                             label_list: List[str],
                             max_seq_length: int,
                             tokenizer: PreTrainedTokenizer,
                             pad_token_label_id=0) -> Iterable[BertFeat]:
    """
    将训练数据转换为bert的格式
    输入：[(["北", "京", "城"], ["B-NT", "I-NT", "I-NT"])]
    输出：
        tokens: 		[CLS] 北 京 城 [SEP]
        input_ids: 		101 1266 776 1814 102 0 0 0 0 0
        input_mask: 	1 1 1 1 1 0 0 0 0 0
        segment_ids: 	0 0 0 0 0 0 0 0 0 0
        label_ids: 		0 1 2 2 0 0 0 0 0 0
    示例：
        _examples = [("北京城", ["B-NT", "I-NT", "I-NT"])]
        _label_list = ["O", "B-NT", "I-NT"]
        _tokenizer = AutoTokenizer.from_pretrained(BASE_DIR.joinpath("data", "modelfiles", "bert-base-chinese"))
        f = convert_example2features(_examples, _label_list, max_seq_length=10, tokenizer=_tokenizer)
        print(len(list(f)))
    """
    label_map = {label: index for index, label in enumerate(label_list)}
    label_pad_id = label_map.get("O")
    ex_index = 0
    for ex_index, (text, labels) in enumerate(examples, start=1):
        label_ids, tokens, origin_tokens = [], [], []
        for word, label in zip(text, labels):
            word_tokens = tokenizer.tokenize(word)
            if not word_tokens:
                continue
            origin_tokens.append(word)
            tokens.extend(word_tokens)
            # 对于多个字符的词，仅保留第1个字对应的标签
            label_ids.extend([label_map[label]] + [pad_token_label_id] * (len(word_tokens) - 1))
        labels_len = len(label_ids)
        tokens_len = len(tokens)
        if len(tokens) > max_seq_length - 2:
            tokens = tokens[: (max_seq_length - 2)]
            label_ids = label_ids[: (max_seq_length - 2)]

        tokens = [tokenizer.cls_token] + tokens + [tokenizer.sep_token]
        origin_tokens = [tokenizer.cls_token] + origin_tokens + [tokenizer.sep_token]
        label_ids = [label_pad_id] + label_ids + [label_pad_id]
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_len = len(input_ids)
        input_mask = [1] * len(input_ids)
        segment_ids = [0] * len(input_ids)
        segment_len = len(segment_ids)
        # 补齐最大长度
        pad_length = max_seq_length - len(input_ids)
        input_ids += [tokenizer.pad_token_id] * pad_length
        input_mask += [0] * pad_length
        label_ids += [label_pad_id] * pad_length
        segment_ids += [0] * pad_length
        if len(label_ids) != max_seq_length:
            logger.info("*** Error ***")
            logger.info("origin tokens: %s" % " ".join([str(x) for x in origin_tokens]))
            logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info("label_ids: %s" % " ".join([str(x) for x in label_ids]))
            logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("labels_len: %s" % str(labels_len))
            logger.info("tokens_len: %s" % str(tokens_len))
            logger.info("input_len: %s" % str(input_len))
            logger.info("segment_ids: %s" % str(segment_len))
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length

        if ex_index <= 5:
            logger.info("*** Example ***")
            logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label_ids: %s" % " ".join([str(x) for x in label_ids]))

        if ex_index % 1000 == 0:
            logger.info("processing example: no.%s", ex_index)

        yield BertFeat(input_ids=input_ids,
                       input_mask=input_mask,
                       segment_ids=segment_ids,
                       label_ids=label_ids)
    logger.info("processing done, total = %s", ex_index)


class NERDataProcessor(object):

    @staticmethod
    def read_sample_from_file(file_path: str, sep: str = " ") -> Tuple[List[str], List[str]]:
        """
        加载文件、预处理
        :param file_path: file path
        :param sep: seg
        :return: [(sentence, label), ...]
        """
        sentence = []
        label = []
        logger.info("loading file: %s", file_path)
        with open(file_path, 'r', encoding='utf8') as reader:
            for line in reader:
                line = line.strip()
                splits = line.split(sep)
                if len(splits) != 2 and len(sentence) > 0:
                    yield sentence, label
                    sentence.clear()
                    label.clear()
                elif len(splits) != 2:
                    continue
                else:
                    sentence.append(splits[0])
                    label.append(splits[-1])


if __name__ == "__main__":
    _examples = [(list("北京城"), ["B-NT", "I-NT", "I-NT"])]
    _label_list = ["O", "B-NT", "I-NT"]
    _tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained("/data/modelfiles/bert-base-chinese")
    f = convert_example2features(_examples, _label_list, max_seq_length=10, tokenizer=_tokenizer)
    print(len(list(f)))

    _text = [f"{_tokenizer.pad_token}"
             f"{_tokenizer.unk_token}"
             f"{_tokenizer.cls_token}"
             f"{_tokenizer.sep_token}"
             f"N202306010山东 字科技ACB公告"]
    _input_ids, _, _ = convert_text_to_features(_text, _tokenizer, max_seq_length=20)
    print(_text)
    for i in _input_ids:
        print(i)
        print(_tokenizer.convert_ids_to_tokens(i))
    print()

    _input_ids, _, _ = convert_text_to_features_v2(_text, _tokenizer, max_seq_length=20)
    print(_text)
    for i in _input_ids:
        print(i)
        print(_tokenizer.convert_ids_to_tokens(i))
    print()
