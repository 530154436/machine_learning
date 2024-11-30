#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @author: zhengchubin
# @time: 2022/7/15 17:18
# @function:
from typing import Set, Dict, Any, Generator


class DictionaryLoader(object):

    @classmethod
    def load(cls, path: str, lower=True) -> Set[str]:
        """
        加载各种词典
        :return {word1, word2}
        """
        words = set()
        if path:
            with open(path, mode='r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    if lower:
                        line = line.lower()
                    words.add(line)
        return words

    @classmethod
    def load_mapper(cls, path: str, separator=' ', lower=True, verbose=False) -> Dict[str, Any]:
        """
        加载二元组
        :return {word: value}
        """
        words: Dict[str, str] = {}
        print("加载词典: %s" % path)
        if path:
            with open(path, mode='r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    tokens = line.split(separator)
                    if len(tokens) == 2:
                        word, tag = tokens[0], tokens[1]
                        if lower:
                            word = word.lower().lstrip().rstrip()
                            tag = tag.lower().lstrip().rstrip()
                        words[word] = tag
                    else:
                        if verbose:
                            print("脏词 len(tokens) != 3: %s, %s" % (path, line))
        return words

    @classmethod
    def load_triple(cls, path: str, separator=' ', lower=True, verbose=False) -> Generator:
        """
        加载三元组
        :return {word: value}
        """
        print("加载词典: %s" % path)
        if path:
            with open(path, mode='r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    tokens = line.split(separator)
                    if len(tokens) == 3:
                        word, freq, tag = tokens[0], tokens[1], tokens[2]
                        if lower:
                            word = word.lower().strip()
                            freq = int(freq.lower().strip())
                            tag = tag.lower().strip()
                        yield word, freq, tag
                    else:
                        if verbose:
                            print("脏词 len(tokens) != 3: %s, %s" % (path, line))
