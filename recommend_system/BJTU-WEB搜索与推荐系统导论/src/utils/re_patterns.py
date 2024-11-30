#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @author: zhengchubin
# @time: 2022/7/8 18:05
# @function:
import re


class RegexPattern(object):

    # HTML标签
    HTML_TAG = re.compile(r"""([&\-_:./0-9a-zA-Z]+(jpg))|(<.*?>)|(\${[a-zA-Z]+?})|(&[a-zA-Z0-9]+)""", re.DOTALL)

    # URL标签
    URI_TAG = re.compile(r"""(http|www)+?[\-，。；！‘“、!#%&'()+,-./:;<=>?@\[\]^_`{|}~？：”’￥…（）丨,.0-9a-zA-Z]+""")

    # 图片标签
    IMG_PATTERN_TAG = re.compile(r"""(<*?img.*?src.*?>)""", re.DOTALL)

    # JS脚本过滤
    JAVA_SCRIPT_TAG = re.compile(r"""<script.*?/script>""", re.DOTALL)

    # 标点符号标记
    PUNCTUATION_TAG = re.compile(
        r"""[�ǻԸˮȫԴΪ，。；！‘“、!#$%&'()*+,-./:;<=>?@\[\]^_`{|}~？：”’￥…（）《\"》【】，-；\u0139\xa0\u25ca\u200b\u3000\t\n\r②丨.,—\s]+""",
        re.DOTALL)

    # 纯数字
    NUM_TAG = re.compile("[0-9]+", re.DOTALL)

    # 换行符、空格、制表符、特殊字符
    BLANK_TAG = re.compile(r"""[\n\r\t\s]+""", re.DOTALL)

    # SQL的所有列名
    COLUMN_PATTERN = re.compile(r"""select (.*?) from """)

    # 汉字
    HAN_PATTERN = re.compile(r"""([\u4E00-\u9FD5a-zA-Z0-9+#&._]+)""")
