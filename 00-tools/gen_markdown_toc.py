#!/usr/bin/env python3
# -*- coding:utf-8 -*--
# 安装intellij的python插件
# https://plugins.jetbrains.com/plugin/631-python
import os
import re


def generate_toc(md_content):
    code_block_pattern = re.compile(r'```[\s\S\n]*?```')
    # Remove code blocks temporarily
    temp_content = re.sub(code_block_pattern, '', md_content)

    toc = []
    heading_marks = re.findall(r'^(#{1,6})', temp_content, re.MULTILINE)
    headings = re.findall(r'^(#{1,6})\s*(.+)', temp_content, re.MULTILINE)
    # print(headings)

    mark2level = dict()
    for hm in heading_marks:
        marks = sorted(set(map(lambda x: (x, x.count("#")), heading_marks)), key=lambda x: x[1])
        n = len(marks)
        for level, (mark, mark_count) in enumerate(marks, start=1):
            mark2level[mark] = level

    for mark, heading in zip(heading_marks, headings):
        title = heading[1]
        anchor = re.sub(r'[^\w\s-]', '', title).strip().lower()
        anchor = re.sub(r'[-\s]+', '-', anchor)
        indent = '&nbsp;' * 4 * (mark2level.get(mark, 1) - 1)
        toc.append(f'{indent}<a href="#{anchor}">{title}</a><br/>\n')
    return '<nav>\n' + ''.join(toc) + '</nav>'


def process_markdown_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        md_content = file.read()

    # Check if TOC already exists
    if re.search(r'<nav>.*</nav>', md_content, re.DOTALL):
        print(f'Skipping {file_path}, TOC already exists.')
        return

    toc = generate_toc(md_content)
    new_md_content = toc + '\n\n' + md_content

    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(new_md_content)
    print(f'Processed {file_path}')


def traverse_directory(directory_path):
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.endswith('.md'):
                file_path = os.path.join(root, file)
                process_markdown_file(file_path)


if __name__ == "__main__":
    # process_markdown_file("../torch_learning/ch01_PyTorch简介与环境配置.md")
    process_markdown_file("../torch_learning/ch02_PyTorch核心模块.md")
