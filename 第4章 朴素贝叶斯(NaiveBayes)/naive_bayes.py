# /usr/bin/env python3
# -*- coding:utf-8 -*-
from math import exp,log

class LaplaceEstimate():
    '''
    拉普拉斯平滑的贝叶斯估计 （对应一个分类）
    '''
    def __init__(self, lambda1=1):
        self.lambda1 = lambda1   # 常数
        self.total = 0           # 在分类 ci 下的总词频
        self.key_freq = {}       # 词-词频

    def getSum(self):
        return self.total

    def getProb(self, key):
        ''' 条件概率：p(xj=a|c) '''
        freq = self.lambda1      # 平滑处理
        if key in self.key_freq:
            freq = self.key_freq[key]
        return float(freq / self.total)

    def add(self, word):
        ''' 增加一个词 '''
        freq = 1
        if word in self.key_freq:
            freq += self.key_freq[word]
        self.key_freq[word] = freq
        self.total += 1

class NaiveBayes():
    ''' 朴素贝叶斯 '''
    def __init__(self, lambda1):
        self.tag_prob = {}   # 先验概率：标签(LaplaceEstimate对象)-概率
        self.total = 0      # 总词频
        self.lambda1 = lambda1

    def getTagPro(self, tag):
        if tag in self.tag_prob:
            return float(self.tag_prob[tag].getSum() / self.total)

    def train(self, data):
        for words, tag in data:
            if tag not in self.tag_prob:
                self.tag_prob[tag] = LaplaceEstimate(self.lambda1)
            for word in words:
                self.tag_prob[tag].add(word)
                self.total += 1

        print(self.tag_prob['pos'].key_freq, self.tag_prob['pos'].getSum())
        print(self.tag_prob['neg'].key_freq, self.tag_prob['neg'].getSum())

    def classify(self, data):
        tag_prob = {}
        for tag in self.tag_prob:
            prob = log(self.tag_prob[tag].getSum()) - log(self.total)    # P(Y=c)
            print('p(y=ck)',tag, prob, exp(prob))
            for word in data:
                prob += log(self.tag_prob[tag].getProb(word))           # p(xj=a|c)
                print(word, self.tag_prob[tag].getProb(word), log(self.tag_prob[tag].getProb(word)) )
            tag_prob[tag] = prob
            print(tag, prob, exp(prob), '\n')

        best_tag, max_prob, sum_prob = None, 0, 0   # p(xj=a|c) / sum( p(xj=a|ck) )
        for k,v in tag_prob.items():
            prob = exp(v)
            sum_prob += prob
            if prob > max_prob:
                max_prob = prob
                best_tag = k
            print(k, prob)
        return best_tag, max_prob/sum_prob

class Sentiment():
    '''情感分析'''
    def __init__(self, classifier):
        self.classifier = classifier

    def segment(self, data, separator=' '):
        return data.split(separator)

    def train(self, pos_data, neg_data):
        docs = []
        for doc in pos_data:
            docs.append([self.segment(doc), 'pos'])
        for doc in neg_data:
            docs.append([self.segment(doc), 'neg'])
        self.classifier.train(docs)

    def classify(self, sentence):
        return self.classifier.classify(self.segment(sentence))

if __name__ == '__main__':
    sentiment = Sentiment(NaiveBayes(lambda1=0.1))
    pos_data = ['好吃 有意思 很好','有意思 很好', '真棒 点赞']
    neg_data = ['难吃 差劲', '垃圾', '吃屎', '差劲']
    sentiment.train(pos_data, neg_data)
    print(sentiment.classify('优秀 好吃'))