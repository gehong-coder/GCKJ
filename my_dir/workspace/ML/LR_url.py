# coding: utf-8
import os
import urllib
import time
import html
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import pandas as pd


def read_data(file_path):
    file_path = file_path
    # df = pd.read_csv(file_path, sep='\t')
    df = open(file_path, 'r', encoding='utf-8').readlines()
    # print(df.head)
    url_list = []
    for i in df:
        d = str(urllib.parse.unquote(i))
        url_list.append(d.strip())
    return list(url_list)

# tokenizer function, this will make 3 grams of each query
# www.foo.com/1 转换为 ['www','ww.','w.f','.fo','foo','oo.','o.c','.co','com','om/','m/1']


def get_ngrams(query):
    tempQuery = str(query)
    ngrams = []
    for i in range(0, len(tempQuery)-3):
        ngrams.append(tempQuery[i:i+3])
    return ngrams


if __name__ == '__main__':
    good_data = read_data(
        file_path='/Users/gehong/Documents/研究生/实习/GCKJ/my_dir/data/url_detect_data/good.txt')
    print("正常请求：{}".format(good_data[0:5]))
    bad_data = read_data(
        '/Users/gehong/Documents/研究生/实习/GCKJ/my_dir/data/url_detect_data/bad.txt')
    print("恶意请求：{}".format(bad_data[0:5]))
    # 构造标签
    good_y = [0 for i in range(0, len(good_data))]
    bad_y = [1 for i in range(0, len(bad_data))]
    print(good_y[0:5])
    print(bad_y[0:5])

    # 数据集
    all_data = bad_data+good_data
    all_label = bad_y+good_y

    # tf-idf特征构造
    # 把不规律的文本字符串列表转换成规律的 ( [i,j], tdidf值) 的矩阵X
    # 用于下一步训练逻辑回归分类器
    vectorizer1 = TfidfVectorizer(tokenizer=get_ngrams)
    all_data_tiidf = vectorizer1.fit_transform(all_data)
    X1_F = vectorizer1.get_feature_names()

    # 数据集分割
    X_train, X_test, y_train, y_test = train_test_split(
        all_data_tiidf, all_label, test_size=20, random_state=42)

    # 逻辑回归
    LR = LogisticRegression()

    LR.fit(X_train, y_train)

    # 测试
    print("模型准确度：{}".format(LR.score(X_test, y_test))
          )
    new_url = ['www.foo.com/id=1<script>alert(1)</script>',
               'www.foo.com/name=admin\' or 1=1', 'abc.com/admin.php',
               '"><svg onload=confirm(1)>']
    X_pre = vectorizer1.transform(new_url)
    re = LR.predict(X_pre)
    res_list = []

    # 结果输出
    for q, r in zip(new_url, re):
        tmp = '正常请求' if r == 0 else '恶意请求'
        q_entity = html.escape(q)
        res_list.append({'url': q_entity, 'res': tmp})

    for n in res_list:
        print(n)
