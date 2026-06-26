import jieba
import pandas as pd
from collections import Counter
from config import *


def count_data():
    # todo:1.加载数据
    data = pd.read_csv('data/train.txt', sep='\t', encoding='utf-8', names=['text', 'label'])
    data['words'] = data['text'].apply(cut_data)
    data['words_size'] = data['words'].str.len()
    label_count = Counter(data['label'])
    for label_idx, label_count in label_count.items():
        print(f"{label_idx}:{label_count / data['label'].count()}")
    print(label_count)
    print('总词数:', data['words_size'].sum())
    # 标准差
    print('标准差:', data['words_size'].std())
    print('最大词数:', data['words_size'].max())
    print('最短次数:', data['words_size'].min())
    print('均值:', data['words_size'].mean())
    print(data.head(5))


# todo:2.处理数据
def cut_data(text):
    text = ' '.join(jieba.lcut(text))[:40]
    return text


def process_data(data):
    data['words'] = data['text'].apply(cut_data)
    return data


def load_save_data(data_path, data_save_path):
    data = pd.read_csv(data_path, sep='\t', names=['text', 'label'])
    data_pre = process_data(data)
    data_pre.to_csv(data_save_path, index=False)


if __name__ == '__main__':
    # count_data()
    load_save_data(config.data.train_path, config.data.train_pre_path)
    load_save_data(config.data.test_path, config.data.test_pre_path)
    load_save_data(config.data.dev_path, config.data.dev_pre_path)
