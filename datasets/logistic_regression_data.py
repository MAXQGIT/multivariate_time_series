import pandas as pd
import glob
import argparse
from typing import *
import numpy as np
import json
import random

'''
1.必须有‘日期’列。
'''


def Argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoding', type=str, default='gbk', required=False)
    parser.add_argument('--target_name', type=str, default='OT')
    parser.add_argument('--history_len', type=int, default=15, required=False)
    parser.add_argument('--predict_len', type=int, default=3, required=False)
    args = parser.parse_args()
    return args


def read_data(data_path, args):
    data = pd.read_csv(data_path, encoding=args.encoding)
    return data


def drop_columns(data, columns: List[str]):
    data = data.drop(columns=columns)
    return data


def processing_date(data):
    data['date'] = pd.to_datetime(data['date'])
    data['year'] = data['date'].dt.year
    data['month'] = data['date'].dt.month
    data['day'] = data['date'].dt.day
    data = drop_columns(data, ['date'])
    return data


def split_data(label_target_data):
    random.shuffle(label_target_data)
    train_len = int(len(label_target_data) * 0.8)
    train_data = label_target_data[:train_len]
    test_data = label_target_data[train_len:]
    return train_data, test_data


def processing_data(data, args):
    label_target_data = []
    label_data = processing_date(data)
    target_data = data[args.target_name]
    for i in range(0, data.shape[0]):
        if i + args.history_len + args.predict_len > data.shape[0]:
            break
        label_seq = label_data.iloc[i:i + args.history_len, :].values.flatten()
        target_seq = target_data.iloc[i + args.history_len:i + args.history_len + args.predict_len].values.flatten()
        label_target_data.append((np.array(label_seq).tolist(), np.array(target_seq).tolist()))
    return label_target_data


def main(args):
    data_path_list = glob.glob('raw_data/*.csv')
    train_data_list, test_data_list = [], []
    for data_path in data_path_list:
        data = read_data(data_path, args)
        label_target_data = processing_data(data, args)
        train_data, test_data = split_data(label_target_data)
        train_data_list += train_data
        test_data_list += test_data
    with open('train_data/train_data.json', 'w') as f:
        json.dump(train_data_list, f, ensure_ascii=False, indent=4)
    with open('train_data/test_data.json', 'w') as f:
        json.dump(test_data_list, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    args = Argparse()
    main(args)
