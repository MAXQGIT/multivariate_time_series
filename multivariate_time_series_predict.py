from config import Argspare
import torch
import os
import pandas as pd


def read_data(args):
    try:
        data = pd.read_csv(os.path.join(args.data_path, 'ETTm1.csv'), encoding='gbk')
    except UnicodeError:
        data = pd.read_csv(os.path.join(args.data_path, 'ETTm1.csv'), encoding='utf-8')
    data = build_data(data)
    index = 100
    data = data.iloc[index:index + args.history_len].values
    return data


def build_data(data):
    date = pd.DataFrame({'date': pd.to_datetime(data['date'])})
    date['year'] = date['date'].dt.year
    date['month'] = date['date'].dt.month
    date['day'] = date['date'].dt.day
    data = data.drop(columns='date')
    date = date.drop(columns='date')
    data = pd.concat([date, data], axis=1)
    return data


def predict(args):
    history_data = read_data(args)
    history_data = torch.tensor(history_data, dtype=torch.float32).to(args.device)
    history_data = torch.unsqueeze(history_data, 0)
    model = torch.load(os.path.join(args.model_save_path, 'model.pt'), weights_only=False)
    model = model.to(args.device)
    pre_values = model(history_data)
    pre_values = pre_values.tolist()[0]
    return pre_values


if __name__ == '__main__':
    args = Argspare()
    pre_values = predict(args)
    print(pre_values)

