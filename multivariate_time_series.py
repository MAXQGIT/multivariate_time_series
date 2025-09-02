from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split
import torch
import glob
import os
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
import logging
from importlib import import_module
from config import Argspare
from sklearn.preprocessing import StandardScaler


def StandardScalerData(data):
    columns = data.columns
    scler = StandardScaler()
    scler.fit(data.values)
    data = scler.transform(data.values)
    data = pd.DataFrame(data, columns=columns)
    return data


class TimeSeriesData(Dataset):
    def __init__(self, data, args):
        self.data = data
        self.history_len = args.history_len
        self.predict_len = args.predict_len
        self.is_scaler = args.is_scaler
        self.data = self._build_date()

    def _build_date(self):
        date = pd.DataFrame({'date': pd.to_datetime(self.data['date'])})
        date['year'] = date['date'].dt.year
        date['month'] = date['date'].dt.month
        date['day'] = date['date'].dt.day
        date = date.drop(columns='date')
        data = self.data.drop(columns='date')
        data = pd.concat([date, data], axis=1)
        if self.is_scaler:
            data = StandardScalerData(data)
        return data

    def __getitem__(self, item):
        x = self.data.iloc[item:item + self.history_len, :].values
        y = self.data.iloc[item + self.history_len:item + self.history_len + self.predict_len, -1].values
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return self.data.shape[0] - self.history_len - self.predict_len + 1


def ModelDatasets(args):
    data_path_list = glob.glob(os.path.join(args.data_path, '*.csv'))
    datasets_list = []
    for data_path in data_path_list:
        try:
            data = pd.read_csv(data_path, encoding='gbk')
        except UnicodeError:
            data = pd.read_csv(data_path, encoding='utf-8')
        datasets = TimeSeriesData(data, args)
        datasets_list.append(datasets)

    datasets = ConcatDataset(datasets_list)
    total_size = len(datasets)
    train_len = int(total_size * args.train_rate)
    val_len = int(train_len * args.val_rate)
    test_len = total_size - train_len - val_len
    train_dataset, val_dataset, test_dataset = random_split(datasets, [train_len, val_len, test_len])
    train_dataset = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers)
    val_dataset = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers)
    test_dataset = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers)
    return train_dataset, val_dataset, test_dataset


def val(args, model, loss_function, data):
    loss_list = []
    with torch.no_grad():
        for x,  y in data:
            x, y = x.to(args.device), y.to(args.device)
            predict = model(x)
            val_loss = loss_function(predict, y)
            loss_list.append(val_loss.item())
    val_loss = sum(loss_list) / len(loss_list)
    return val_loss


def train(args):
    os.makedirs(args.model_save_path, exist_ok=True)
    logging.basicConfig(filename=os.path.join(args.model_save_path, 'train.log'), filemode='w', encoding='gbk',
                        level=logging.INFO, format='%(asctime)s - %(message)s')
    logging.info(f"模型：{args.model}")
    train_dataset, val_dataset, test_dataset = ModelDatasets(args)
    model = import_module('deeplearning_model.' + args.model)
    model = model.Model(args).to(args.device)
    loss_function = nn.L1Loss()
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=args.step_size, gamma=args.gamma)
    train_loss_list, val_loss_list, test_loss_list = [], [], []
    start_loss = 100
    no_improve_count = 0
    for epoch in range(args.epochs):
        model.train()
        train_list = []
        for x,  y in train_dataset:
            x, y = x.to(args.device), y.to(args.device)
            predict = model(x)
            optim.zero_grad()
            train_loss = loss_function(predict, y)
            train_loss.backward()
            optim.step()
            train_list.append(train_loss.item())
        scheduler.step()
        train_loss = sum(train_list) / len(train_list)
        val_loss = val(args, model, loss_function, val_dataset)
        test_loss = val(args, model, loss_function, test_dataset)
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        test_loss_list.append(test_loss)
        logging.info(f'epoch:{epoch + 1} train_loss:{train_loss:4f} val_loss:{val_loss:.4f} test_loss:{test_loss:.4f}')
        print(f'epoch:{epoch + 1} train_loss:{train_loss:4f} val_loss:{val_loss:.4f} test_loss:{test_loss:.4f}')
        if val_loss < start_loss:
            start_loss = val_loss
            no_improve_count = 0
            torch.save(model, os.path.join(args.model_save_path, 'model.pt'))
        else:
            no_improve_count += 1
        if no_improve_count >= args.patience:
            break
    plot_loss(args, train_loss_list, val_loss_list, test_loss_list)


def plot_loss(args, train_loss_list, val_loss_list, test_loss_list):
    plt.figure()
    plt.plot(train_loss_list, label='train_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(args.model_save_path, 'train_loss.png'))
    plt.close()

    plt.figure()
    plt.plot(val_loss_list, label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(args.model_save_path, 'val_loss.png'))
    plt.close()

    plt.figure()
    plt.plot(test_loss_list, label='test_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(args.model_save_path, 'test_loss.png'))
    plt.close()


if __name__ == '__main__':
    args = Argspare()
    train(args)
