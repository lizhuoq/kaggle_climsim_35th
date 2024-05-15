from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from utils.dtw_metric import dtw,accelerated_dtw
from utils.augmentation import run_augmentation,run_augmentation_single
from sklearn.metrics import r2_score
import polars as pl
from joblib import load
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')


class Exp_1D(Exp_Basic):
    def __init__(self, args):
        super(Exp_1D, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        cal_grad_index = np.nonzero(vali_data.weight[0] != 0)[0].tolist()
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (seq_x, x, seq_y, y) in enumerate(vali_loader):
                seq_x = seq_x.float().to(self.device)
                x = x.float().to(self.device)
                seq_y = seq_y.float().to(self.device)
                y = y.float().to(self.device)

                seq_o, o = self.model(seq_x, x)
                
                outputs = torch.cat([seq_o.reshape(seq_o.shape[0], -1), o], dim=1)
                batch_y = torch.cat([seq_y.reshape(seq_y.shape[0], -1), y], dim=1)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred[:, cal_grad_index], true[:, cal_grad_index])

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        cal_grad_index = np.nonzero(vali_data.weight[0] != 0)[0].tolist()

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (seq_x, x, seq_y, y) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                seq_x = seq_x.float().to(self.device)
                x = x.float().to(self.device)
                seq_y = seq_y.float().to(self.device)
                y = y.float().to(self.device)

                seq_o, o = self.model(seq_x, x)

                outputs = torch.cat([seq_o.reshape(seq_o.shape[0], -1), o], dim=1)
                batch_y = torch.cat([seq_y.reshape(seq_y.shape[0], -1), y], dim=1)
                loss = criterion(outputs[:, cal_grad_index], batch_y[:, cal_grad_index])
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []

        self.model.eval()
        with torch.no_grad():
            for i, (seq_x, x, seq_y, y) in enumerate(test_loader):
                seq_x = seq_x.float().to(self.device)
                x = x.float().to(self.device)
                seq_y = seq_y.float().to(self.device)
                y = y.float().to(self.device)

                seq_o, o = self.model(seq_x, x)

                outputs = torch.cat([seq_o.reshape(seq_o.shape[0], -1), o], dim=1)
                batch_y = torch.cat([seq_y.reshape(seq_y.shape[0], -1), y], dim=1)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                if self.args.inverse:
                    outputs = test_data.multiply_weight(test_data.inverse_transform(outputs))
                    batch_y = test_data.multiply_weight(test_data.inverse_transform(batch_y))

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # mae, mse, rmse, mape, mspe = metric(preds, trues)
        r2 = r2_score(trues, preds, multioutput="raw_values")
        print('r2:{}'.format(r2.mean()))
        f = open("result_1d.txt", 'a')
        f.write(setting + "  \n")
        f.write('r2:{}'.format(r2.mean()))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', r2)
        if self.args.save_results:
            preds = pl.DataFrame(preds, schema=test_data.targets, orient="row")
            trues = pl.DataFrame(trues, schema=test_data.targets, orient="row")
            preds.write_parquet(folder_path + 'pred.parquet')
            trues.write_parquet(folder_path + 'true.parquet')
        
        return
