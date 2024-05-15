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

warnings.filterwarnings('ignore')


class Exp_Small(Exp_Basic):
    def __init__(self, args):
        super(Exp_Small, self).__init__(args)

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
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                outputs = self.model(batch_x)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')

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
            for i, (batch_x, batch_y) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                outputs = self.model(batch_x)

                loss = criterion(outputs, batch_y)
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
            for i, (batch_x, batch_y) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                outputs = self.model(batch_x)

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

        r2 = r2_score(trues, preds, multioutput='raw_values')
        r2 = np.concatenate([np.ones(63), r2])
        print('r2:{}'.format(r2.mean()))
        f = open("result_small.txt", 'a')
        f.write(setting + "  \n")
        f.write('r2:{}'.format(r2.mean()))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', r2)
        if self.args.save_results:
            np.save(folder_path + 'pred.npy', preds)
            np.save(folder_path + 'true.npy', trues)

        return
    
    def submit(self, setting, test=0):
        import polars as pl
        from sklearn.preprocessing import StandardScaler
        from joblib import load
        df_test = pl.read_parquet(self.args.test_path)
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        feature_scaler: StandardScaler = load(self.args.feature_scaler_path)
        target_scaler: StandardScaler = load(self.args.target_scaler_path)
        feature_name = feature_scaler.feature_names_in_.tolist()
        target_name = target_scaler.feature_names_in_.tolist()

        df_weight = pl.read_parquet(self.args.weight_path)
        weight = df_weight[target_name].to_numpy()

        if self.args.postprocess:
            r2 = np.load('./results/' + setting + '/' + 'metrics.npy')
            unpredict_target_index = np.nonzero(r2 < 0)[0]

        preds = []
        self.model.eval()
        with torch.no_grad():
            for chunk in df_test.iter_slices(self.args.batch_size):
                data_x = chunk[feature_name].to_numpy()
                data_x = feature_scaler.transform(data_x)
                data_x = torch.tensor(data_x).float().to(self.device)

                pred = self.model(data_x)
                
                pred = pred.detach().cpu().numpy()

                if self.args.postprocess:
                    pred[:, unpredict_target_index] = 0

                if self.args.inverse:
                    pred = target_scaler.inverse_transform(pred) * weight
                    pred = pl.DataFrame(pred, schema=target_name, orient="row")

                preds.append(pl.concat([chunk[["sample_id"]], pred], how="horizontal"))

        preds = pl.concat(preds, how="vertical")
        zero_target = [x for x in df_weight.columns if x not in target_name + ["sample_id"]]
        assert len(zero_target) == 63
        preds = pl.concat([preds, pl.DataFrame(np.zeros(shape=(len(preds), len(zero_target))), schema=zero_target, orient="row")], how="horizontal")

        folder_path = './output/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        preds[df_weight.columns].write_parquet(folder_path + "submission.parquet")
