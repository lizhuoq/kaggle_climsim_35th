from data_provider.data_factory import data_provider, data_provider_2D
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


class Exp_2D(Exp_Basic):
    def __init__(self, args):
        super(Exp_2D, self).__init__(args)

    def _build_model(self):
        model: nn.Module = self.model_dict[self.args.model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider_2D(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        if self.args.optimizer == "adam":
            print("Optimizer: Adam")
            model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
            return model_optim
        elif self.args.optimizer == "sgd":
            print("Optimizer: SGD")
            model_optim = optim.SGD(self.model.parameters(), lr=self.args.learning_rate, momentum=0.9, weight_decay=self.args.weight_decay)
            return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion, r2=True):
        cal_grad_targets = np.nonzero(vali_data.weight[0] != 0)[0].tolist()
        no_grad_targets = vali_data.get_no_grad_targets()
        scale_, mean_ = vali_data.get_last_eight_std_mean()
        scale_, mean_ = torch.tensor(scale_).float().to(self.device), torch.tensor(mean_).float().to(self.device)
        cal_grad_targets = [x for x in cal_grad_targets if x not in no_grad_targets]
        total_loss = []
        
        if r2:
            preds = []
            trues = []

        self.model.eval()
        with torch.no_grad():
            for i, (level_x, single_x, level_y, single_y) in enumerate(vali_loader):
                batch_x = torch.cat([level_x, single_x.unsqueeze(-1).repeat(1, 1, 1, 60)], dim=-2) # B, G, C, L
                level_y = level_y.reshape(-1, 6, 60).reshape(-1, 360)
                single_y = single_y.reshape(-1, 8)
                batch_y = torch.cat([level_y, single_y], dim=1)
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                
                outputs = self.model(batch_x)

                outputs = outputs.reshape(-1, 6 + 8, 60)
                outputs_level = outputs[:, :6, :].reshape(-1, 6 * 60)
                outputs_single = outputs[:, 6:, :].mean(dim=2)
                outputs = torch.cat([outputs_level, outputs_single], dim=1)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                if r2:
                    pred = vali_data.inverse_transform(pred.numpy())
                    pred[:, -8:] = np.where(pred[:, -8:] < 0, 0, pred[:, -8:])
                    pred = vali_data.multiply_weight(pred)
                    true = vali_data.multiply_weight(vali_data.inverse_transform(true.numpy()))
                    preds.append(pred)
                    trues.append(true)
                else:
                    pred[:, -8:] = torch.nn.functional.relu(pred[:, -8:] * scale_.to(pred.device) + mean_.to(pred.device))
                    pred[:, -8:] = (pred[:, -8:] - mean_.to(pred.device)) / scale_.to(pred.device) 
                    loss = criterion(pred[:, cal_grad_targets], true[:, cal_grad_targets]).mean()

                    total_loss.append(loss)

        if r2:
            preds = np.concatenate(preds, axis=0)
            trues = np.concatenate(trues, axis=0)
            metric = r2_score(trues[:, cal_grad_targets], preds[:, cal_grad_targets], multioutput="raw_values")
            weight = 1 - metric
            weight = weight / weight.sum()
            metric = np.concatenate([metric, np.ones(preds.shape[1] - len(cal_grad_targets))]).mean() * -1
        else:
            total_loss = np.average(total_loss)
        self.model.train()
        if r2:
            return metric, weight
        return total_loss
    
    def finetune(self, setting):
        print('loading model')
        from utils.tools import load_partial_state_dict
        load_partial_state_dict(self.model, torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))
        return self.train(setting)

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        cal_grad_targets = np.nonzero(vali_data.weight[0] != 0)[0].tolist()
        no_grad_targets = vali_data.get_no_grad_targets()
        cal_grad_targets = [x for x in cal_grad_targets if x not in no_grad_targets]
        scale_, mean_ = vali_data.get_last_eight_std_mean()
        scale_, mean_ = torch.tensor(scale_).float().to(self.device), torch.tensor(mean_).float().to(self.device)

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.is_training == -2:
            # vali_loss = np.load('./results/' + setting + '/' + 'metrics.npy').mean() * -1
            if self.args.finetune_vali_loss is None:
                vali_loss, vali_weight = self.vali(vali_data, vali_loader, criterion)
            else:
                vali_loss = self.args.finetune_vali_loss * -1
            print('r2:{}'.format(vali_loss * -1))
            early_stopping(vali_loss, self.model, path, init_save=False)
        else:
            vali_weight = np.ones(len(cal_grad_targets)) / len(cal_grad_targets)

        vali_weight = np.ones(len(cal_grad_targets)) / len(cal_grad_targets)
        
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (level_x, single_x, level_y, single_y) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = torch.cat([level_x, single_x.unsqueeze(-1).repeat(1, 1, 1, 60)], dim=-2) # B, G, C, L
                level_y = level_y.reshape(-1, 6, 60).reshape(-1, 360)
                single_y = single_y.reshape(-1, 8)
                batch_y = torch.cat([level_y, single_y], dim=1)
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                
                outputs = self.model(batch_x)

                outputs = outputs.reshape(-1, 6 + 8, 60)
                outputs_level = outputs[:, :6, :].reshape(-1, 6 * 60)
                outputs_single = outputs[:, 6:, :].mean(dim=2)
                outputs = torch.cat([outputs_level, outputs_single], dim=1)

                outputs[:, -8:] = torch.nn.functional.relu(outputs[:, -8:] * scale_.to(outputs.device) + mean_.to(outputs.device))
                outputs[:, -8:] = (outputs[:, -8:] - mean_.to(outputs.device)) / scale_.to(outputs.device)

                loss = criterion(outputs[:, cal_grad_targets], batch_y[:, cal_grad_targets])
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
            vali_loss, vali_weight = self.vali(vali_data, vali_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

            # mean loss
            vali_weight = np.ones(len(cal_grad_targets)) / len(cal_grad_targets)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        if self.args.test_train:
            test_data, test_loader = self._get_data(flag='train')
        else:
            test_data, test_loader = self._get_data(flag='test')
        no_grad_targets = test_data.get_no_grad_targets()
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []

        self.model.eval()
        with torch.no_grad():
            for i, (level_x, single_x, level_y, single_y) in enumerate(test_loader):
                batch_x = torch.cat([level_x, single_x.unsqueeze(-1).repeat(1, 1, 1, 60)], dim=-2) # B, G, C, L
                level_y = level_y.reshape(-1, 6, 60).reshape(-1, 360)
                single_y = single_y.reshape(-1, 8)
                batch_y = torch.cat([level_y, single_y], dim=1)
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                
                outputs = self.model(batch_x)

                outputs = outputs.reshape(-1, 6 + 8, 60)
                outputs_level = outputs[:, :6, :].reshape(-1, 6 * 60)
                outputs_single = outputs[:, 6:, :].mean(dim=2)
                outputs = torch.cat([outputs_level, outputs_single], dim=1)

                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                if self.args.inverse:
                    outputs = test_data.inverse_transform(outputs)
                    outputs[:, -8:] = np.where(outputs[:, -8:] < 0, 0, outputs[:, -8:])
                    outputs = test_data.multiply_weight(outputs)
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
        r2[no_grad_targets] = 1.0
        print('r2:{}'.format(r2.mean()))
        print('adjust r2:{}'.format(np.where(r2 < 0, 0, r2).mean()))
        f = open("result_seq2seq.txt", 'a')
        f.write(setting + "  \n")
        f.write('r2:{}'.format(np.where(r2 < 0, 0, r2).mean()))
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
        df_test = pl.read_parquet("/data/home/scv7343/run/climsim_new/dataset/ClimSim/test.parquet")
        if self.args.add_feature_engineering:
            df_fe = pl.read_parquet("/data/home/scv7343/run/climsim_new/dataset/ClimSim/feature_engineering_test.parquet")
            df_test = pl.concat([df_test, df_fe.drop("sample_id")], how="horizontal")
        if self.args.add_fe_v1:
            df_fev1 = pl.read_parquet("/data/home/scv7343/run/climsim_new/dataset/ClimSim/feature_engineering_version1_test.parquet")
            df_test = pl.concat([df_test, df_fev1.drop("sample_id")], how="horizontal")
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        feature_scaler: StandardScaler = load("/data/home/scv7343/run/climsim_new/dataset/ClimSim/feature_scaler.joblib")
        target_scaler: StandardScaler = load("/data/home/scv7343/run/climsim_new/dataset/ClimSim/target_scaler.joblib")
        feature_name = feature_scaler.feature_names_in_.tolist()
        if self.args.add_fe_v1:
            fe_v1_feature_scaler : StandardScaler = load("/data/home/scv7343/run/climsim_new/dataset/ClimSim/feature_engineering_version1_feature_scaler.joblib")
            fe_v1_feature_name = fe_v1_feature_scaler.feature_names_in_.tolist()
        if self.args.add_feature_engineering:
            fe_feature_scaler: StandardScaler = load("/data/home/scv7343/run/climsim_new/dataset/ClimSim/feature_engineering_feature_scaler.joblib")
            fe_feature_name = fe_feature_scaler.feature_names_in_.tolist()
            feature_name = feature_name[:9 * 60] + fe_feature_name[:6 * 60] + fe_feature_name[6 * 60:] + feature_name[9 * 60:]
            if self.args.add_fe_v1:
                feature_name = feature_name[:15 * 60] + fe_v1_feature_name + feature_name[15 * 60:]
        else:
            if self.args.add_fe_v1:
                feature_name = feature_name[:9 * 60] + fe_v1_feature_name + feature_name[9 * 60:]

        target_name = target_scaler.feature_names_in_.tolist()

        df_weight = pl.read_parquet("/data/home/scv7343/run/climsim_new/dataset/ClimSim/sample_submission.parquet")
        weight = df_weight[target_name].to_numpy()

        if self.args.postprocess:
            r2 = np.load('./results/' + setting + '/' + 'metrics.npy')
            print('r2:{}'.format(r2.mean()))
            print('adjust r2:{}'.format(np.where(r2 < 0, 0, r2).mean()))
            unpredict_target_index = np.nonzero(r2 < 0)[0]
            print("unpredict target: ", [target_name[x] for x in unpredict_target_index])

        preds = []
        self.model.eval()
        with torch.no_grad():
            for chunk in df_test.iter_slices(self.args.batch_size):
                data_x = chunk[feature_name].to_numpy()
                batch_x = torch.tensor(data_x).float().to(self.device)

                batch_x_0 = batch_x[:, :23 * 60].reshape(-1, 23, 60).transpose(1, 2) # B, L, C
                batch_x_1 = batch_x[:, 23 * 60:].unsqueeze(1).repeat(1, 60, 1)
                batch_x = torch.concat([batch_x_0, batch_x_1], dim=2)

                outputs = self.model(batch_x)

                outputs_0 = outputs[:, :, :6].transpose(1, 2).reshape(-1, 6 * 60)
                outputs_1 = outputs[:, :, 6:].mean(dim=1)
                outputs = torch.concat([outputs_0, outputs_1], dim=1)
                
                pred = outputs.detach().cpu().numpy()

                if self.args.postprocess:
                    pred[:, unpredict_target_index] = 0

                if self.args.inverse:
                    pred = target_scaler.inverse_transform(pred.astype(np.float64))
                    pred[:, -8:] = np.where(pred[:, -8:] < 0, 0, pred[:, -8:])
                    pred = pred * weight
                    pred = pl.DataFrame(pred, schema=target_name, orient="row")

                preds.append(pl.concat([chunk[["sample_id"]], pred], how="horizontal"))

        preds = pl.concat(preds, how="vertical")

        folder_path = './output/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        preds[df_weight.columns].write_parquet(folder_path + "submission.parquet")