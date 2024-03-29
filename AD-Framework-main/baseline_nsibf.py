import sqlite3
import traceback
from pathlib import Path
import matplotlib as mpl
from sklearn import preprocessing
from torch.autograd import Variable
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['xtick.labelsize'] = 15
plt.rcParams['ytick.labelsize'] = 15
plt.rcParams["font.size"] = 16
import numpy as np
from sklearn.metrics import cohen_kappa_score, recall_score, \
    precision_score, confusion_matrix, fbeta_score, roc_curve, auc, precision_recall_curve
from utils.config import NSIBFConfig
from utils.device import get_free_device
from utils.mail import send_email_notification
from utils.outputs import NSIBFOutput
from utils.utils import str2bool
import os
import torch
import torch.nn as nn
import argparse
from utils.logger import create_logger
from utils.metrics import calculate_average_metric, MetricsResult, zscore, \
    create_label_based_on_zscore, create_label_based_on_quantile
import numpy as np
from utils.data_provider import read_GD_dataset, read_HSS_dataset, read_S5_dataset, read_NAB_dataset, read_2D_dataset, \
    read_ECG_dataset, get_loader, generate_synthetic_dataset, read_SMD_dataset, rolling_window_2D, \
    cutting_window_2D, unroll_window_3D, read_SMAP_dataset, read_MSL_dataset
from statistics import mean
from torch.optim import Adam, lr_scheduler

from utils.data_provider import dataset2path, read_dataset
from utils.metrics import SD_autothreshold, MAD_autothreshold, IQR_autothreshold, get_labels_by_threshold
from utils.utils import make_result_dataframe
from sklearn.metrics import f1_score
import time


class NSIBF(nn.Module):
    def __init__(self, file_name, config):
        super(NSIBF, self).__init__()
        # file info
        self.dataset = config.dataset
        self.file_name = file_name

        # dim info
        self.x_dim = config.x_dim
        self.h_dim = config.h_dim

        # sequence info
        self.preprocessing = config.preprocessing
        self.use_overlapping = config.use_overlapping
        self.use_last_point = config.use_last_point
        self.rolling_size = config.rolling_size

        # optimization info
        self.epochs = config.epochs
        self.milestone_epochs = config.milestone_epochs
        self.lr = config.lr
        self.gamma = config.gamma
        self.batch_size = config.batch_size
        self.weight_decay = config.weight_decay
        self.early_stopping = config.early_stopping
        self.loss_function = config.loss_function
        self.display_epoch = config.display_epoch
        self.use_clip_norm = config.use_clip_norm
        self.gradient_clip_norm = config.gradient_clip_norm

        # dropout
        self.dropout = config.dropout
        self.continue_training = config.continue_training

        self.robustness = config.robustness

        # NSIBF
        self.hnet_hidden_layers = config.hnet_hidden_layers
        self.fnet_hidden_layers = config.fnet_hidden_layers
        self.fnet_hidden_dim = config.fnet_hidden_dim
        self.uencoding_layers = config.uencoding_layers
        self.uencoding_dim = config.uencoding_dim
        self.z_activation = config.z_activation

        # layers
        self.rnn_layers = config.rnn_layers
        self.use_bidirection = config.use_bidirection
        self.force_teaching = config.force_teaching
        self.force_teaching_threshold = config.force_teaching_threshold

        # pid
        self.pid = config.pid

        self.save_model = config.save_model
        if self.save_model:
            if not os.path.exists('./save_model/{}/'.format(self.dataset)):
                os.makedirs('./save_model/{}/'.format(self.dataset))
            self.save_model_path = \
                './save_model/{}/NSIBF_hdim_{}_rollingsize_{}' \
                '_{}_pid={}.pt'.format(self.dataset, config.h_dim, config.rolling_size, Path(self.file_name).stem, self.pid)
        else:
            self.save_model_path = None

        self.load_model = config.load_model
        if self.load_model:
            self.load_model_path = \
                './save_model/{}/NSIBF_hdim_{}_rollingsize_{}' \
                '_{}_pid={}.pt'.format(self.dataset, config.h_dim, config.rolling_size, Path(self.file_name).stem, self.pid)
        else:
            self.load_model_path = None

        # units
        # make networks
        # g_net
        self.g_dense = nn.Sequential()
        self.g_dense.append(nn.Linear(self.x_dim, self.h_dim))
        self.g_dense.append(nn.ReLU())
        for i in range(self.hnet_hidden_layers):
            self.g_dense.append(nn.Linear(self.h_dim, self.h_dim))
            self.g_dense.append(nn.ReLU())
        self.g_out = nn.Sequential()
        self.g_out.append(nn.Linear(self.h_dim, self.h_dim))
        if self.z_activation.lower() == 'tanh':
            self.g_out.append(nn.Tanh())
        else:
            self.g_out.append(nn.ReLU())

        # h_net
        self.h_dense = nn.Sequential()
        self.h_dense.append(nn.Linear(self.h_dim, self.h_dim))
        self.h_dense.append(nn.ReLU())
        for i in range(self.hnet_hidden_layers):
            self.h_dense.append(nn.Linear(self.h_dim, self.h_dim))
            self.h_dense.append(nn.ReLU())
        self.h_out = nn.Sequential()
        self.h_out.append(nn.Linear(self.h_dim, self.x_dim))

        # f_net
        self.f_uencoding = nn.LSTM(self.x_dim, self.h_dim, batch_first=True)
        self.f_out = nn.Sequential()
        self.f_out.append(nn.Linear(self.h_dim + self.h_dim, self.h_dim))
        self.f_out.append(nn.ReLU())
        self.f_out.append(nn.Linear(self.h_dim, self.h_dim))
        if self.z_activation.lower() == "tanh":
            self.f_out.append(nn.Tanh())
        else:
            self.f_out.append(nn.ReLU())

    def reset_parameters(self):
        pass

    def forward(self, input):
        # input: batch_size, seq_len, x_dim

        # get the lastest step
        input_t = input[:, -1, :]

        g_dense = self.g_dense(input_t)
        g_out = self.g_out(g_dense)

        h_dense = self.h_dense(g_out)
        h_out = self.h_out(h_dense)

        _, (f_dense, _) = self.f_uencoding(input)
        f_dense = f_dense.transpose(0, 1).contiguous().squeeze(1)
        f_cat = torch.cat([f_dense, g_out], dim=-1)
        f_out = self.f_out(f_cat)

        fh_dense = self.h_dense(f_out)
        fh_out = self.h_out(fh_dense)

        gz_t_1 = g_out
        gx_t_1 = h_out
        fz_t = f_out
        fx_t = fh_out

        return gz_t_1, gx_t_1, fz_t, fx_t

    def fit(self, train_input, train_label, valid_input, valid_label, test_input, test_label, abnormal_data, abnormal_label, original_x_dim):
        # TN = []
        # TP = []
        # FN = []
        # FP = []
        # PRECISION = []
        # RECALL = []
        # FBETA = []
        # PR_AUC = []
        # ROC_AUC = []
        # CKS = []
        loss_fn = nn.MSELoss()
        opt = Adam(list(self.parameters()), lr=self.lr, weight_decay=self.weight_decay)
        sched = lr_scheduler.StepLR(optimizer=opt, step_size=self.milestone_epochs, gamma=self.gamma)
        # get batch data
        train_data = get_loader(input=train_input, label=train_label, batch_size=self.batch_size, from_numpy=True,
                                drop_last=False, shuffle=False)
        valid_data = get_loader(input=valid_input, label=valid_label, batch_size=self.batch_size, from_numpy=True,
                                drop_last=False, shuffle=False)
        test_data = get_loader(input=test_input, label=test_label, batch_size=self.batch_size, from_numpy=True,
                               drop_last=False, shuffle=False)
        min_valid_loss, all_patience, cur_patience, best_epoch = 1e20, 10, 1, 0
        training_time, testing_time = 0, 0
        if self.load_model == True and self.continue_training == False:
            epoch_valid_losses = [-1]
            start_time = time.time()
            self.load_state_dict(torch.load(self.load_model_path))
        elif self.load_model == True and self.continue_training == True:
            self.load_state_dict(torch.load(self.load_model_path))
            # train model
            start_time = time.time()
            epoch_losses = []
            epoch_valid_losses = []
            loss_weights = [0.45,0.45,0.1]
            for epoch in range(self.epochs):
                train_losses = []
                # opt.zero_grad()
                self.train()
                for i, (batch_x, _) in enumerate(train_data):
                    opt.zero_grad()
                    batch_x = batch_x.to(device)
                    batch_x, batch_y = batch_x[:,:-1,:], batch_x[:,-1,:]
                    gz_t_1, gx_t_1, fz_t, fx_t = self.forward(batch_x)
                    batch_loss = loss_weights[0] * loss_fn(gx_t_1, batch_x[:, -1, :]) \
                        + loss_weights[1] * loss_fn(fx_t, batch_y) \
                        + loss_weights[2] * loss_fn(gz_t_1, fz_t)
                    batch_loss.backward()
                    if self.use_clip_norm:
                        torch.nn.utils.clip_grad_norm_(list(self.parameters()), self.gradient_clip_norm)
                    opt.step()
                    sched.step()
                    train_losses.append(batch_loss.item())
                epoch_losses.append(mean(train_losses))
                if epoch % self.display_epoch == 0:
                    train_logger.info('epoch = {} , train loss = {}'.format(epoch, epoch_losses[-1]))

                valid_losses = []
                # opt.zero_grad()
                self.eval()
                with torch.no_grad():
                    for i, (val_batch_x, _) in enumerate(valid_data):
                        val_batch_x = val_batch_x.to(device)
                        val_batch_x, val_batch_y = batch_x[:, :-1, :], batch_x[:, -1, :]
                        val_gz_t_1, val_gx_t_1, val_fz_t, val_fx_t = self.forward(val_batch_x)
                        val_batch_loss = loss_weights[0] * loss_fn(val_gx_t_1, val_batch_x[:, -1, :]) \
                            + loss_weights[1] * loss_fn(val_fx_t, val_batch_y) \
                            + loss_weights[2] * loss_fn(val_gz_t_1, val_fz_t)
                        valid_losses.append(val_batch_loss.item())
                epoch_valid_losses.append(mean(valid_losses))
                if epoch % self.display_epoch == 0:
                    train_logger.info('epoch = {} , valid loss = {}'.format(epoch, epoch_valid_losses[-1]))

                if self.early_stopping:
                    if len(epoch_valid_losses) > 1:
                        if epoch_valid_losses[best_epoch] - epoch_valid_losses[-1] < 3e-4:
                            train_logger.info('EarlyStopping counter: {} out of {}'.format(cur_patience, all_patience))
                            if cur_patience == all_patience:
                                train_logger.info('Early Stopping!')
                                break
                            cur_patience += 1
                        else:
                            train_logger.info("Saving Model.")
                            torch.save(self.state_dict(), self.save_model_path)
                            best_epoch = epoch
                            cur_patience = 1
                    else:
                        torch.save(self.state_dict(), self.save_model_path)
        else:
            # train model
            epoch_losses = []
            epoch_valid_losses = []
            start_time = time.time()
            loss_weights = [0.45,0.45,0.1]
            for epoch in range(self.epochs):
                train_losses = []
                # opt.zero_grad()
                self.train()
                for i, (batch_x, batch_y) in enumerate(train_data):
                    opt.zero_grad()
                    batch_x = batch_x.to(device)
                    batch_x, batch_y = batch_x[:,:-1,:], batch_x[:,-1,:]
                    gz_t_1, gx_t_1, fz_t, fx_t = self.forward(batch_x)
                    batch_loss = loss_weights[0] * loss_fn(gx_t_1, batch_x[:, -1, :]) \
                        + loss_weights[1] * loss_fn(fx_t, batch_y) \
                        + loss_weights[2] * loss_fn(gz_t_1, fz_t)
                    batch_loss.backward()
                    if self.use_clip_norm:
                        torch.nn.utils.clip_grad_norm_(list(self.parameters()), self.gradient_clip_norm)
                    opt.step()
                    sched.step()
                    train_losses.append(batch_loss.item())
                epoch_losses.append(mean(train_losses))
                if epoch % self.display_epoch == 0:
                    train_logger.info('epoch = {} , train loss = {}'.format(epoch, epoch_losses[-1]))

                valid_losses = []
                self.eval()
                with torch.no_grad():
                    for i, (val_batch_x, _) in enumerate(valid_data):
                        val_batch_x = val_batch_x.to(device)
                        val_batch_x, val_batch_y = batch_x[:, :-1, :], batch_x[:, -1, :]
                        val_gz_t_1, val_gx_t_1, val_fz_t, val_fx_t = self.forward(val_batch_x)
                        val_batch_loss = loss_weights[0] * loss_fn(val_gx_t_1, val_batch_x[:, -1, :]) \
                            + loss_weights[1] * loss_fn(val_fx_t, val_batch_y) \
                            + loss_weights[2] * loss_fn(val_gz_t_1, val_fz_t)
                        valid_losses.append(val_batch_loss.item())
                epoch_valid_losses.append(mean(valid_losses))
                if epoch % self.display_epoch == 0:
                    train_logger.info('epoch = {} , valid loss = {}'.format(epoch, epoch_valid_losses[-1]))

                if self.early_stopping:
                    if len(epoch_valid_losses) > 1:
                        if epoch_valid_losses[best_epoch] - epoch_valid_losses[-1] < 3e-4:
                            train_logger.info('EarlyStopping counter: {} out of {}'.format(cur_patience, all_patience))
                            if cur_patience == all_patience:
                                train_logger.info('Early Stopping!')
                                break
                            cur_patience += 1
                        else:
                            train_logger.info("Saving Model.")
                            torch.save(self.state_dict(), self.save_model_path)
                            best_epoch = epoch
                            cur_patience = 1
                    else:
                        torch.save(self.state_dict(), self.save_model_path)
        training_time = time.time() - start_time
        min_valid_loss = min(epoch_valid_losses)
        self.load_state_dict(torch.load(self.save_model_path))
        # load the trained model
        self.eval()
        start_time = time.time()
        with torch.no_grad():
            cat_xs_rec = []
            cat_xs_pred = []
            for i, (batch_x, _) in enumerate(test_data):
                batch_x = batch_x.to(device)
                batch_x, batch_y = batch_x[:,:-1,:], batch_x[:,-1,0]
                gz_t_1, gx_t_1, fz_t, fx_t = self.forward(batch_x)
                cat_xs_rec.append(gx_t_1)
                cat_xs_pred.append(fx_t)

            cat_xs_rec = torch.cat(cat_xs_rec)
            cat_xs_pred = torch.cat(cat_xs_pred)
            testing_time = time.time() - start_time
            # rae_output = RAEOutput(dec_means=cat_xs, best_TN=max(TN), best_FP=max(FP), best_FN=max(FN), best_TP=max(TP),
            #                        best_precision=max(PRECISION), best_recall=max(RECALL), best_fbeta=max(FBETA),
            #                        best_pr_auc=max(PR_AUC), best_roc_auc=max(ROC_AUC), best_cks=max(CKS))
            nsibf_output = NSIBFOutput(dec_means_rec=cat_xs_rec, dec_means_pred=cat_xs_pred, best_TN=None, best_FP=None, best_FN=None, best_TP=None,
                                   best_precision=None, best_recall=None, best_fbeta=None,
                                   best_pr_auc=None, best_roc_auc=None, best_cks=None, min_valid_loss=min_valid_loss,
                                   training_time=training_time, testing_time=testing_time)
            return nsibf_output


def RunModel(train_filename, test_filename, label_filename, config, ratio):
    negative_sample = True if "noise" in config.dataset else False
    train_data, abnormal_data, abnormal_label = read_dataset(train_filename, test_filename, label_filename,
                                                             normalize=True, file_logger=file_logger, negative_sample=negative_sample, ratio=ratio)
    original_x_dim = abnormal_data.shape[1]

    rolling_train_data = None
    rolling_valid_data = None
    if config.preprocessing:
        if config.use_overlapping:
            if train_data is not None:
                rolling_train_data, rolling_abnormal_data, rolling_abnormal_label = rolling_window_2D(train_data, config.rolling_size), rolling_window_2D(abnormal_data, config.rolling_size), rolling_window_2D(abnormal_label, config.rolling_size)
                train_split_idx = int(rolling_train_data.shape[0] * 0.7)
                rolling_train_data, rolling_valid_data = rolling_train_data[:train_split_idx], rolling_train_data[train_split_idx:]
            else:
                rolling_abnormal_data, rolling_abnormal_label = rolling_window_2D(abnormal_data, config.rolling_size), rolling_window_2D(abnormal_label, config.rolling_size)
        else:
            if train_data is not None:
                rolling_train_data, rolling_abnormal_data, rolling_abnormal_label = cutting_window_2D(train_data, config.rolling_size), cutting_window_2D(abnormal_data, config.rolling_size), cutting_window_2D(abnormal_label, config.rolling_size)
                train_split_idx = int(rolling_train_data.shape[0] * 0.7)
                rolling_train_data, rolling_valid_data = rolling_train_data[:train_split_idx], rolling_train_data[train_split_idx:]
            else:
                rolling_abnormal_data, rolling_abnormal_label = cutting_window_2D(abnormal_data, config.rolling_size), cutting_window_2D(abnormal_label, config.rolling_size)
    else:
        if train_data is not None:
            rolling_train_data, rolling_abnormal_data, rolling_abnormal_label = np.expand_dims(train_data, axis=0), np.expand_dims(abnormal_data, axis=0), np.expand_dims(abnormal_label, axis=0)
            train_split_idx = int(rolling_train_data.shape[0] * 0.7)
            rolling_train_data, rolling_valid_data = rolling_train_data[:train_split_idx], rolling_train_data[train_split_idx:]
        else:
            rolling_abnormal_data, rolling_abnormal_label = np.expand_dims(abnormal_data, axis=0), np.expand_dims(abnormal_label, axis=0)

    config.x_dim = rolling_abnormal_data.shape[2]

    model = NSIBF(file_name=train_filename, config=config)
    model = model.to(device)
    nsibf_output = None
    if train_data is not None and config.robustness == False:
        nsibf_output = model.fit(train_input=rolling_train_data, train_label=rolling_train_data,
                               valid_input=rolling_valid_data, valid_label=rolling_valid_data,
                               test_input=rolling_abnormal_data, test_label=rolling_abnormal_label,
                               abnormal_data=abnormal_data, abnormal_label=abnormal_label, original_x_dim=original_x_dim)
    elif train_data is None or config.robustness == True:
        nsibf_output = model.fit(train_input=rolling_abnormal_data, train_label=rolling_abnormal_data,
                               valid_input=rolling_valid_data, valid_label=rolling_valid_data,
                               test_input=rolling_abnormal_data, test_label=rolling_abnormal_label,
                               abnormal_data=abnormal_data, abnormal_label=abnormal_label, original_x_dim=original_x_dim)
    # %%
    min_max_scaler = preprocessing.MinMaxScaler()
    if config.preprocessing:
        if config.use_overlapping:
            if config.use_last_point:
                dec_mean_unroll_rec = nsibf_output.dec_means_rec.detach().cpu().numpy()
                dec_mean_unroll_pred = nsibf_output.dec_means_pred.detach().cpu().numpy()
                dec_mean_unroll_rec = min_max_scaler.fit_transform(dec_mean_unroll_rec)
                dec_mean_unroll_pred = min_max_scaler.fit_transform(dec_mean_unroll_pred)
                x_original_unroll_rec = abnormal_data[config.rolling_size-2: -1]
                x_original_unroll_pred = abnormal_data[config.rolling_size - 1:]
            else:
                # the unroll_window_3D will recover the shape as abnormal_data
                # and we only use the [config.rolling_size-1:] to calculate the error, in which we ignore
                # the first config.rolling_size time steps.
                raise ValueError('NSIBF should set the use_last_point==True')
                # dec_mean_unroll_rec = unroll_window_3D(nsibf_output.dec_means_rec.detach().cpu().numpy())[::-1]
                # dec_mean_unroll_pred = unroll_window_3D(nsibf_output.dec_means_pred.detach().cpu().numpy())[::-1]
                # dec_mean_unroll_rec = min_max_scaler.fit_transform(dec_mean_unroll_rec)
                # dec_mean_unroll_pred = min_max_scaler.fit_transform(dec_mean_unroll_rec)
                # x_original_unroll = abnormal_data[: dec_mean_unroll_rec.shape[0]]

        else:
            # raise ValueError
            raise ValueError('NSIBF should set the use_last_point==True')
            dec_mean_unroll = np.reshape(rae_output.dec_means.detach().cpu().numpy(), (-1, original_x_dim))
            dec_mean_unroll = min_max_scaler.fit_transform(dec_mean_unroll)
            x_original_unroll = abnormal_data[: dec_mean_unroll.shape[0]]
    else:
        # raise ValueError
        raise ValueError('NSIBF should set the use_last_point==True')
        dec_mean_unroll = rae_output.dec_means.detach().cpu().numpy()
        dec_mean_unroll = np.squeeze(dec_mean_unroll, axis=0)
        dec_mean_unroll = min_max_scaler.fit_transform(dec_mean_unroll)
        x_original_unroll = abnormal_data

    if config.save_output:
        if not os.path.exists('./outputs/NPY/{}/'.format(config.dataset)):
            os.makedirs('./outputs/NPY/{}/'.format(config.dataset))
        np.save('./outputs/NPY/{}/Dec_NSIBF_hdim_{}_rollingsize_{}_{}_pid={}.npy'.format(config.dataset, config.h_dim, config.rolling_size, train_filename.stem, config.pid), [dec_mean_unroll_pred, dec_mean_unroll_rec])

    error_rec = np.sum(x_original_unroll_rec - np.reshape(dec_mean_unroll_rec, [-1, original_x_dim]), axis=1) ** 2
    error_pred = np.sum(x_original_unroll_pred - np.reshape(dec_mean_unroll_pred, [-1, original_x_dim]), axis=1) ** 2
    # TODO: the original NSIBF use mahalanobis distance to calculate the anomaly score
    # error = error_rec + error_pred
    error = error_pred
    # final_zscore = zscore(error)
    # np_decision = create_label_based_on_zscore(final_zscore, 2.5, True)
    #np_decision = create_label_based_on_quantile(error, quantile=99)
    SD_Tmin, SD_Tmax = SD_autothreshold(error)
    SD_y_hat = get_labels_by_threshold(error, Tmax=SD_Tmax, use_max=True, use_min=False)
    MAD_Tmin, MAD_Tmax = MAD_autothreshold(error)
    MAD_y_hat = get_labels_by_threshold(error, Tmax=MAD_Tmax, use_max=True, use_min=False)
    IQR_Tmin, IQR_Tmax = IQR_autothreshold(error)
    IQR_y_hat = get_labels_by_threshold(error, Tmax=IQR_Tmax, use_max=True, use_min=False)
    np_decision = {}
    np_decision["SD"] = SD_y_hat
    np_decision["MAD"] = MAD_y_hat
    np_decision["IQR"] = IQR_y_hat

    # TODO metrics computation.

    # %%
    if config.save_figure:
        if original_x_dim == 1:
            plt.figure(figsize=(9, 3))
            plt.plot(x_original_unroll_pred, color='blue', lw=1.5)
            plt.title('Original Data')
            plt.grid(True)
            plt.tight_layout()
            # plt.show()
            plt.savefig('./figures/{}/Ori_NSIBF_hdim_{}_rollingsize_{}_{}_pid={}.png'.format(config.dataset, config.h_dim, config.rolling_size, train_filename.stem, config.pid), dpi=600)
            plt.close()

            # Plot decoder output
            plt.figure(figsize=(9, 3))
            plt.plot(dec_mean_unroll_pred, color='blue', lw=1.5)
            plt.title('Decoding Output')
            plt.grid(True)
            plt.tight_layout()
            # plt.show()
            plt.savefig('./figures/{}/Dec_NSIBF_hdim_{}_rolling_size_{}_{}_pid={}.png'.format(config.dataset, config.h_dim, config.rolling_size, train_filename.stem, config.pid), dpi=600)
            plt.close()

            t = np.arange(0, abnormal_data.shape[0])
            # markercolors = ['blue' if i == 1 else 'red' for i in abnormal_label[: dec_mean_unroll.shape[0]]]
            # markersize = [4 if i == 1 else 25 for i in abnormal_label[: dec_mean_unroll.shape[0]]]
            # plt.figure(figsize=(9, 3))
            # ax = plt.axes()
            # plt.yticks([0, 0.25, 0.5, 0.75, 1])
            # ax.set_xlim(t[0] - 10, t[-1] + 10)
            # ax.set_ylim(-0.10, 1.10)
            # plt.xlabel('$t$')
            # plt.ylabel('$s$')
            # plt.grid(True)
            # plt.tight_layout()
            # plt.margins(0.1)
            # plt.plot(np.squeeze(abnormal_data[: dec_mean_unroll.shape[0]]), alpha=0.7)
            # plt.scatter(t[: dec_mean_unroll.shape[0]], x_original_unroll[: dec_mean_unroll.shape[0]], s=markersize, c=markercolors)
            # # plt.show()
            # plt.savefig('./figures/{}/VisInp_RAE_{}_pid={}.png'.format(config.dataset, Path(file_name).stem, config.pid), dpi=600)
            # plt.close()

            markercolors = ['blue' for i in range(config.rolling_size - 1)] + ['blue' if i == 1 else 'red' for i in np_decision["SD"]]
            markersize = [4 for i in range(config.rolling_size - 1)] + [4 if i == 1 else 25 for i in np_decision["SD"]]
            plt.figure(figsize=(9, 3))
            ax = plt.axes()
            plt.yticks([0, 0.25, 0.5, 0.75, 1])
            ax.set_xlim(t[0] - 10, t[-1] + 10)
            ax.set_ylim(-0.10, 1.10)
            plt.xlabel('$t$')
            plt.ylabel('$s$')
            plt.grid(True)
            plt.tight_layout()
            plt.margins(0.1)
            plt.plot(np.squeeze(abnormal_data), alpha=0.7)
            plt.scatter(t, abnormal_data, s=markersize, c=markercolors)
            # plt.show()
            plt.savefig('./figures/{}/VisOut_NSIBF_hdim_{}_rollingsize_{}_SD_{}_pid={}.png'.format(config.dataset, config.h_dim, config.rolling_size, train_filename.stem, config.pid), dpi=300)
            plt.close()

            markercolors = ['blue' for i in range(config.rolling_size - 1)] + ['blue' if i == 1 else 'red' for i in np_decision["MAD"]]
            markersize = [4 for i in range(config.rolling_size - 1)] + [4 if i == 1 else 25 for i in np_decision["MAD"]]
            plt.figure(figsize=(9, 3))
            ax = plt.axes()
            plt.yticks([0, 0.25, 0.5, 0.75, 1])
            ax.set_xlim(t[0] - 10, t[-1] + 10)
            ax.set_ylim(-0.10, 1.10)
            plt.xlabel('$t$')
            plt.ylabel('$s$')
            plt.grid(True)
            plt.tight_layout()
            plt.margins(0.1)
            plt.plot(np.squeeze(abnormal_data), alpha=0.7)
            plt.scatter(t, abnormal_data, s=markersize, c=markercolors)
            # plt.show()
            plt.savefig('./figures/{}/VisOut_NSIBF_hdim_{}_rollingsize_{}_MAD_{}_pid={}.png'.format(config.dataset, config.h_dim, config.rolling_size, train_filename.stem, config.pid), dpi=300)
            plt.close()

            markercolors = ['blue' for i in range(config.rolling_size - 1)] + ['blue' if i == 1 else 'red' for i in np_decision["IQR"]]
            markersize = [4 for i in range(config.rolling_size - 1)] + [4 if i == 1 else 25 for i in np_decision["IQR"]]
            plt.figure(figsize=(9, 3))
            ax = plt.axes()
            plt.yticks([0, 0.25, 0.5, 0.75, 1])
            ax.set_xlim(t[0] - 10, t[-1] + 10)
            ax.set_ylim(-0.10, 1.10)
            plt.xlabel('$t$')
            plt.ylabel('$s$')
            plt.grid(True)
            plt.tight_layout()
            plt.margins(0.1)
            plt.plot(np.squeeze(abnormal_data), alpha=0.7)
            plt.scatter(t, abnormal_data, s=markersize, c=markercolors)
            # plt.show()
            plt.savefig('./figures/{}/VisOut_NSIBF_hdim_{}_rollingsize_{}_IQR_{}_pid={}.png'.format(config.dataset, config.h_dim, config.rolling_size, train_filename.stem, config.pid), dpi=300)
            plt.close()
        else:
            file_logger.info('cannot plot image with x_dim > 1')

    if config.use_spot:
        pass
    else:
        pos_label = -1
        TN, FP, FN, TP, precision, recall, f1 = {}, {}, {}, {}, {}, {}, {}
        for threshold_method in np_decision:
            cm = confusion_matrix(y_true=abnormal_label[config.rolling_size-1:], y_pred=np_decision[threshold_method],
                                  labels=[1, -1])
            TN[threshold_method] = cm[0][0]
            FP[threshold_method] = cm[0][1]
            FN[threshold_method] = cm[1][0]
            TP[threshold_method] = cm[1][1]
            precision[threshold_method] = precision_score(y_true=abnormal_label[config.rolling_size-1:],
                                                          y_pred=np_decision[threshold_method], pos_label=pos_label)
            recall[threshold_method] = recall_score(y_true=abnormal_label[config.rolling_size-1:],
                                                    y_pred=np_decision[threshold_method], pos_label=pos_label)
            f1[threshold_method] = f1_score(y_true=abnormal_label[config.rolling_size-1:],
                                            y_pred=np_decision[threshold_method], pos_label=pos_label)

        fpr, tpr, _ = roc_curve(y_true=abnormal_label[config.rolling_size-1:], y_score=np.nan_to_num(error),
                                pos_label=pos_label)
        roc_auc = auc(fpr, tpr)
        pre, re, _ = precision_recall_curve(y_true=abnormal_label[config.rolling_size-1:],
                                            probas_pred=np.nan_to_num(error),
                                            pos_label=pos_label)
        pr_auc = auc(re, pre)
        metrics_result = MetricsResult(TN=TN, FP=FP, FN=FN, TP=TP, precision=precision,
                                       recall=recall, fbeta=f1, pr_auc=pr_auc, roc_auc=roc_auc,
                                       best_TN=nsibf_output.best_TN, best_FP=nsibf_output.best_FP,
                                       best_FN=nsibf_output.best_FN, best_TP=nsibf_output.best_TP,
                                       best_precision=nsibf_output.best_precision, best_recall=nsibf_output.best_recall,
                                       best_fbeta=nsibf_output.best_fbeta, best_pr_auc=nsibf_output.best_pr_auc,
                                       best_roc_auc=nsibf_output.best_roc_auc, best_cks=nsibf_output.best_cks,
                                       min_valid_loss=nsibf_output.min_valid_loss)
        return metrics_result

if __name__ == '__main__':

    # %%
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=int, default=0)
    parser.add_argument('--x_dim', type=int, default=1)
    parser.add_argument('--h_dim', type=int, default=64)
    parser.add_argument('--preprocessing', type=str2bool, default=True)
    parser.add_argument('--use_overlapping', type=str2bool, default=True)
    # rolling_size means window size.
    parser.add_argument('--rolling_size', type=int, default=32)
    #parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--epochs', type=int, default=200)
    # parser.add_argument('--epochs', type=int, default=1)
    # milestone_epochs is used to reduce the learning rate.
    parser.add_argument('--milestone_epochs', type=int, default=50)
    parser.add_argument('--ratio', type=float, default=0.05)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--weight_decay', type=float, default=1e-8)
    parser.add_argument('--early_stopping', type=str2bool, default=True)
    parser.add_argument('--loss_function', type=str, default='mse')
    parser.add_argument('--rnn_layers', type=int, default=1)
    parser.add_argument('--use_clip_norm', type=str2bool, default=True)
    parser.add_argument('--gradient_clip_norm', type=float, default=10)
    parser.add_argument('--use_bidirection', type=str2bool, default=False)
    parser.add_argument('--force_teaching', type=str2bool, default=False)
    parser.add_argument('--force_teaching_threshold', type=float, default=0.75)
    parser.add_argument('--display_epoch', type=int, default=5)
    parser.add_argument('--save_output', type=str2bool, default=True)
    parser.add_argument('--save_figure', type=str2bool, default=False)
    parser.add_argument('--save_model', type=str2bool, default=True)  # save model
    parser.add_argument('--save_results', type=str2bool, default=True)  # save results
    parser.add_argument('--load_model', type=str2bool, default=False)  # load model
    parser.add_argument('--continue_training', type=str2bool, default=False)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--use_spot', type=str2bool, default=False)
    parser.add_argument('--use_last_point', type=str2bool, default=True)
    parser.add_argument('--save_config', type=str2bool, default=True)
    parser.add_argument('--load_config', type=str2bool, default=False)
    parser.add_argument('--server_run', type=str2bool, default=False)
    parser.add_argument('--robustness', type=str2bool, default=False)

    # the following parameters are for NSIBF. The parameters above should be refined, i.e., some of them should be deleted.
    parser.add_argument('--hnet_hidden_layers', type=int, default=1)
    parser.add_argument('--fnet_hidden_layers', type=int, default=1)
    parser.add_argument('--fnet_hidden_dim', type=int, default=8)
    parser.add_argument('--uencoding_layers', type=int, default=1)
    parser.add_argument('--uencoding_dim', type=int, default=8)
    parser.add_argument('--z_activation', type=str, default='tanh')

    parser.add_argument('--pid', type=int, default=0)
    args = parser.parse_args()

    # for registered_dataset in ["MSL", "SMAP", "SMD", "NAB", "AIOps", "Credit", "ECG", "nyc_taxi", "SWAT", "Yahoo"]:
    for registered_dataset in ["NAB", "AIOps", "Credit", "ECG", "nyc_taxi", "SWAT", "Yahoo"]:
    #for registered_dataset in ["NAB"]:

        # the dim in args is useless, which should be deleted in the future version.
        if "noise" in registered_dataset:
            args.dataset = registered_dataset + "_{:.2f}".format(args.ratio)
        else:
            args.dataset = registered_dataset

        if args.load_config:
            config = NSIBFConfig(dataset=None, x_dim=None, h_dim=None, preprocessing=None, use_overlapping=None,
                               rolling_size=None, epochs=None, milestone_epochs=None, lr=None, gamma=None, batch_size=None,
                               weight_decay=None, early_stopping=None, loss_function=None, rnn_layers=None,
                               use_clip_norm=None, gradient_clip_norm=None, use_bidirection=None, force_teaching=None,
                               force_teaching_threshold=None, display_epoch=None, save_output=None, save_figure=None,
                               save_model=None, load_model=None, continue_training=None, dropout=None, use_spot=None,
                               use_last_point=None, save_config=None, load_config=None, server_run=None, robustness=None,
                               pid=None, save_results=None, hnet_hidden_layers=None, fnet_hidden_layers=None,
                               fnet_hidden_dim=None, uencoding_layers=None, uencoding_dim=None, z_activation=None)
            try:
                config.import_config('./config/{}/Config_NSIBF_hdim_{}_rollingsize_{}_pid={}.json'.format(config.dataset, config.h_dim, config.rolling_size, config.pid))
            except:
                print('There is no config.')
        else:
            config = NSIBFConfig(dataset=args.dataset, x_dim=args.x_dim, h_dim=args.h_dim, preprocessing=args.preprocessing,
                               use_overlapping=args.use_overlapping, rolling_size=args.rolling_size, epochs=args.epochs,
                               milestone_epochs=args.milestone_epochs, lr=args.lr, gamma=args.gamma,
                               batch_size=args.batch_size, weight_decay=args.weight_decay,
                               early_stopping=args.early_stopping, loss_function=args.loss_function,
                               use_clip_norm=args.use_clip_norm, gradient_clip_norm=args.gradient_clip_norm,
                               rnn_layers=args.rnn_layers, use_bidirection=args.use_bidirection,
                               force_teaching=args.force_teaching, force_teaching_threshold=args.force_teaching_threshold,
                               display_epoch=args.display_epoch, save_output=args.save_output, save_figure=args.save_figure,
                               save_model=args.save_model, load_model=args.load_model,
                               continue_training=args.continue_training, dropout=args.dropout, use_spot=args.use_spot,
                               use_last_point=args.use_last_point, save_config=args.save_config,
                               load_config=args.load_config, server_run=args.server_run, robustness=args.robustness,
                               pid=args.pid, save_results=args.save_results, hnet_hidden_layers=args.hnet_hidden_layers,
                               fnet_hidden_layers=args.fnet_hidden_layers, fnet_hidden_dim=args.fnet_hidden_dim,
                               uencoding_layers=args.uencoding_layers, uencoding_dim=args.uencoding_dim,
                               z_activation=args.z_activation)
        if args.save_config:
            if not os.path.exists('./config/{}/'.format(config.dataset)):
                os.makedirs('./config/{}/'.format(config.dataset))
            config.export_config('./config/{}/Config_NSIBF_hdim_{}_rollingsize_{}_pid={}.json'.format(config.dataset, config.h_dim, config.rolling_size, config.pid))
        # %%
        if config.dataset not in dataset2path:
            raise ValueError("dataset {} is not registered.".format(config.dataset))
        else:
            train_path = dataset2path[config.dataset]["train"]
            test_path = dataset2path[config.dataset]["test"]
            label_path = dataset2path[config.dataset]["test_label"]
        # %%
        device = torch.device(get_free_device())

        train_logger, file_logger, meta_logger = create_logger(dataset=args.dataset,
                                                               h_dim=config.h_dim,
                                                               rolling_size=config.rolling_size,
                                                               train_logger_name='rae_train_logger',
                                                               file_logger_name='rae_file_logger',
                                                               meta_logger_name='rae_meta_logger',
                                                               model_name='NSIBF',
                                                               pid=args.pid)

        # logging setting
        file_logger.info('============================')
        for key, value in vars(args).items():
            file_logger.info(key + ' = {}'.format(value))
        file_logger.info('============================')

        meta_logger.info('============================')
        for key, value in vars(args).items():
            meta_logger.info(key + ' = {}'.format(value))
        meta_logger.info('============================')

        for train_file in train_path.iterdir():
        #for train_file in [Path('../datasets/train/MSL/M-1.pkl')]:
            test_file = test_path / train_file.name
            label_file = label_path / train_file.name
            file_logger.info('============================')
            file_logger.info(train_file)

            metrics_result = RunModel(train_filename=train_file, test_filename=test_file, label_filename=label_file, config=config, ratio=args.ratio)
            result_dataframe = make_result_dataframe(metrics_result)

            if config.save_results == True:
                if not os.path.exists('./results/{}/'.format(config.dataset)):
                    os.makedirs('./results/{}/'.format(config.dataset))
                result_dataframe.to_csv('./results/{}/Results_NSIBF_hdim_{}_rollingsize_{}_{}_pid={}.csv'.format(config.dataset, config.h_dim, config.rolling_size, train_file.stem, config.pid),
                                        index=False)

            # TODO: output the result to the file_logger
            # TODO: output the average result to the meta_logger

