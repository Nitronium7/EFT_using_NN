print("Initializing...")
from datetime import datetime
import os
from tqdm import tqdm
import pandas as pd
import numpy as np
import random
import time
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from matplotlib.colors import ListedColormap
from matplotlib import ticker
from collections import OrderedDict
from operator import itemgetter
import itertools
from scipy.stats import skew
import language_tool_python
import os.path
from datetime import datetime
import torchvision
from torchvision import transforms, datasets
import Net
import random
import torch.utils.data
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from dataset_from_df import dataset_from_df
import sys
import math
import tensorflow as tf
import shap
import torch
from test import *


def find_nodes(c_in, c_out):
    return 2 / 3 * c_in + c_out


def csv_to_data_frame(path, sample):
    # using function to read from path
    print("Reading data file...")
    df = pd.read_csv(path, float_precision='legacy')
    #df = df.loc[(df['c_pt'] < 250)
                 #| (df['c_pt'] > 350)]
    #df = df.loc[(df["fatjet_pt"] > 250) & (df['fatjet_pt']<450)]
    #df = df.drop(['fatjet_pt'], axis=1)
    print(df)

    df.reset_index(drop=True, inplace=True)

    return df.head(sample)


def excel_to_data_frame(path, sample):
    # using function to read from path
    print("Reading data file...")
    df = pd.read_excel(path)

    return df.head(sample)


def data_processor(df, response_name, tr_ratio, stored_normalization):
    positive_rows = []
    negative_rows = []
    df = df.sample(frac=1).reset_index(drop=True)
    for i in range(len(df)):
        if df.at[i, response_name] == 1:
            positive_rows.append(i)
        else:
            negative_rows.append(i)
    ptrain, ptest = splitter(positive_rows, tr_ratio)
    ntrain, ntest = splitter(negative_rows, tr_ratio)

    width = len(df.columns)
    df_array = df.to_numpy()
    df_array = np.delete(df_array, 0, axis=1)

    df_array = df_array.astype(float)
    df_array = normalizer(df_array, stored_normalization)

    return balancer_randomizer_normalizer(ptrain, ntrain, df_array, width), balancer_randomizer_normalizer(ptest, ntest,
                                                                                                           df_array,
                                                                                                           width)


def test_processor(df):
    df_array = df.to_numpy()
    df_array = np.delete(df_array, 0, axis=1)

    df_array = df_array.astype(float)
    df_array = normalizer(df_array, False)
    return df_array


def balancer_randomizer_normalizer(positive, negative, arr, width):
    balanced = []
    count_p = len(positive)
    count_n = len(negative)
    #sample_count = min(count_p, count_n) #oversampler, undersampler

    sample_count = max(count_p, count_n)
    for j in range(sample_count):
        balanced.append(positive[(j % count_p)])
        balanced.append(negative[(j % count_n)])
    random.shuffle(balanced)
    balanced.reverse()
    random_balanced_array = np.zeros((sample_count * 2, (width - 1)))

    for i in range(len(balanced)):
        random_balanced_array[i, :] = arr[balanced[i], :width]

    return random_balanced_array


def normalizer(array, use_stored):
    # returns column of data
    shape = array.shape

    h = shape[0]
    w = shape[1]
    sum = np.sum(array[:, :w - 1], axis=0)
    avg = sum / h

    std = np.std(array[:, :w - 1], axis=0)
    if use_stored:
        avg = [7.62213879e-03, -1.76085296e-02, 3.63937945e-02, 3.58513211e+02,
               8.58320715e-03, 1.06243023e-02, -1.54544229e-02, 6.25129270e-03]

        std = [1.11162964, 1.8074318, 0.99933753, 120.38897248, 0.97516392,
               1.81312432, 1.80667721, 0.8430421]

    for i in range(w - 1):
        for j in range(h):
            array[j][i] = (array[j][i] - avg[i]) / std[i]
    return array


def normalizer_h(array):
    shape = array.shape
    h = shape[0]
    price_indices = []
    volume_indices = [4, 9, 14, 19, 24, 29, 34, 39, 44, 49]

    for m in range(50):
        if m not in volume_indices:
            price_indices.append(m)
    for i in range(h):
        l = array[i]
        prices = np.take(l, price_indices)
        volumes = np.take(l, volume_indices)
        p_sum = np.sum(prices)
        p_std = np.std(prices)
        v_sum = np.sum(volumes)
        v_std = np.std(volumes)
        avg_p = p_sum / len(prices)
        avg_v = v_sum / len(volumes)
        if p_std == 0 or v_std == 0:
            array[i] = array[i - 1]
        else:
            for k in price_indices:
                array[i][k] = (array[i][k] - avg_p) / p_std
            for k in volume_indices:
                array[i][k] = (array[i][k] - avg_v) / v_std
    return array


def splitter(in_list, tr_ratio):
    # randomly separates data into test and training data sets
    pos = round(tr_ratio * len(in_list))

    l1 = in_list[:pos]
    l2 = in_list[pos:]
    return l1, l2


def sigmoid(x):
    if x < -20:
        return 0
    elif x > 20:
        return 1
    return 1 / (1 + np.exp(-1 * x))


def log_moid(list):
    newlist = []
    log = False
    if log:
        for each in list:
            if each >= 0:
                each = sigmoid(math.log(each + 0.00001, 5))
                newlist.append(each)
            else:
                each = sigmoid(-1 * math.log(-1 * each + 0.00001, 5))
                newlist.append(each)
        return newlist
    for each in list:
        each = sigmoid(each)
        newlist.append(each)
    return newlist

    # return same_list


# transform_f = lambda x: model_list
def plot_hist(hist_list, c_bins):
    weights_0 = np.ones_like(hist_list) / float(len(hist_list))
    plt.hist(hist_list, c_bins, weights=weights_0)
    plt.xlabel('Accuracy')
    plt.ylabel('Frequency')
    plt.title("Accuracy Distribution")
    plt.grid(True)
    time_stamp = datetime.strftime(datetime.now(), "%m_%d_%Y_%H_%M_%S")
    file_name = "output_figures/accuracy_distribution" + time_stamp + ".png"
    plt.savefig(file_name)
    plt.show()


def plot_double_hist(hist_list_0, hist_list_1, c_bins, alpha):
    weights_0 = np.ones_like(hist_list_0) / (0.01 * float(len(hist_list_0)))
    weights_1 = np.ones_like(hist_list_1) / (0.01 * float(len(hist_list_1)))

    hist_list_0 = log_moid(hist_list_0)
    hist_list_1 = log_moid(hist_list_1)
    plt.hist(hist_list_0, c_bins, alpha=alpha, weights=weights_0, range=(0, 1))
    plt.hist(hist_list_1, c_bins, alpha=alpha, weights=weights_1, range=(0, 1))

    plt.xlabel('NN score')
    plt.ylabel('Percent')
    title = "NN Score Distribution "
    plt.title(title)
    plt.legend(labels=["Diboson", "EFT"])
    plt.grid(True)
    time_stamp = datetime.strftime(datetime.now(), "%m_%d_%Y_%H_%M_%S")
    file_name = "output_figures/distribution_figure_" + time_stamp + ".png"
    plt.savefig(file_name)
    plt.show()


def quad_hist(a, b, c, d, c_bins, alpha):
    weights_0 = np.ones_like(a) / (0.01 * float(len(a)))
    weights_1 = np.ones_like(b) / (0.01 * float(len(b)))
    weights_2 = np.ones_like(c) / (0.01 * float(len(c)))
    weights_3 = np.ones_like(d) / (0.01 * float(len(d)))
    a = log_moid(a)
    b = log_moid(b)
    c = log_moid(c)
    d = log_moid(d)
    plt.hist(a, c_bins, alpha=alpha, weights=weights_0, range=(0, 1))
    plt.hist(b, c_bins, alpha=alpha, weights=weights_1, range=(0, 1))
    plt.hist(c, c_bins, alpha=alpha, weights=weights_2, range=(0, 1))
    plt.hist(d, c_bins, alpha=alpha, weights=weights_3, range=(0, 1))
    plt.legend(labels=["Diboson", "EFT"])


def plot_line(plt_list, title, x_label, y_label):
    linear = list(range(1, len(plt_list) + 1))
    plt.plot(linear, plt_list)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    time_stamp = datetime.strftime(datetime.now(), "%m_%d_%Y_%H_%M_%S")
    file_name = "output_figures/" + title + "_" + time_stamp + ".png"
    plt.savefig(file_name)
    plt.show()
