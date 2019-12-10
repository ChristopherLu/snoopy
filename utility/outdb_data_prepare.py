# This script take .mat files as inputs and outputs training data for RNN classifier
import os
import shutil
from shutil import copyfile
from functions import *
import ConfigParser

# # clearn and build
# test_dir = '../../../data/generative_attack/mat_test'
# if os.path.isdir(test_dir):
#     shutil.rmtree(test_dir)
#     os.mkdir(test_dir)
# print(' dir /mat_test cleared...')
#
# # use matlab script to derive s
# eng = matlab.engine.start_matlab()
# mfile_path = '../../matlab/pre_processing/generative_attack/'
# eng.addpath(eng.genpath(mfile_path), nargout=0)
# eng.preprocess_main_test(nargout=0)

# target folder to iterate .mat files
if os.getcwd() == '/media/sambaShare/Xiaoxuan_Lu/password_watch/Code_base/Python/rnn_classifier':
    train_dir = '/media/sambaShare/Xiaoxuan_Lu/password_watch/mat_data'
else:
    train_dir = '../../../data/cross_attack/mat_train'
    val_dir = '../../../data/cross_attack/mat_val'
    test_dir = '../../../data/generative_attack/mat_test'

# iterate train dir to load train .mat into np.array
X_train, y_train = [], []
for filename in os.listdir(train_dir):
    if filename.endswith(".mat"):
        fea, label = retrive_mat(os.path.join(train_dir, filename), 'accgyro')
        X_train.append(fea)
        y_train.append(label)

X_train, y_train = np.array(X_train), np.array(y_train).astype(np.int32)

# iterate val dir to load val .mat into np.array
X_val, y_val = [], []
for filename in os.listdir(val_dir):
    if filename.endswith(".mat"):
        fea, label = retrive_mat(os.path.join(val_dir, filename), 'accgyro')
        X_val.append(fea)
        y_val.append(label)

X_val, y_val = np.array(X_val), np.array(y_val).astype(np.int32)

# iterate test dir to load test mat into np.array
X_test, y_test = [], []
for filename in os.listdir(test_dir):
    if filename.endswith(".mat"):
        fea, label = retrive_mat(os.path.join(test_dir, filename), 'accgyro')
        X_test.append(fea)
        y_test.append(label)

X_test, y_test = np.array(X_test), np.array(y_test).astype(np.int32)

# build output sequence
y_train_flat, y_val_flat, y_test_flat = flatten(y_train), flatten(y_val), flatten(y_test)
y_train_flat, y_val_flat, y_test_flat = fill_zeros(y_train_flat), fill_zeros(y_val_flat), fill_zeros(y_test_flat)
y_train_flat, y_val_flat, y_test_flat = seq_ls_build(y_train_flat), seq_ls_build(y_val_flat), seq_ls_build(y_test_flat)

# write out for analysis
train_length = length_stat(X_train)
val_length = length_stat(X_val)
np.savetxt('./tmp/pwd_length_stat.out', train_length + val_length, delimiter=',', fmt='%i')

# grap config
config = ConfigParser.ConfigParser()
config.read('auxilary/config.ini')
ds = config.getboolean('data_prepare', 'use')
freq = config.getint('data_prepare', 'downsample_freq')
max_len = config.getint('data_prepare', 'max_len')

# filter out samples whose length exceeds upper-bound
tmp = np.array(length_filter(zip(X_train, y_train_flat), max_len))
X_train, y_train_flat = np.array(zip(*tmp)[0]), np.array(zip(*tmp)[1])
tmp = np.array(length_filter(zip(X_val, y_val_flat), max_len))
X_val, y_val_flat = np.array(zip(*tmp)[0]), np.array(zip(*tmp)[1])
tmp = np.array(length_filter(zip(X_test, y_test_flat), max_len))
X_test, y_test_flat = np.array(zip(*tmp)[0]), np.array(zip(*tmp)[1])

# downsample if necessary
if ds:
    print('X_train[0].shape is {}'.format(X_train[0].shape))
    X_train, X_val, X_test = sub_sample(X_train, freq), sub_sample(X_val, freq), sub_sample(X_test, freq)
    print('After downsampling, X_train[0].shape is {}'.format(X_train[0].shape))
