import scipy.io as sio
import numpy as np
import collections
import re
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm
from joblib import Parallel, delayed
import multiprocessing

import warnings
warnings.filterwarnings("ignore")


def flag2index(x):
    return {
        'acc': range(3),
        'gyro': range(3, 6),
    }.get(x, range(6))  # range(6) is default if x not found


def retrive_mat(mat_path, slice_flag):
    mat_contents = sio.loadmat(mat_path)
    fea = mat_contents['fea']
    label = mat_contents['label']
    index_range = flag2index(slice_flag)
    fea = fea[:, index_range]
    return fea, label


def max_fea_seq_length(x):
    max_length = 0
    for i in x:
        tmp_length = i.shape[0]
        if max_length < tmp_length:
            max_length = tmp_length

    return max_length


def length_stat(x):
    l = []
    for i in x:
        l.append(i.shape[0])
    return l


def length_filter(x, upper_bound):
    x = list(x)
    l = [i for i in x if (i[0].shape[0] < upper_bound)]
    return l


def partial_confusion_matrix(gt, pred, total_classes):
    partial_observation = list(set(gt))
    cm = np.zeros((total_classes, total_classes))
    for item in partial_observation:
        tmp_pos = np.where(gt == item)
        tmp_pred = pred[tmp_pos]
        tmp_counter = collections.Counter(tmp_pred)
        tmp_row = np.zeros(total_classes)
        for tup in tmp_counter.iteritems():
            tmp_row[tup[0]] = tup[1]
        tmp_row_normalized = tmp_row/np.sum(tmp_row, dtype=np.float32)
        cm[item] = tmp_row_normalized

    return cm


def unlock_baseline(nb_pwd, nb_trial):
    tmp = 1
    for i in range(nb_trial):
        tmp = tmp * (nb_pwd-i-1)/float(nb_pwd-i)

    return 1-tmp


def prob2decision(pred_prob, top_k):
    pred_top_k = []
    for instance in pred_prob:
        tmp = [i[0] for i in sorted(enumerate(instance), reverse=True, key=lambda x: x[1])]
        pred_top_k.append(tmp[:top_k])

    return np.array(pred_top_k)


def flatten(l):
    l_f = [item for sublist in l for item in sublist]

    return l_f


def fill_zeros(pwd_ls):
    out_ls =[]
    for pwd in pwd_ls:
        tmp = str(pwd).ljust(9, '0')
        out_ls.append(tmp)
    return out_ls


def num2list(x):
    out_ls = list()
    for num in x:
        out_ls.append([[int(i)] for i in str(num)])
    
    return out_ls

def digit_encoder():
    digits = range(0, 10)
    str_ls = [[str(i)] for i in digits]
    le = OneHotEncoder(categorical_features = [0])
    le.fit(str_ls)

    return le


def top_k_guesses(logits, ground_truth, num_chances):
    logits, ground_truth = np.array(logits), np.array(ground_truth)
    prediction_top_indices = prob2decision(logits, num_chances)
    apls = all_apl_generator()
    guessed_apls = [apls[pred] for pred in prediction_top_indices]
    correct_prediction_counter = 0
    for truth, guesses in zip(ground_truth, guessed_apls):
        if truth in guesses:
            correct_prediction_counter += 1
    refined_acc = np.array(correct_prediction_counter, dtype=np.float32) / ground_truth.shape[0]
    return refined_acc


def pwd2seq(pwd):
#     print(pwd)
    seq = []
    enc = digit_encoder()
    for digit in pwd:
        digit_enc = enc.transform([[digit]]).toarray()
        seq.append(flatten(digit_enc))

    return seq


def seq_ls_build(pwd_ls):
    seq_ls = []
    for pwd in pwd_ls:
        seq = pwd2seq(pwd)
        seq_ls.append(seq)

    return seq_ls


def model_name_parser(model_name):
    para_list = []
    pattern = re.compile('_*_')
    for m in re.finditer(pattern, model_name):
        para_list.append((m.start(), m.end()))

    model = model_name[para_list[0][1]: para_list[1][0]]
    hidden_dim = int(model_name[para_list[1][1]: para_list[2][0]])
    lr = float(model_name[para_list[2][1]: para_list[3][0]])
    depth = int(model_name[para_list[6][1]])

    return model, hidden_dim, lr, depth


def all_apl_generator(int_flag=False):
    try:
        tmp = np.loadtxt('./auxilary/all_apl.txt').astype(np.int)
    except:
        tmp = np.loadtxt('../auxilary/all_apl.txt').astype(np.int)
    if int_flag:
        out = tmp
    else:
        out = np.array([str(i).ljust(9, '0') for i in tmp])

    return out


def apl_log_lik(apl, seq):
    log_lik = 0
    length = len(apl)
    # look at each
    for i in range(0, length):
        prob_arr = seq[i]
        apl_dot = int(apl[i])
        log_lik += np.log10(prob_arr[apl_dot]+1)
    return log_lik


def apl_filter(apls, length):
    indices = []
    for i, elem in enumerate(apls):
        if len(elem) == length:
            indices.append(i)

    return indices


def gt2str(gt):
    out = []
    for i in gt:
        out.append(''.join(str(e) for e in i))
    return out


def apls_encoder(apls, seq):
    lik_ls = []
    for apl in apls:
        log_lik = apl_log_lik(apl, seq)
        lik_ls.append(log_lik)
    return lik_ls


def decode_prob(arr):
    # arr is a 3d array with shape (num_sample, len_seq, dim)
    apls = all_apl_generator()
    # for seq in tqdm(arr):
    #     # examine all pins on
    #     lik_ls = apls_encoder(apls, seq)
    #     out.append(lik_ls)
    num_cores = multiprocessing.cpu_count()
    # num_cores = np.min([num_cores, 24])
    print('Apl encoding using {} cores'.format(num_cores))
    out = Parallel(n_jobs=num_cores)(
        delayed(apls_encoder)(apls, seq) for seq in tqdm(arr))

    out = np.array(out)
    return out


def decode_prob_with_length(prediction, labels):
    out = []
    apls_int = all_apl_generator(int_flag=True)
    apls = [str(i) for i in apls_int]
    for pred, gt in tqdm(zip(prediction, labels)):
        # get the length of seq
        length = np.count_nonzero(gt)
        # filter apls_int with length
        valid_indices = apl_filter(apls, length)
        # examine all pins on
        lik_arr = np.zeros(apls_int.shape[0])
        for index in valid_indices:
            apl = apls[index]
            log_lik = apl_log_lik(apl, pred)
            lik_arr[index] = log_lik
        out.append(lik_arr)
    out = np.array(out)
    return out


def decode_ground_truth(ground_truth):
    out = []
    for prob_arr in ground_truth:
        pwd = []
        for digit in prob_arr:
            pwd.append(np.argmax(digit))
        out.append(pwd)

    out = np.array(out)
    return out


def sub_sample(arr, freq=200):
    step = 200/freq
    out = list()
    for sample in arr:
        d_sample = sample[::step, :]
        out.append(d_sample)
    return np.array(out)

