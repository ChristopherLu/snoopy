# seq2seq without attention
import ConfigParser
config = ConfigParser.ConfigParser()
config.read('config.ini')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import csv
import tensorflow as tf
from keras import backend as K
config_tf = tf.ConfigProto()
config_tf.gpu_options.visible_device_list = config.get('swipe', 'gpu')
sess = tf.Session(config=config_tf)
K.set_session(sess)

from keras.models import Sequential
from seq2seq import SimpleSeq2Seq, Seq2Seq, AttentionSeq2Seq
from keras.callbacks import ModelCheckpoint
from keras.layers.core import *
from keras.layers.wrappers import *
from keras.models import Model
from keras.layers import Input, LSTM, Dense
from keras.optimizers import *
from keras.preprocessing.sequence import *
from utility.indb_prepare_prepare import *

max_length = max_fea_seq_length(X_train)
print('max length is {:}'.format(max_length))
X_train = pad_sequences(X_train, maxlen=max_length, dtype=np.float64)
X_val = pad_sequences(X_val, maxlen=max_length, dtype=np.float64)

# config seq_cater
input_dim = X_train[0].shape[1]
output_dim = int(config.get('swipe', 'output_dim'))
hidden_dim = int(config.get('swipe', 'hidden_dim'))
output_length = int(config.get('swipe', 'output_length'))
epoch_num = int(config.get('swipe', 'epoch_num'))
batch_size = int(config.get('swipe', 'batch_size'))
learning_rate = float(config.get('swipe', 'learning_rate'))
depth = int(config.get('swipe', 'depth'))
drop_out_ratio = float(config.get('swipe', 'drop_out_ratio'))

# model name
model_name = 'seq' + '_' + str(hidden_dim) + '_' + str(learning_rate) + '_' + str(epoch_num) \
             + '_' + str(X_train.shape[0]) + '_' + str(drop_out_ratio) + '_' + str(depth)
print('model name is {}'.format(model_name))
model_weights_name = "model_" + model_name + ".hdf5"
model_weights_path = './tmp/models/' + model_weights_name
directory = './tmp/models/'
if not os.path.exists(directory):
    os.makedirs(directory)

checkpoint = ModelCheckpoint(model_weights_path, monitor='val_acc', verbose=1,
                             save_best_only=True, save_weights_only=True, mode='max')
callbacks_list = [checkpoint]

model = Sequential()
model.add(Seq2Seq(output_dim=output_dim, hidden_dim=hidden_dim,
                  output_length=output_length, input_shape=(X_train[0].shape[0], X_train[0].shape[1]),
                  dropout=drop_out_ratio, depth=depth, peek=True))
# model.add(SimpleSeq2Seq(output_dim=output_dim, hidden_dim=hidden_dim, output_length=output_length,
#                         input_shape=(X_train[0].shape[0], X_train[0].shape[1])))
model.add(TimeDistributed(Dense(output_dim)))
model.add(Activation('softmax'))

model.summary()
optimizer = Adam()
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
hist = model.fit(X_train, y_train_flat, epochs=epoch_num, batch_size=batch_size,
                 validation_data=(X_val, y_val_flat), shuffle=True,
                 verbose=1,  callbacks=callbacks_list)

# write out as files
directory = './history/'
if not os.path.exists(directory):
    os.makedirs(directory)
csvName = './history/hist_' + model_name + '.csv'

with open(csvName, 'ab') as fp:
    a = csv.writer(fp)
    for key, val in hist.history.items():
        a.writerow([key, val])

# move final models from ./tmp/models to ./models
model_weights_path_new = './models/' + model_weights_name
shutil.move(model_weights_path, model_weights_path_new)
