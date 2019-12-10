import cPickle as pickle
import re
import csv
from keras.models import Sequential
from seq2seq import SimpleSeq2Seq, Seq2Seq, AttentionSeq2Seq
from keras.layers.core import *
from keras.layers.wrappers import *
from keras.models import Model
from keras.layers import Input, LSTM, Dense
from keras.preprocessing.sequence import *
import sys
import ConfigParser
from keras.optimizers import *
from tqdm import tqdm
from utility.indb_prepare_prepare import *

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# grasp fea_flag from sys inputs
if len(sys.argv) > 1:
    tmp = sys.argv[1]
    model_name = tmp[:-5]

# Fix the model name
model_path = './models/' + model_name + '.hdf5'
para_list = []
pattern = re.compile('_*_')
for m in re.finditer(pattern, model_name):
    para_list.append((m.start(), m.end()))

net_name, hidden_dim, lr, depth = model_name_parser(model_name)

# hard-code output dim and length
output_dim = 10
output_length = 9

# zero padding inputs
max_length = max_fea_seq_length(X_train)
print('max_length of x_train is {}'.format(max_length))
X_test = pad_sequences(X_test, maxlen=max_length, dtype=np.float64)

# load model
model = Sequential()
if 'att' in net_name:
    model.add(AttentionSeq2Seq(output_dim=output_dim, hidden_dim=hidden_dim,
                               output_length=output_length, input_shape=(X_test[0].shape[0], X_test[0].shape[1]),
                               dropout=0, depth=depth))
elif 'seq' in net_name:
    model.add(Seq2Seq(output_dim=output_dim, hidden_dim=hidden_dim,
                      output_length=output_length, input_shape=(X_test[0].shape[0], X_test[0].shape[1]),
                      depth=depth, peek=True))
model.add(TimeDistributed(Dense(10)))
model.add(Activation('softmax'))

model.load_weights(model_path)
optimizer = Adam()
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
print("Successfully loaded the model of " + model_name)

# estimate accuracy on whole dataset using loaded weights
scores = model.evaluate(X_test, y_test_flat, verbose=0)
print("We got accuracy of {:.2%}".format(scores[1]))

# save for plot
batch_size = int(float(100)/hidden_dim * 300)
prediction_prob = model.predict(X_test, batch_size=batch_size, verbose=0)
ground_truth = y_test_flat

if not os.path.exists('./results/in_db/'):
    os.makedirs('./results/in_db/')
result_path = './results/in_db/' + model_name + '.pckl'

with open(result_path, 'w') as f:
    pickle.dump([prediction_prob, ground_truth], f)
    f.close()
