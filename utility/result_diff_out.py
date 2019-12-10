import os


def diff(first, second):
    second = set(second)
    return [item for item in first if item not in second]


result_dir = '../results/out_db/'
if not os.path.exists(result_dir):
    os.makedirs(result_dir)
model_dir = '../results/out_db/'

# develop the list of models
for root, subdirs, files in os.walk(result_dir):
    old_f = [f[:-4] for f in files if f[-3:] == 'mat']
for root, subdirs, files in os.walk(model_dir):
    new_f = [f[:-5] for f in files if f[-4:] == 'pckl']

# model difference
diff_f = diff(new_f, old_f)
print('going to use {}'.format(diff_f))

# save to txt
target_file = open('../tmp/new_added_models.txt', 'w')

for item in diff_f:
    item = item + '.pckl'
    target_file.write("%s\n" % item)