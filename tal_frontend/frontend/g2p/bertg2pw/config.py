root = './data_process/'

manual_seed = 1313
model_source = '/mnt/cfs/NLP/hub_models/bert-base-multilingual-cased'
polyphonic_chars_path = './data_process/POLYPHONIC_CHARS.txt'
window_size = None
num_workers = 2
use_mask = True
use_conditional = True
param_conditional = {
    'bias': True,
    'char-linear': True,
    'pos-linear': False,
    'char+pos-second': True,
}

# for training
exp_name = 'mul'
train_sent_path = root + 'train_data.json'
train_lb_path = root + 'train.lb'
valid_sent_path = root + 'dev_data.json'
valid_lb_path = root + 'dev.lb'
test_sent_path = root + 'test_data.json'
test_lb_path = root + 'test.lb'
batch_size = 128
lr = 5e-5
val_interval = 200
num_iter = 10000
use_pos = False
param_pos = {
    'weight': 0.1,
    'pos_joint_training': False,
    'train_pos_path': root + 'train.pos',
    'valid_pos_path': root + 'dev.pos',
    'test_pos_path': root + 'test.pos'
}
