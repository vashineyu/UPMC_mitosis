from __future__ import print_function

import os
import sys

import numpy as np
import pandas as pd
import time
import glob
import re
import random
import datetime

from sklearn.model_selection import train_test_split
from PIL import Image
import cv2

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', type = str)
parser.add_argument('--split_id', type = str, help = 'split_id, from 0 to 7')
parser.add_argument('--full_random', default = 0)
parser.add_argument('--with_transfer', default = 1)
parser.add_argument('--message', type = str, help = "Recording experiment infomation")
FLAGS = parser.parse_args()
print(FLAGS)

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = str(FLAGS.gpu_id)
import tensorflow as tf

from config import Basic_Setup, Hparams, Augmentation_Setup
from DataGenerator import Create_Data_Generator

sys.path.append("/home/seanyu/Common_tools/")
from model import create_model
from trainer import model_trainer
from CONFIG_MODEL import PRETRAIN_MODEL_SETUP
from Recording import Experiment_Recoding
from TF_callbacks import (
    EarlyStopping, 
    Model_checkpoint, 
    ReduceLROnPlateau, 
    Model_Timer,
    Run_collected_functions
)

recording_manager = Experiment_Recoding(Basic_Setup.result_dir, message=FLAGS.message)
start_time = time.time()

if not FLAGS.full_random:
    f_imgs = os.listdir(Basic_Setup.data_dir)
    f_imgs = [os.path.join(Basic_Setup.data_dir, i) for i in f_imgs]

    training_files = pd.read_csv(os.path.join(Basic_Setup.label_dir, "sp"+FLAGS.split_id, 'tr_lst'), header=None)
    training_files.columns = ['pid', 'cate']

    validation_files = pd.read_csv(os.path.join(Basic_Setup.label_dir, "sp"+FLAGS.split_id, 'tt_lst'), header=None)
    validation_files.columns = ['pid', 'cate']

    training_files['img_path'] = training_files['pid'].apply(lambda x: os.path.join(Basic_Setup.data_dir, 'image' + str (x) + ".png"))
    validation_files['img_path'] = validation_files['pid'].apply(lambda x: os.path.join(Basic_Setup.data_dir, 'image' + str (x) + ".png"))

else:
    df = pd.DataFrame()
    for root_path, middle_path, target_file in os.walk(Basic_Setup.label_dir):
        if len(target_file) == 0:
            pass
        else:
            for sub_file in target_file:
                full_name = os.path.join(root_path, sub_file)
                tmp_files = pd.read_csv(full_name, header=None)
                df = pd.concat((df, tmp_files), axis = 0)
    df.columns = ["pid","cate"]
    df.drop_duplicates(inplace=True)
    df['img_path'] = df['pid'].apply(lambda x: os.path.join(Basic_Setup.data_dir, 'image' + str (x) + ".png"))
    training_files, validation_files = train_test_split(df, test_size = 0.1, stratify =df.cate.values.astype('int')) # around 1:10

train_0 = training_files[training_files['cate'] == 0]
train_1 = training_files[training_files['cate'] == 1]
train_0.reset_index(drop=True, inplace=True)
train_1.reset_index(drop=True, inplace=True)

data_gen = Create_Data_Generator(df = [train_0, train_1],
                                 image_size=Hparams.input_size,
                                 batch_size=Hparams.batch_size, n_classes=Hparams.n_class,
                                 nd_inputs_preprocessing_handler=PRETRAIN_MODEL_SETUP.CORRESPONDING_PREPROC[Hparams.pretrain_model],
                                 do_augment=Basic_Setup.do_augmentation, aug_params=Augmentation_Setup.augmentation
                                )


data_gen.start_train_threads(jobs=Basic_Setup.n_threads, dq_size=Basic_Setup.dq_size)
train_gen = data_gen.get_data()
x_val, y_val = data_gen.get_evaluate_data(validation_files)
print(x_val.shape)

cb_dict = {
    'reduce_lr' : ReduceLROnPlateau(lr=Hparams.reduce_lr_options['lr'], 
                                    factor=Hparams.reduce_lr_options['factor'], 
                                    patience=Hparams.reduce_lr_options['patience']),
    'earlystop' : EarlyStopping(min_delta = Hparams.earlystop_options['min_delta'], 
                                patience= Hparams.earlystop_options['patience']),
    'checkpoint' : Model_checkpoint(model_name=Basic_Setup.result_dir + '/' +  Basic_Setup.model_saving_name, 
                                    save_best_only=True),
    'model_timer': Model_Timer(),
}

callback_dict = {
    'on_session_begin':[cb_dict['model_timer']], # start of a session
    'on_batch_begin':[], # start of a training batch
    'on_batch_end':[], # end of a training batch
    'on_epoch_begin':[], # start of a epoch
    'on_epoch_end':[cb_dict['earlystop'], 
                    cb_dict['reduce_lr'],
                    cb_dict['checkpoint']], # end of a epoch
    'on_session_end':[cb_dict['model_timer']] # end of a session
    }
callback_manager = Run_collected_functions(callback_dict)


model_ops, metric_history = create_model(Hparams)
trainer = model_trainer(hflags=Hparams, # hyper-parameters
                        data_gen=data_gen, # data generator, for get infos
                        data_gen_get_data = train_gen, # for yield patches
                        model_ops=model_ops, # model graph
                        metric_history=metric_history, # metric recording
                        callback_manager=callback_manager # runable callbacks
                        )
trainer.initalize()
if FLAGS.with_transfer:
    pretrain_ckpt = PRETRAIN_MODEL_SETUP.PRETRAIN_DICT[Hparams.pretrain_model]
    trainer.restore(model_to_restore=pretrain_ckpt, partial_restore=True)
trainer.do_training(cb_dict=cb_dict, validation_set=(x_val, y_val))

val_pred = trainer.predict(x_= x_val, 
                           model_to_restore=os.path.join(Basic_Setup.result_dir, Basic_Setup.model_saving_name) + '.ckpt', 
                           bz=Hparams.batch_size)

val_pred = np.concatenate([item for sublist in val_pred for item in sublist])
acc = np.sum(val_pred.argmax(axis = 1) == y_val.argmax(axis = 1)) / len(y_val)
print(acc)

validation_files['y_pred'] = val_pred[:,1]

if not FLAGS.full_random:
    output_df_name = os.path.join(Basic_Setup.result_dir, 'result_sp' + str(FLAGS.split_id) + '.csv')
    history_file_name = os.path.join(Basic_Setup.result_dir, 'log_sp' + str(FLAGS.split_id) + '.csv')
else:
    output_df_name = os.path.join(Basic_Setup.result_dir, 'result_randomsp' + str(FLAGS.split_id) + '.csv')
    history_file_name = os.path.join(Basic_Setup.result_dir, 'log_randomsp' + str(FLAGS.split_id) + '.csv')

validation_files.to_csv(path_or_buf=output_df_name, index=False)

recording_manager.write_new_msg('Validation Performance Accuracy: %.3f' % (acc) )

history_logs = pd.DataFrame({'train_accuracy': trainer.metric_history['accuracy']['train'],
                             'valid_accuracy': trainer.metric_history['accuracy']['valid']})

history_logs.to_csv(history_file_name, index=False)