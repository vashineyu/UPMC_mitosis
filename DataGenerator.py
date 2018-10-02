import os
import sys
import glob
import numpy as np
import pandas as pd
import time
from threading import Thread, Event
import queue
import cv2
import random
import tensorflow as tf

class Create_Data_Generator():
    def __init__(self,
                 df,
                 image_size,
                 nd_inputs_preprocessing_handler = None,
                 batch_size = 32,
                 n_classes = 2,
                 do_augment = False,
                 aug_params = None):
        """
        Define parameters and dataset
        
        Args:
            - df: list of pandas dataframe. [df_class0, df_class1]
            - image_size: input_image size
            - nd_inputs_preprocessing_handler: input preprocessing function
            - batch_size: size per batch yield
            - n_classes: numbers of target classes
            - do_augment: to do augmentation on inputs?
            - aug_params: augmentation options
        """
        
        self.df = df
        self.f_inputs_preproc = nd_inputs_preprocessing_handler # how to do image preprocessing
        self.image_size = image_size
        self.bz = batch_size
        self.n_classes = n_classes
        self.do_augment = do_augment
        self.aug = aug_params
    
    def get_train_data(self):
        """
        Function to get a single batch. 
        Yield a balanced batch
        """
        while True:
            idxs = self.train_idx_queue.get()
            select_list = []

            for df, idx in zip(self.df, idxs):
                select_list.append(df.iloc[idx])
            select_list = pd.concat(select_list)

            x_ = np.array([cv_load_and_resize(iid, image_size = self.image_size, 
                                              is_training = True, 
                                              do_augment = self.do_augment, seq = self.aug) for iid in select_list.img_path], dtype=np.float32)
            x_ = x_.astype(np.float32)
            """ do preprocessing here"""
            if self.f_inputs_preproc:
                x_ = self.f_inputs_preproc(x_)
            else:
                pass

            """ Y out """
            y_ = np.array(select_list['cate'])
            y_ = tf.keras.utils.to_categorical(y_, self.n_classes)
            
            yield x_, y_


    def get_data(self):
        """
        Function to connect outside part (called function)
        """
        while True:
            x_, y_ = self.train_sample_queue.get()
            yield x_, y_
                
    def get_evaluate_data(self, target_df):
        """
        Get validation set
        Args:
            - target_df: pandas dataframe that list img_path & cate (category)
        """
        x_ = np.array([cv_load_and_resize(i, image_size = self.image_size, is_training = False) for i in target_df.img_path], dtype=np.float32) # don't do augmentation!
    
        """ do preprocessing here"""
        if self.f_inputs_preproc:
            x_ = self.f_inputs_preproc(x_)
        else:
            pass
        
        """ Y out """
        y_ = np.array(target_df['cate'])
        y_ = tf.keras.utils.to_categorical(y_, num_classes=self.n_classes)
        
        return x_, y_
    
    def _get_train_idx(self):
        """ Description 
        Get training data index for each data frame in the data list
        # note1: self.df should be list of data frame with different categories
        # note2: if there is only 1 class (or for regression problem, should still be embraced [this_df] )
        """
        len_list = [len(df) for df in self.df]
        
        bz_t = self.bz//len(len_list)
        batch_num = [x//bz_t for x in len_list]

        batch_nth = [0] * len(len_list)

        select = [list(range(x)) for x in len_list]

        for s in select:
            random.shuffle(s)

        while True:
            idxs = []
            for i in range(len(len_list)):
                if batch_nth[i] >= batch_num[i]:
                    batch_nth[i] = 0
                    random.shuffle(select[i])
                idx = select[i][batch_nth[i]*bz_t:(batch_nth[i]+1)*bz_t]
                batch_nth[i] += 1
                idxs.append(idx)

            yield idxs
    
    def start_train_threads(self, jobs = 1, dq_size = 10):
        
        self.train_sample_queue = queue.Queue(maxsize = dq_size * 5)
        self.train_idx_queue =queue.Queue(maxsize = dq_size * 100)
        ### for stop threads after training ###
        self.events= list()
        self.threading = list()

        ### enqueue train index ###
        event = Event()
        thread = Thread(target = enqueue, 
                        args = (self.train_idx_queue, 
                                event, 
                                self._get_train_idx))
        thread.daemon = True 
        thread.start()
        self.events.append(event)
        self.threading.append(thread)

        ### enqueue train samples
        for i in range(jobs):
            event = Event()
            thread = Thread(target = enqueue, args = (self.train_sample_queue,
                                                      event,
                                                      self.get_train_data))
            thread.daemon = True 
            thread.start()
            self.events.append(event)
            self.threading.append(thread)

    def stop_train_threads(self):
        """
        Stop the threading
        """
        # block until all tasks are done
        for t in self.events:
            t.set()
        
        self.train_sample_queue.queue.clear()
        self.train_idx_queue.queue.clear()
        
        for i, t in enumerate(self.threading):
            t.join(timeout=1)
            print("Stopping Thread %i" % i)


# ----- #
def enqueue(queue, stop, gen_func):
    gen = gen_func()
    while True:
        if stop.is_set():
            return
        queue.put(next(gen))
        
        
def cv_load_and_resize(x, image_size, is_training = True, do_augment = False, seq = None):
    im_w, im_h, im_c = image_size
    im = cv2.imread(x)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    if im.shape != image_size:
        im = cv2.resize(im, (im_w, im_h))
    if do_augment and is_training:
        im = seq.augment_image(im)
    return im