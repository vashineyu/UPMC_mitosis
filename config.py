import os
import sys

class Basic_Setup(object):
    data_dir = "/data/seanyu/UPMC_pathology/Pathology_Analysis/data/" # path to data
    label_dir = "/data/seanyu/UPMC_pathology/Pathology_Analysis/splits/" # path to label
    result_dir = "/data/seanyu/UPMC_pathology/result/resnet50_RandomSplit" # path to store results
    model_saving_name = "model" # model name prefix
    do_augmentation = False
    
    n_threads = 8
    dq_size = 50
    
# ======================= #
try:
    import imgaug as ia
    from imgaug import augmenters as iaa
except:
    print("Import Error, Please make sure you have imgaug")
        
try:
    sys.path.append("Common_tools/")
    from customized_imgaug_func import keypoint_func, img_channelswap
except:
    print("Warning, if you used customized imgaug function")

class Augmentation_Setup(object):  
    sometimes = lambda aug: iaa.Sometimes(0.5, aug)
    lesstimes = lambda aug: iaa.Sometimes(0.2, aug)
    
    augmentation = iaa.Sequential([
        iaa.Fliplr(0.5, name="FlipLR"),
        iaa.Flipud(0.5, name="FlipUD"),
        iaa.ContrastNormalization((0.8, 1.2), name = "Contrast"),
        iaa.Add((-15, 15), per_channel = 0.5),
   #     iaa.OneOf([iaa.Multiply((0.8, 1.2), per_channel = 0.5, name = "Multiply"),
   #                iaa.AddToHueAndSaturation((-15,15),name = "Hue"),
   #               ]),
        sometimes(iaa.GaussianBlur((0, 1.5), name="GaussianBlur")),
        iaa.OneOf([iaa.Affine(rotate = 90),
                   iaa.Affine(rotate = 180),
                   iaa.Affine(rotate = 270)]),
        sometimes(iaa.Affine(
                    scale = (0.6,1.2),
                    #translate_percent = (-0.2, 0.2),
                    rotate = (-15, 15),
                    mode = 'wrap'
                    ))
    ])
    
# ============================== #
import tensorflow as tf
class Hparams(object):
    """
    Model related setup
    """
    input_size = (256, 256, 3)
    batch_size = 32
    epochs = 200
    n_batch = 300
    n_class = 2
    class_weight = None # leave None if don't use it
    optimizer = 'sgd'
    pretrain_model = "resnet_50"
    
    learning_rate = 0.00017
    reduce_lr_options = {'lr':learning_rate, 'factor':0.5, 'patience':4}
    earlystop_options = {'min_delta': 1e-4, 'patience':9}
