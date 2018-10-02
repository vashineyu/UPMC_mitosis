## UPMC mitosis classification
Framework: Tensorflow v1.8

This version of uploaded repo is only for visualization (see [vis.ipynb](https://github.com/vashineyu/UPMC_mitosis/blob/master/vis.ipynb) )
To see several images example, please see [show_image_examples.ipynb](https://github.com/vashineyu/UPMC_mitosis/blob/master/show_image_examples.ipynb)
If want to run the code, You'll need "Common_tools" which only a soft-link in the repo, you can add it manually from https://github.com/vashineyu/Common_tools.

The Common_tools contains pretrain-weights, corresponding preprocessing function, etc.
To run the training, either run.py (for single run) or go.sh (run over splits)
[NOTE] Remember to change the data_dir / label_dir / result_dir in the config.py

To change the setting, see and modify the config.py
To change the whole model structure, see and modify the model.py

If you have any other questions, please contact seanyu@aetherai.com or r01227113@g.ntu.edu.tw

===
### Table summary
| Model Setting | Accuracy | AUC | Accuracy @best threshold |
| ------------- | --------:| ---:| ------------------------:|
| setting1      | 0.961 +/- (0.01) | 0.987 +/- (0.01) | 0.957 +/- (0.01) |
| setting2      | 0.966 +/- (0.01) | 0.985 +/- (0.01) | 0.969 +/- (0.01) |
| setting3      | 0.956 +/- (0.01) | 0.992 +/- (0.00) | 0.967 +/- (0.01) |
| setting4      | 0.960 +/- (0.01) | 0.989 +/- (0.01) | 0.969 +/- (0.01) |
| setting5      | 0.963 +/- (0.00) | 0.986 +/- (0.01) | 0.957 +/- (0.00) |
| setting6      | 0.973 +/- (0.00) | 0.992 +/- (0.01) | 0.974 +/- (0.00) |

* setting1: Resnet_50 / pretrained / sgd, same split, input size: 256 x 256
* seeting2: Resnet_50 / pretrained / sgd, random split, input size: 256 x 256
* setting3: Inception_Resnet / pretrained / sgd, same split, input size: 256 x 256
* setting4: Inception_Resnet / pretrained / sgd, random split, input size: 256 x 256
* setting5: Inception_Resnet / pretrained / adam, random split, input size: 256 x 256
* setting6: Resnet_50 / no-pretrained / adam, random split, input size: 256 x 256





