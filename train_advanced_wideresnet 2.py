#!/usr/bin/python
# -*- coding:utf-8 -*-

import argparse
import os
from datetime import datetime
from utils.logger import setlogger
import logging
import global_habbas3
from utils.train_utils_combines import train_utils
from utils.train_utils_open_univ import train_utils_open_univ
import torch
import numpy
import sklearn
import warnings
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_curve, auc
print(torch.__version__)
warnings.filterwarnings('ignore')

import faulthandler; faulthandler.enable()

args = None
global_habbas3.init()          # Call only once

import pickle
import matplotlib.pyplot as plt

def save_variables(variables, filename):
    with open(filename, 'wb') as f:
        pickle.dump(variables, f)

def save_plot(plot, filename):
    plot.savefig(filename)

def parse_args():
    parser = argparse.ArgumentParser(description='Train')
    # model and data parameters
    parser.add_argument('--model_name', type=str, default='WideResNet_mh', help='the name of the model')
    parser.add_argument('--data_name', type=str, default='CWRU_inconsistent', help='the name of the data')
    parser.add_argument('--data_dir', type=str, default='/Users/moondiab/Documents/Dissertation/UDTL_Lable_Inconsistent-main/datasets/CWRU_dataset', help='the directory of the data') #D:/data/Paderborn_University_Bearing_Data            D:/data/CWRU
    parser.add_argument('--transfer_task', type=list, default=[[0], [1]], help='transfer learning tasks')
    parser.add_argument('--normlizetype', type=str, default='mean-std', help='nomalization type')
    parser.add_argument('--method', type=str, default='sngp', choices=['deterministic', 'sngp'])
    parser.add_argument('--gp_hidden_dim', type=int, default=2048)
    parser.add_argument('--spectral_norm_bound', type=float, default=0.95)
    parser.add_argument('--n_power_iterations', type=int, default=1)
    parser.add_argument('--nesterov', default=True, type=bool, help='nesterov momentum')
    # parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
    #                     help='weight decay (default: 5e-4)')
    parser.add_argument('--print-freq', '-p', default=10, type=int,
                        help='print frequency (default: 10)')
    parser.add_argument('--layers', default=28, type=int,
                        help='total number of layers (default: 28)')
    parser.add_argument('--widen-factor', default=2, type=int,
                    help='widen factor (default: 10)')
    parser.add_argument('--droprate', default=0, type=float,
                    help='dropout probability (default: 0.0)')

    # training parameters
    parser.add_argument('--cuda_device', type=str, default='0', help='assign device')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint', help='the directory to save the model')
    parser.add_argument("--pretrained", type=bool, default=False, help='whether to load the pretrained model')
    parser.add_argument('--batch_size', type=int, default=128, help='batchsize of the training process')
    parser.add_argument('--num_workers', type=int, default=0, help='the number of training process')

    parser.add_argument('--bottleneck', type=bool, default=True, help='whether using the bottleneck layer')
    # parser.add_argument('--bottleneck_num', type=int, default=256, help='whether using the bottleneck layer')
    parser.add_argument('--bottleneck_num', type=int, default=256, help='whether using the bottleneck layer')
    parser.add_argument('--last_batch', type=bool, default=False, help='whether using the last batch')

    parser.add_argument('--hidden_size', type=int, default=256, help='whether using the last batch')
    parser.add_argument('--trade_off_adversarial', type=str, default='Step', help='')
    parser.add_argument('--lam_adversarial', type=float, default=1, help='this is used for Cons')

    # optimization information
    parser.add_argument('--opt', type=str, choices=['sgd', 'adam'], default='adam', help='the optimizer')
    parser.add_argument('--lr', type=float, default=1e-3, help='the initial learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='the momentum for sgd')
    parser.add_argument('--weight-decay', type=float, default=1e-5, help='the weight decay')
    parser.add_argument('--lr_scheduler', type=str, choices=['step', 'exp', 'stepLR', 'fix'], default='step', help='the learning rate schedule')
    parser.add_argument('--gamma', type=float, default=0.1, help='learning rate scheduler parameter for step and exp')
    parser.add_argument('--steps', type=str, default='150, 250', help='the learning rate decay for step and stepLR')

    # save, load and display information
    parser.add_argument('--middle_epoch', type=int, default=0, help='max number of epoch')
    parser.add_argument('--max_epoch', type=int, default=300, help='max number of epoch') #300
    parser.add_argument('--print_step', type=int, default=50, help='the interval of log training information')

    parser.add_argument('--inconsistent', type=str, choices=['PADA', 'OSBP', 'UAN'], default='UAN', help='which adversarial loss you use')
    parser.add_argument('--th', type=float, default=0.5, help='theshold')

    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_device.strip()
    # Prepare the saving path for the model
    sub_dir = args.model_name + '_' + datetime.strftime(datetime.now(), '%m%d-%H%M%S')
    save_dir = os.path.join(args.checkpoint_dir, sub_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # set the logger
    setlogger(os.path.join(save_dir, 'train.log'))

    # save the args
    for k, v in args.__dict__.items():
        logging.info("{}: {}".format(k, v))

    if args.inconsistent == 'OSBP' or args.inconsistent == 'UAN':
        trainer = train_utils_open_univ(args, save_dir)
    else:
        trainer = train_utils(args, save_dir)

    trainer.setup()
    trainer.train()
    
    #Source Training data
    source_train_acc_tor = global_habbas3.source_train_acc
    source_train_acc = numpy.array(source_train_acc_tor)
    source_train_labels_tor = (global_habbas3.source_train_labels)
    source_train_labels = numpy.array(source_train_labels_tor)
    source_train_predictions_tor = (global_habbas3.source_train_predictions)
    source_train_predictions = numpy.array(source_train_predictions_tor)
    conf_matrix_source_train = sklearn.metrics.confusion_matrix(source_train_labels, source_train_predictions)
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix_source_train)
    disp.plot()
    disp.ax_.set_title("Source Training Data")
    plt.show()
    
    #Source Validation data
    source_val_acc_tor = global_habbas3.source_val_acc
    source_val_acc = numpy.array(source_val_acc_tor)
    source_val_labels_tor = (global_habbas3.source_val_labels)
    source_val_labels = numpy.array(source_val_labels_tor)
    source_val_predictions_tor = (global_habbas3.source_val_predictions)
    source_val_predictions = numpy.array(source_val_predictions_tor)
    conf_matrix_source_val = sklearn.metrics.confusion_matrix(source_val_labels, source_val_predictions)
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix_source_val)
    disp.plot()
    disp.ax_.set_title("Source Validation Data")
    plt.show()
    
    #Target Validation data
    target_val_labels_tor = (global_habbas3.target_val_labels)
    target_val_labels = numpy.array(target_val_labels_tor)
    target_val_predictions_tor = (global_habbas3.target_val_predictions)
    target_val_predictions = numpy.array(target_val_predictions_tor)
    tmp = target_val_predictions.argmax(1)
    target_val_acc_tor = global_habbas3.target_val_acc
    target_val_acc = numpy.array(target_val_acc_tor)
    target_outlier_acc = global_habbas3.target_outlier_acc
    target_common_acc = global_habbas3.target_common_acc
    
    conf_matrix_target_val = sklearn.metrics.confusion_matrix(target_val_labels, tmp)
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix_target_val)
    disp.plot()
    disp.ax_.set_title("Target Validation Data")
    plt.show()


#SNGP HABBAS3
amount_target = global_habbas3.amount_target
correct_target = global_habbas3.correct_target
probs_list = global_habbas3.probs_list
accuracy_score = global_habbas3.accuracy_score

probs = numpy.array(probs_list)
pred_probs = probs.max(1)
pred_probs = 1-pred_probs


precision, recall, _ = sklearn.metrics.precision_recall_curve(target_val_labels, pred_probs, pos_label=3)

auprc = sklearn.metrics.auc(recall, precision)
print(f'SNGP AUPRC: {auprc:.4f}')


def save_variables(variables, filename):
    with open(filename, 'wb') as f:
        pickle.dump(variables, f)
        
        
sub_dir = args.model_name + '_' + str(args.transfer_task)
datestamp = datetime.now().strftime("%m%d%Y")
save_dir = "WideResNet_mh_[[0], [1]]"

args.model_architecture = 'WideResNet_mh'
variables_to_save = {
    'source_train_acc': source_train_acc,  
    'source_val_acc': source_val_acc,
    'target_val_acc': target_val_acc,  
    'target_outlier_acc': target_outlier_acc,
    'target_common_acc': target_common_acc,  
    'accuracy_score': accuracy_score,
    'amount_target': amount_target,  
    'auprc': auprc,
    'correct_target': correct_target,  
    'precision': precision,
    'pred_probs': pred_probs,  
    'probs': probs,
    'recall': recall,
    'source_val_labels': source_val_labels,  
    'source_val_predictions': source_val_predictions,
    'target_val_labels': target_val_labels,
    'target_val_predictions': target_val_predictions,
    'transfer_task': args.transfer_task,  
    'model_architecture': args.model_architecture,
    
}
save_variables(variables_to_save, os.path.join(save_dir, 'comparison_variables.pkl'))
# prob_true, prob_pred = sklearn.calibration.calibration_curve(target_val_labels, pred_probs, pos_label=3, n_bins=10, strategy='quantile')

