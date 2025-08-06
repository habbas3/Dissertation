#!/usr/bin/python
# -*- coding:utf-8 -*-

import argparse
import matplotlib.pyplot as plt
import pickle
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
import random
import optuna
import json

SEED = 123  # Choose your own seed

torch.manual_seed(SEED)
numpy.random.seed(SEED)
random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)  # if you are using multi-GPU.
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

args = None
global_habbas3.init()          # Call only once




def save_variables(variables, filename):
    with open(filename, 'wb') as f:
        pickle.dump(variables, f)

def savefig(plot, filename):
    plot.savefig(filename)

def parse_args():
    parser = argparse.ArgumentParser(description='Train')
    # model and data parameters
    # parser.add_argument('--model_name', type=str, default='WideResNet_sa', help='the name of the model')
    parser.add_argument('--data_name', type=str, default='CWRU_inconsistent', help='the name of the data')
    # parser.add_argument('--data_name', type=str, default='CWRUFFT_inconsistent', help='the name of the data')
    parser.add_argument('--data_dir', type=str, default='/Users/moondiab/Documents/Dissertation/UDTL_Lable_Inconsistent-main/datasets/CWRU_dataset', help='the directory of the data') #D:/data/Paderborn_University_Bearing_Data            D:/data/CWRU
    # parser.add_argument('--transfer_task', type=list, default=[[0], [1]], help='transfer learning tasks')
    parser.add_argument('--normlizetype', type=str, default='mean-std', help='nomalization type')
    parser.add_argument('--method', type=str, default='deterministic', choices=['deterministic', 'sngp'])
    parser.add_argument('--gp_hidden_dim', type=int, default=2048)
    parser.add_argument('--spectral_norm_bound', type=float, default=0.95)
    parser.add_argument('--n_power_iterations', type=int, default=1)
    parser.add_argument('--nesterov', default=True, type=bool, help='nesterov momentum')
    # parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
    #                     help='weight decay (default: 5e-4)')
    parser.add_argument('--print-freq', '-p', default=10, type=int,
                        help='print frequency (default: 10)')
    parser.add_argument('--layers', default=16, type=int, #28
                        help='total number of layers (default: 28)')
    parser.add_argument('--widen-factor', default=1, type=int, #2
                    help='widen factor (default: 10)') #0.1
    parser.add_argument('--droprate', default=0.3, type=float,
                    help='dropout probability (default: 0.0)')

    # training parameters
    parser.add_argument('--cuda_device', type=str, default='0', help='assign device')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint', help='the directory to save the model')
    parser.add_argument("--pretrained", type=bool, default=False, help='whether to load the pretrained model')
    parser.add_argument('--batch_size', type=int, default=64, help='batchsize of the training process') #Hadi 128, Zhao 64
    parser.add_argument('--num_workers', type=int, default=0, help='the number of training process')

    parser.add_argument('--bottleneck', type=bool, default=True, help='whether using the bottleneck layer')
    # parser.add_argument('--bottleneck_num', type=int, default=256, help='whether using the bottleneck layer')
    parser.add_argument('--bottleneck_num', type=int, default=256, help='whether using the bottleneck layer')
    parser.add_argument('--last_batch', type=bool, default=False, help='whether using the last batch')

    parser.add_argument('--hidden_size', type=int, default=1024, help='whether using the last batch') #Hadi 256. Zhao 1024
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
    parser.add_argument('--middle_epoch', type=int, default=30, help='max number of epoch') #Hadi 0, Zhao 50
    parser.add_argument('--max_epoch', type=int, default=100, help='max number of epoch') #Hadi 100, Zhao 300
    parser.add_argument('--print_step', type=int, default=50, help='the interval of log training information')

    parser.add_argument('--inconsistent', type=str, choices=['PADA', 'OSBP', 'UAN'], default='UAN', help='which adversarial loss you use')
    parser.add_argument('--th', type=float, default=0.5, help='theshold')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    
    def save_variables(variables, filename):
        with open(filename, 'wb') as f:
            pickle.dump(variables, f)

    def savefig(plot, filename):
        plot.savefig(filename)  
    
    def run_experiment(args, save_dir):
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

if __name__ == '__main__':
    args = parse_args()
    
    trial_results = []
    # best_configs = {}
    # best_hscores = {}

    
    def objective(trial, transfer_task, model_architecture):
        args.transfer_task = transfer_task
        args.model_name = model_architecture
        # Suggest hyperparameters
        args.lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
        args.batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
        args.hidden_size = trial.suggest_categorical("hidden_size", [128, 256, 512, 1024])
        args.dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
        args.bottleneck_num = trial.suggest_int('bottleneck_num', 128, 256, 512)
        # args.bottleneck = 
        # args.method = trial.suggest_categorical("method", ['sngp', 'deterministic'])
        args.method = trial.suggest_categorical("method", ['sngp'])


        # Your existing training setup and execution
        save_dir = os.path.join(args.checkpoint_dir, "optuna_trial")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        trainer = train_utils_open_univ(args, save_dir)
        trainer.setup()
        # validation_accuracy = trainer.train()  # Ensure this returns a validation metric
        best_hscore = trainer.train()
        
        trial_result = {
        'trial_number': trial.number,
        'params': trial.params,
        'best_hscore': best_hscore
        }
        
        # Append to the global list
        trial_results.append(trial_result)
        
        # Optionally write to file (appending to the file in each trial)
        # with open('trial_results.json', 'a') as f:
        #     f.write(json.dumps(trial_result) + '\n')
        
        return best_hscore
        # return validation_accuracy

    
    def run_optuna_study(transfer_task,model_architecture, num_trials=15):
        study = optuna.create_study(study_name=f'habbas_transfer_study_{model_architecture}_{transfer_task}', direction='maximize')
        study.optimize(lambda trial: objective(trial, transfer_task, model_architecture), n_trials=num_trials)

        best_trial = study.best_trial

        print("Best trial:")
        print(f"  Value: {best_trial.value}")
        print("  Params: ")
        for key, value in best_trial.params.items():
            print(f"    {key}: {value}")

        return best_trial
    
    
    

    # Define transfer learning tasks
    transfer_tasks = [
        [[0], [1]],
        [[0], [2]], 
        [[0], [3]],
        [[1], [0]], 
        [[1], [2]], 
        [[1], [3]], 
        [[2], [0]], 
        [[2], [1]], 
        [[2], [3]], 
        [[3], [0]], 
        [[3], [1]], 
        [[3], [2]]
        
    ]

    # Define model architectures
    model_architectures = [
        'cnn_features_1d',
        # 'cnn_features_1d_sa', 
        # 'cnn_openmax',
        # 'WideResNet', 
        # 'WideResNet_sa', 
        # 'WideResNet_edited'
    ]

    for transfer_task in transfer_tasks:
        for model_architecture in model_architectures:
            global_habbas3.init()
            global_habbas3.accuracy_score = 0
            args.transfer_task = transfer_task
            args.model_name = model_architecture

            os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_device.strip()
            sub_dir = args.model_name + '_' + str(args.transfer_task)
            datestamp = datetime.now().strftime("%m%d%Y")
            save_dir = os.path.join(args.checkpoint_dir, f"OptimizatedHyperPar_CWRU_{sub_dir}_{datestamp}")
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            if args.model_name != 'cnn_features_1d':
                best_trial = run_optuna_study(transfer_task, model_architecture, num_trials=15)
                args.lr = best_trial.params['lr']
                args.batch_size = best_trial.params['batch_size']
                args.hidden_size = best_trial.params['hidden_size']
                args.dropout_rate = best_trial.params['dropout_rate']
                args.bottleneck_num = best_trial.params['bottleneck_num']

            # Run experiment with the best hyperparameters
            best_hscore = run_experiment(args, save_dir)
            
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
            # Save plot
            plt.savefig(os.path.join(save_dir, 'source_training_experiment_plot.png'))
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
            
            plt.savefig(os.path.join(save_dir, 'source_validation_comparison_plot.png'))
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
            
            plt.savefig(os.path.join(save_dir, 'target_validation_comparison_plot.png'))
            plt.show()

            #SNGP HABBAS3
            amount_target = global_habbas3.amount_target
            correct_target = global_habbas3.correct_target
            probs_list = global_habbas3.probs_list
            accuracy_score = global_habbas3.accuracy_score
            best_hscore = global_habbas3.best_hscore
    
            probs = numpy.array(probs_list)
            pred_probs = probs.max(1)
            pred_probs = 1-pred_probs
    
    
            from sklearn.metrics import precision_recall_curve, auc
            from sklearn.preprocessing import label_binarize
            
            # Assuming target_val_labels is a 1D array of labels
            num_classes = numpy.unique(target_val_labels).size
            target_val_binarized = label_binarize(target_val_labels, classes=range(num_classes))
            
            # Calculate AUPRC for each class
            auprc_scores = []
            for i in range(num_classes):
                # Binarize predictions for the current class
                pred_probs_class = (pred_probs == i).astype(int)
                
                precision, recall, _ = precision_recall_curve(target_val_binarized[:, i], pred_probs_class)
                auprc = auc(recall, precision)
                auprc_scores.append(auprc)
                print(f'Class {i} AUPRC: {auprc:.4f}')
            
            # Calculate average AUPRC
            average_auprc = numpy.mean(auprc_scores)
            print(f'Average AUPRC: {average_auprc:.4f}')
            
            #SNGP CALCULATION
            import numpy as np
            from sklearn.metrics import accuracy_score, precision_recall_curve, auc
            from sklearn.preprocessing import label_binarize
            

            
            # Convert probability predictions to class labels
            predicted_labels = np.argmax(target_val_predictions, axis=1)
            
            # Calculate accuracy
            accuracy = accuracy_score(target_val_labels, predicted_labels)
            
            # Binarize the labels for multi-class AUPRC calculation
            num_classes = np.unique(target_val_labels).size
            target_val_labels_binarized = label_binarize(target_val_labels, classes=range(num_classes))
            
            # Calculate AUPRC for each class
            auprc_scores = []
            for i in range(num_classes):
                precision, recall, _ = precision_recall_curve(target_val_labels_binarized[:, i], probs[:, i])
                auprc = auc(recall, precision)
                auprc_scores.append(auprc)
            
            # Calculate average AUPRC
            average_auprc = np.mean(auprc_scores)
            
            # Calculate H-score
            common_acc = target_common_acc
            outlier_acc = target_outlier_acc
            hscore = 2 * common_acc * outlier_acc / (common_acc + outlier_acc) if (common_acc + outlier_acc) != 0 else 0
            
            # SNGP accuracy score can be a weighted combination of these metrics
            # You can define the weights based on the importance of each metric in your context
            sngp_accuracy_score = (accuracy + hscore + average_auprc) / 3
            print("Accuracy Score:", accuracy)
            print("Best H-Score:", hscore)
            print("Average AUPRC:", average_auprc)
            print("SNGP Accuracy Score:", sngp_accuracy_score)
            print("target common accuracy", target_common_acc)
            print("target outlier accuracy", target_outlier_acc)
            print("target validation accuracy", target_val_acc)
            
        # Save variables
        variables_to_save = {
            'source_train_acc': source_train_acc,  
            'source_val_acc': source_val_acc,
            'target_val_acc': target_val_acc,  
            'target_outlier_acc': target_outlier_acc,
            'target_common_acc': target_common_acc,  
            'accuracy_score': accuracy_score,
            'amount_target': amount_target,  
            'auprc': average_auprc,
            'correct_target': correct_target,  
            'precision': precision,
            'pred_probs': pred_probs,  
            'probs': probs,
            'recall': recall,
            'source_val_labels': source_val_labels,  
            'source_val_predictions': source_val_predictions,
            'target_val_labels': target_val_labels,
            'target_val_predictions': target_val_predictions,
            'transfer_task': transfer_task,  
            'model_architecture': model_architecture,
            'best_hscore': best_hscore,
            'optimal_lr': args.lr,
            'optimal_batch_size': args.batch_size,
            'optimal_hidden_size': args.hidden_size,
            # 'optimal_dropout_rate': args.dropout_rate,
            'optimal_bottleneck_num': args.bottleneck_num,
            'SNGP_Accuracy_Score:': sngp_accuracy_score,
            
        }
        save_variables(variables_to_save, os.path.join(save_dir, 'comparison_variables.pkl'))

    # Save best configurations and h-scores for all transfer tasks and model architectures
           
            
    #Load Results
    # file_path = os.path.join(save_dir, 'comparison_variables.pkl')
    # with open(file_path, 'rb') as file:
    #     # Load the data from the file
    #     data = pickle.load(file)
    
    

