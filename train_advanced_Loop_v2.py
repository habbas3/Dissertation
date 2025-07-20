#!/usr/bin/python
# -*- coding:utf-8 -*-

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
import numpy as np
import sklearn
import warnings
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_curve, auc
import faulthandler; faulthandler.enable()
import random
import optuna
import json
from my_datasets.CWRU_label_inconsistent import CWRU_inconsistent
from my_datasets.Battery_label_inconsistent import load_battery_dataset
from collections import Counter
from models.optuna_search import run_optuna_search
import pandas as pd
from torch.utils.data import Dataset, DataLoader


import sys
sys.path.append(os.path.dirname(__file__))

print(torch.__version__)
warnings.filterwarnings('ignore')

SEED = 123
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def parse_args():
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('--data_name', type=str, default='Battery_inconsistent')
    parser.add_argument('--data_dir', type=str, default='./my_datasets/Battery')
    parser.add_argument('--csv', type=str, default='./my_datasets/Battery/battery_data_labeled.csv')
    parser.add_argument('--normlizetype', type=str, default='mean-std')
    parser.add_argument('--method', type=str, default='determenistic', choices=['deterministic', 'sngp'])
    parser.add_argument('--gp_hidden_dim', type=int, default=2048)
    parser.add_argument('--spectral_norm_bound', type=float, default=0.95)
    parser.add_argument('--n_power_iterations', type=int, default=1)
    parser.add_argument('--nesterov', type=bool, default=True)
    parser.add_argument('--print-freq', '-p', default=10, type=int)
    parser.add_argument('--layers', default=16, type=int)
    parser.add_argument('--widen-factor', default=1, type=int)
    parser.add_argument('--droprate', default=0.3, type=float)
    parser.add_argument('--cuda_device', type=str, default='0')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint')
    parser.add_argument('--pretrained', type=bool, default=False)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--bottleneck', type=bool, default=True)
    parser.add_argument('--bottleneck_num', type=int, default=256)
    parser.add_argument('--last_batch', type=bool, default=False)
    parser.add_argument('--hidden_size', type=int, default=1024)
    parser.add_argument('--trade_off_adversarial', type=str, default='Step')
    parser.add_argument('--lam_adversarial', type=float, default=1)
    parser.add_argument('--opt', type=str, choices=['sgd', 'adam'], default='adam')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=1e-5)
    parser.add_argument('--lr_scheduler', type=str, default='step')
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--steps', type=str, default='150, 250')
    parser.add_argument('--middle_epoch', type=int, default=30) #30
    parser.add_argument('--max_epoch', type=int, default=100) #100
    parser.add_argument('--print_step', type=int, default=50) #50
    parser.add_argument('--inconsistent', type=str, default='UAN')
    parser.add_argument('--th', type=float, default=0.5)
    parser.add_argument('--input_channels', type=int, default=7)
    parser.add_argument('--classification_label', type=str, default='eol_class')
    parser.add_argument('--sequence_length', type=int, default=32)
    parser.add_argument('--source_cathode', nargs='+', default=["NMC532", "NMC811", "HE5050", "NMC111"])
    parser.add_argument('--target_cathode', nargs='+', default=["NMC622", "5Vspinel"])
    parser.add_argument('--domain_temperature', type=float, default=1.0,
                            help='Temperature scaling for domain predictions')
    parser.add_argument('--class_temperature', type=float, default=10.0,
                            help='Temperature scaling for class predictions')
    return parser.parse_args()

def run_experiment(args, save_dir, trial=None):
    setlogger(os.path.join(save_dir, 'train.log'))
    for k, v in vars(args).items():
        logging.info(f"{k}: {v}")

    # ✅ Load dataset using cathode filters
    source_train_dataset, source_val_dataset, target_train_dataset, target_val_dataset, label_names, df = load_battery_dataset(
        csv_path=args.csv,
        source_cathodes=args.source_cathode,
        target_cathodes=args.target_cathode,
        classification_label=args.classification_label,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
    )
    args.num_classes = len(label_names)

    # ✅ Build dataloaders
    source_train_loader = DataLoader(source_train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    source_val_loader = DataLoader(source_val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)

    # ✅ Build target loaders *only if available*
    if target_train_dataset is not None:
        target_train_loader = DataLoader(target_train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    else:
        target_train_loader = None

    if target_val_dataset is not None:
        target_val_loader = DataLoader(target_val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
    else:
        target_val_loader = None

    # ✅ Inject Optuna trial hyperparameters
    if trial is not None:
        args.lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
        args.hidden_size = trial.suggest_categorical("hidden_size", [256, 512, 1024])
        args.bottleneck_num = trial.suggest_categorical("bottleneck_num", [128, 256])
        args.droprate = trial.suggest_float("droprate", 0.1, 0.5)

    # ✅ Call trainer fully universally now:
    trainer = train_utils_open_univ(
        args, save_dir,
        source_train_loader, source_val_loader,
        target_train_loader, target_val_loader,
        source_train_dataset, target_val_dataset
    )

    trainer.setup()
    return trainer.train()



def main():
    args = parse_args()
    global_habbas3.init()
    df_all = pd.read_csv(args.csv)
    
    df_all["cathode"] = df_all["cathode"].astype(str).str.strip()
    all_cathodes = sorted(df_all["cathode"].unique().tolist())

    # Define cathodes
    # pretrain_cathodes = ["HE5050", "NMC111", "NMC532", "FCG", "NMC811"]
    # transfer_cathodes = ["NMC622", "Li1.2Ni0.3Mn0.6O2", "Li1.35Ni0.33Mn0.67O2.35"]
    pretrain_cathodes = [
        "HE5050",
        "NMC111",
        "NMC532",
        "FCG",
        "NMC811",
        "Li1.2Ni0.3Mn0.6O2",
    ]

    transfer_cathodes = [
        "NMC622",
        "5Vspinel",
        "Li1.35Ni0.33Mn0.67O2.35",
    ]

    model_architectures = [
        "cnn_features_1d",
        "cnn_features_1d_sa",
        "cnn_openmax",
        "WideResNet",
        "WideResNet_sa",
        "WideResNet_edited",
    ]

    # skip_pretraining = False

    # if not skip_pretraining:
    #     print("🔧 Starting Pretraining per cathode type")

    #     for cathode in pretrain_cathodes:
    #         # Use all rows belonging to this cathode
    #         args.source_cathode = [cathode]
    #         args.target_cathode = []

    #         for model_name in model_architectures:
    #             global_habbas3.init()
    #             args.model_name = model_name
    #             os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device.strip()

    #             pretrain_dir = os.path.join(args.checkpoint_dir, f"Pretrain_{model_name}_{cathode}_{datetime.now().strftime('%m%d')}")
    #             os.makedirs(pretrain_dir, exist_ok=True)

    #             model_pre, src_acc = run_experiment(args, pretrain_dir)
    #             args.pretrained = True
    #             args.pretrained_model_path = os.path.join(pretrain_dir, "best_model.pth")
    #             print(f"✅  Pretrained {model_name} on {cathode}: src_val_acc={src_acc:.4f}")
    
    results = []

    # Transfer Learning Stage
    # for target_cathode in transfer_cathodes:
        # ---- Baseline Training on each cathode independently ----
    print("\n📊 Baseline training (target only)")
    for cathode in transfer_cathodes:
        args.source_cathode = [cathode]
        args.target_cathode = [cathode]
        for model_name in model_architectures:
            global_habbas3.init()
            args.model_name = model_name

            # Use all pretrain_cathodes except target as source_cathodes
            # args.source_cathode = [c for c in pretrain_cathodes if c != target_cathode]
            # args.target_cathode = [target_cathode]
            args.pretrained = False
            args.pretrained_model_path = None
            os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device.strip()

            # print(f"\n🚀 Running Optuna for {model_name} → {target_cathode}")

            # # Define pretrained checkpoint directory for this model
            # # (here we simply use the first source cathode pretrained file for simplicity)
            # pretrained_cathode = args.source_cathode[0]
            # pretrained_dir = os.path.join(args.checkpoint_dir, f"Pretrain_{model_name}_{pretrained_cathode}_{datetime.now().strftime('%m%d')}")
            # args.pretrained_model_path = os.path.join(pretrained_dir, "best_model.pth")
            # args.transfer = True
            
            base_dir = os.path.join(args.checkpoint_dir, f"baseline_{model_name}_{cathode}_{datetime.now().strftime('%m%d')}")
            os.makedirs(base_dir, exist_ok=True)
            _, base_acc = run_experiment(args, base_dir)
            results.append({"cathode": cathode, "model": model_name, "baseline": base_acc})

            # run_optuna_search(args, model_name, n_trials=25)
            
            # ---- Pretrain on other cathodes then fine-tune on target ----
    print("\n🔧 Transfer learning per cathode")
    for cathode in transfer_cathodes:
        other_cathodes = [c for c in pretrain_cathodes if c != cathode]
        for model_name in model_architectures:
            global_habbas3.init()
            args.model_name = model_name
            os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device.strip()

            # best_params = json.load(open(os.path.join(args.checkpoint_dir, f"optuna_{model_name}/{model_name}_best_params.json")))
            # final_dir = os.path.join(args.checkpoint_dir, f"optuna_{model_name}/final")
            # os.makedirs(final_dir, exist_ok=True)
            # model_ft, tgt_acc = run_experiment(args, final_dir, None)
            # print(f"✅  Fine-tuned {model_name} → {target_cathode}: tgt_val_acc={tgt_acc:.4f}")
            
            # Pretrain on all other cathodes
            args.source_cathode = other_cathodes
            args.target_cathode = []
            pre_dir = os.path.join(args.checkpoint_dir, f"pretrain_{model_name}_{cathode}_{datetime.now().strftime('%m%d')}")
            os.makedirs(pre_dir, exist_ok=True)
            _, _ = run_experiment(args, pre_dir)

            # Fine-tune on target cathode
            args.pretrained = True
            args.pretrained_model_path = os.path.join(pre_dir, "best_model.pth")
            args.source_cathode = [cathode]
            args.target_cathode = [cathode]
            ft_dir = os.path.join(args.checkpoint_dir, f"transfer_{model_name}_{cathode}_{datetime.now().strftime('%m%d')}")
            os.makedirs(ft_dir, exist_ok=True)
            _, transfer_acc = run_experiment(args, ft_dir)

            # Update results list with transfer accuracy
            base_value = 0
            for r in results:
                if r["cathode"] == cathode and r["model"] == model_name:
                    base_value = r["baseline"]
                    r["transfer"] = transfer_acc
                    break
            print(f"✅ {model_name} on {cathode}: baseline -> {base_value:.4f} | transfer -> {transfer_acc:.4f}")

    # Print final summary
    print("\n===== Summary =====")
    for r in results:
        b = r.get("baseline", 0)
        t = r.get("transfer", 0)
        print(f"{r['model']} {r['cathode']}: baseline {b:.4f} → transfer {t:.4f}")


if __name__ == '__main__':
    main()




# if __name__ == '__main__':
#     args = parse_args()
    
#     trial_results = []
#     # best_configs = {}
#     # best_hscores = {}

    
#     def objective(trial, transfer_task, model_architecture):
#         args.transfer_task = transfer_task
#         args.model_name = model_architecture
#         # Suggest hyperparameters
#         args.lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
#         args.batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
#         args.hidden_size = trial.suggest_categorical("hidden_size", [128, 256, 512, 1024])
#         args.dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
#         args.bottleneck_num = trial.suggest_int('bottleneck_num', 128, 256, 512)
#         # args.bottleneck = 
#         # args.method = trial.suggest_categorical("method", ['sngp', 'deterministic'])
#         args.method = trial.suggest_categorical("method", ['sngp'])


#         # Your existing training setup and execution
#         save_dir = os.path.join(args.checkpoint_dir, "optuna_trial")
#         if not os.path.exists(save_dir):
#             os.makedirs(save_dir)
#         trainer = train_utils_open_univ(args, save_dir)
#         trainer.setup()
#         # validation_accuracy = trainer.train()  # Ensure this returns a validation metric
#         best_hscore = trainer.train()
        
#         trial_result = {
#         'trial_number': trial.number,
#         'params': trial.params,
#         'best_hscore': best_hscore
#         }
        
#         # Append to the global list
#         trial_results.append(trial_result)
        
#         # Optionally write to file (appending to the file in each trial)
#         # with open('trial_results.json', 'a') as f:
#         #     f.write(json.dumps(trial_result) + '\n')
        
#         return best_hscore
#         # return validation_accuracy

    
#     def run_optuna_study(transfer_task,model_architecture, num_trials=15):
#         study = optuna.create_study(study_name=f'habbas_transfer_study_{model_architecture}_{transfer_task}', direction='maximize')
#         study.optimize(lambda trial: objective(trial, transfer_task, model_architecture), n_trials=num_trials)

#         best_trial = study.best_trial

#         print("Best trial:")
#         print(f"  Value: {best_trial.value}")
#         print("  Params: ")
#         for key, value in best_trial.params.items():
#             print(f"    {key}: {value}")

#         return best_trial
    
    
    

#     # Define transfer learning tasks
#     transfer_tasks = get_transfer_tasks(args.data_name)


#     # Define model architectures
#     model_architectures = [
#         'cnn_features_1d',
#         'cnn_features_1d_sa', 
#         'cnn_openmax',
#         'WideResNet', 
#         'WideResNet_sa', 
#         'WideResNet_edited'
#     ]

#     for transfer_task in transfer_tasks:
#         for model_architecture in model_architectures:
#             global_habbas3.init()
#             global_habbas3.accuracy_score = 0
#             args.transfer_task = transfer_task
#             args.model_name = model_architecture

#             os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_device.strip()
#             sub_dir = args.model_name + '_' + str(args.transfer_task)
#             datestamp = datetime.now().strftime("%m%d%Y")
#             save_dir = os.path.join(args.checkpoint_dir, f"OptimizatedHyperPar_CWRU_{sub_dir}_{datestamp}")
#             if not os.path.exists(save_dir):
#                 os.makedirs(save_dir)

#             if args.model_name != 'cnn_features_1d':
#                 best_trial = run_optuna_study(transfer_task, model_architecture, num_trials=15)
#                 args.lr = best_trial.params['lr']
#                 args.batch_size = best_trial.params['batch_size']
#                 args.hidden_size = best_trial.params['hidden_size']
#                 args.dropout_rate = best_trial.params['dropout_rate']
#                 args.bottleneck_num = best_trial.params['bottleneck_num']

#             # Run experiment with the best hyperparameters
#             best_hscore = run_experiment(args, save_dir)
            
#             #Source Training data
#             source_train_acc_tor = global_habbas3.source_train_acc
#             source_train_acc = numpy.array(source_train_acc_tor)
#             source_train_labels_tor = (global_habbas3.source_train_labels)
#             source_train_labels = numpy.array(source_train_labels_tor)
#             source_train_predictions_tor = (global_habbas3.source_train_predictions)
#             source_train_predictions = numpy.array(source_train_predictions_tor)
#             conf_matrix_source_train = sklearn.metrics.confusion_matrix(source_train_labels, source_train_predictions)
#             import matplotlib.pyplot as plt
#             from sklearn.metrics import ConfusionMatrixDisplay
#             disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix_source_train)
#             disp.plot()
#             disp.ax_.set_title("Source Training Data")
#             # Save plot
#             plt.savefig(os.path.join(save_dir, 'source_training_experiment_plot.png'))
#             plt.show()
            
            
#             #Source Validation data
#             source_val_acc_tor = global_habbas3.source_val_acc
#             source_val_acc = numpy.array(source_val_acc_tor)
#             source_val_labels_tor = (global_habbas3.source_val_labels)
#             source_val_labels = numpy.array(source_val_labels_tor)
#             source_val_predictions_tor = (global_habbas3.source_val_predictions)
#             source_val_predictions = numpy.array(source_val_predictions_tor)
#             conf_matrix_source_val = sklearn.metrics.confusion_matrix(source_val_labels, source_val_predictions)
#             import matplotlib.pyplot as plt
#             from sklearn.metrics import ConfusionMatrixDisplay
#             disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix_source_val)
#             disp.plot()
#             disp.ax_.set_title("Source Validation Data")
            
#             plt.savefig(os.path.join(save_dir, 'source_validation_comparison_plot.png'))
#             plt.show()
            
#             #Target Validation data
#             target_val_labels_tor = (global_habbas3.target_val_labels)
#             target_val_labels = numpy.array(target_val_labels_tor)
#             target_val_predictions_tor = (global_habbas3.target_val_predictions)
#             target_val_predictions = numpy.array(target_val_predictions_tor)
#             tmp = target_val_predictions.argmax(1)
#             target_val_acc_tor = global_habbas3.target_val_acc
            
#             target_val_acc = numpy.array(target_val_acc_tor)
#             target_outlier_acc = global_habbas3.target_outlier_acc
#             target_common_acc = global_habbas3.target_common_acc
            
#             conf_matrix_target_val = sklearn.metrics.confusion_matrix(target_val_labels, tmp)
#             import matplotlib.pyplot as plt
#             from sklearn.metrics import ConfusionMatrixDisplay
#             disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix_target_val)
#             disp.plot()
#             disp.ax_.set_title("Target Validation Data")
            
#             plt.savefig(os.path.join(save_dir, 'target_validation_comparison_plot.png'))
#             plt.show()

#             #SNGP HABBAS3
#             amount_target = global_habbas3.amount_target
#             correct_target = global_habbas3.correct_target
#             probs_list = global_habbas3.probs_list
#             accuracy_score = global_habbas3.accuracy_score
#             best_hscore = global_habbas3.best_hscore
    
#             probs = numpy.array(probs_list)
#             pred_probs = probs.max(1)
#             pred_probs = 1-pred_probs
    
    
#             from sklearn.metrics import precision_recall_curve, auc
#             from sklearn.preprocessing import label_binarize
            
#             # Assuming target_val_labels is a 1D array of labels
#             num_classes = numpy.unique(target_val_labels).size
#             target_val_binarized = label_binarize(target_val_labels, classes=range(num_classes))
            
#             # Calculate AUPRC for each class
#             auprc_scores = []
#             for i in range(num_classes):
#                 # Binarize predictions for the current class
#                 pred_probs_class = (pred_probs == i).astype(int)
                
#                 precision, recall, _ = precision_recall_curve(target_val_binarized[:, i], pred_probs_class)
#                 auprc = auc(recall, precision)
#                 auprc_scores.append(auprc)
#                 print(f'Class {i} AUPRC: {auprc:.4f}')
            
#             # Calculate average AUPRC
#             average_auprc = numpy.mean(auprc_scores)
#             print(f'Average AUPRC: {average_auprc:.4f}')
            
#             #SNGP CALCULATION
#             import numpy as np
#             from sklearn.metrics import accuracy_score, precision_recall_curve, auc
#             from sklearn.preprocessing import label_binarize
            

            
#             # Convert probability predictions to class labels
#             predicted_labels = np.argmax(target_val_predictions, axis=1)
            
#             # Calculate accuracy
#             accuracy = accuracy_score(target_val_labels, predicted_labels)
            
#             # Binarize the labels for multi-class AUPRC calculation
#             num_classes = np.unique(target_val_labels).size
#             target_val_labels_binarized = label_binarize(target_val_labels, classes=range(num_classes))
            
#             # Calculate AUPRC for each class
#             auprc_scores = []
#             for i in range(num_classes):
#                 precision, recall, _ = precision_recall_curve(target_val_labels_binarized[:, i], probs[:, i])
#                 auprc = auc(recall, precision)
#                 auprc_scores.append(auprc)
            
#             # Calculate average AUPRC
#             average_auprc = np.mean(auprc_scores)
            
#             # Calculate H-score
#             common_acc = target_common_acc
#             outlier_acc = target_outlier_acc
#             hscore = 2 * common_acc * outlier_acc / (common_acc + outlier_acc) if (common_acc + outlier_acc) != 0 else 0
            
#             # SNGP accuracy score can be a weighted combination of these metrics
#             # You can define the weights based on the importance of each metric in your context
#             sngp_accuracy_score = (accuracy + hscore + average_auprc) / 3
#             print("Accuracy Score:", accuracy)
#             print("Best H-Score:", hscore)
#             print("Average AUPRC:", average_auprc)
#             print("SNGP Accuracy Score:", sngp_accuracy_score)
#             print("target common accuracy", target_common_acc)
#             print("target outlier accuracy", target_outlier_acc)
#             print("target validation accuracy", target_val_acc)
            
#         # Save variables
#         variables_to_save = {
#             'source_train_acc': source_train_acc,  
#             'source_val_acc': source_val_acc,
#             'target_val_acc': target_val_acc,  
#             'target_outlier_acc': target_outlier_acc,
#             'target_common_acc': target_common_acc,  
#             'accuracy_score': accuracy_score,
#             'amount_target': amount_target,  
#             'auprc': average_auprc,
#             'correct_target': correct_target,  
#             'precision': precision,
#             'pred_probs': pred_probs,  
#             'probs': probs,
#             'recall': recall,
#             'source_val_labels': source_val_labels,  
#             'source_val_predictions': source_val_predictions,
#             'target_val_labels': target_val_labels,
#             'target_val_predictions': target_val_predictions,
#             'transfer_task': transfer_task,  
#             'model_architecture': model_architecture,
#             'best_hscore': best_hscore,
#             'optimal_lr': args.lr,
#             'optimal_batch_size': args.batch_size,
#             'optimal_hidden_size': args.hidden_size,
#             # 'optimal_dropout_rate': args.dropout_rate,
#             'optimal_bottleneck_num': args.bottleneck_num,
#             'SNGP_Accuracy_Score:': sngp_accuracy_score,
            
#         }
#         save_variables(variables_to_save, os.path.join(save_dir, 'comparison_variables.pkl'))

    # Save best configurations and h-scores for all transfer tasks and model architectures
           
            
    #Load Results
    # file_path = os.path.join(save_dir, 'comparison_variables.pkl')
    # with open(file_path, 'rb') as file:
    #     # Load the data from the file
    #     data = pickle.load(file)
    
    

