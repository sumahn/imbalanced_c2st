'''Python code for implement simulations (Section 5.1)'''

import argparse 
from pathlib import Path
import pickle 
import torch
import numpy as np
import yaml
from typing import List

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from model import NeuralNet
from utils import GenerateData, MatConvert, find_optimal_cutoff, GenerateData

dtype = torch.float
device = torch.device('cuda:5')

def calc_power(n_minor : int, ir: list, level, n_sims: int, exp_number: int, 
               lr: float, n_epochs: int, batch_size : int, device, dtype, data_dir, data_name: str) -> float:
    """
    Method to implement experiments. 
    All the parameters can be controlled via the file 'config.yml'. 
    :param n_minor: sample size of minority class
    :param ir: imbalance ratio (ir = n_major / n_minor)
    :param level: significance level of test ; usually set to 0.05
    :param n_sims: number of simulations
    :param exp_number: the order of experiments
    :param lr: learning rate of Neural Network (default = 0.001)
    :param n_epochs: number of epochs for Neural Network (default = 500)
    :param batch_size: batch_size for Neural Network (default = 128)
    """
    
    np.random.seed(1203)
    torch.manual_seed(1203)
    torch.cuda.manual_seed(1203)
    torch.backends.cudnn.deterministic = True 
    
    acc_filename = "acc_{}_{}_{}_{}_{}_{}_{}.data".format(args.data_name, args.exp_number, n_minor, n_sims, lr, n_epochs, batch_size)
    pwr_filename = "pwr_{}_{}_{}_{}_{}_{}_{}.data".format(args.data_name, args.exp_number, n_minor, n_sims, lr, n_epochs, batch_size)
    d_filename = "default_{}_{}_{}_{}_{}_{}_{}.data".format(args.data_name, args.exp_number, n_minor, n_sims, lr, n_epochs, batch_size)
    acc_path = data_dir.joinpath(acc_filename)
    pwr_path = data_dir.joinpath(pwr_filename)
    d_path = data_dir.joinpath(d_filename)
    print(acc_filename)
    stats_a_dict = {}
    stats_d_dict = {}
    stats_p_dict = {}

    for r in ir:
        print(r)
        data_generator = GenerateData(n_minor, r)
        data = data_generator.normal_normal(mu0=0, cov0=1, mu1=0, cov1=0.8, dim=1)
        X = data[:, 0].reshape(-1,1)
        y = data[:, 1]
        y = MatConvert(y, device, dtype)
        stats_a = {
                'kNN': [],
                'Random Forest': [],
                'Balanced Random Forest': [],
                'LDA': [],
                'Logistic Regression': [],
                'Balanced Logistic Regression': [],
                'XGBoost': [],
                'NN': []
            }
        stats_p = {
                'kNN': [],
                'Random Forest': [],
                'Balanced Random Forest': [],
                'LDA':[],
                'Logistic Regression': [],
                'Balanced Logistic Regression': [],
                'XGBoost': [],
                'NN': []
            }
        stats_d = {
                'kNN': [],
                'Random Forest': [],
                'Balanced Random Forest': [],
                'LDA':[],
                'Logistic Regression': [],
                'Balanced Logistic Regression': [],
                'XGBoost': [],
                'NN': []
            }
        
        for sim in range(n_sims):
            print(sim+1)
            classifiers= {
            'kNN': KNeighborsClassifier(n_neighbors=int(np.sqrt(n_minor + n_minor*r))),
            'Random Forest': RandomForestClassifier(),
            'Balanced Random Forest': RandomForestClassifier(class_weight='balanced_subsample'),
            'LDA': LinearDiscriminantAnalysis(),
            'Logistic Regression': LogisticRegression(),
            'Balanced Logistic Regression': LogisticRegression(class_weight='balanced'),
            'XGBoost': XGBClassifier(),
            'NN': NeuralNet(x_in=1, H=50, x_out=2, device=device, dtype=dtype)
            }
            y_np = y.to('cpu').numpy()
        
            # generate train and test indices
            train_idx, test_idx, train_y, test_y = train_test_split(np.arange(len(y_np)), y_np, test_size=0.5, stratify=y_np)
        
            # generate validation indices 
            tr_idx, val_idx, tr_y, val_y = train_test_split(train_idx, y_np[train_idx], test_size=0.5, stratify=y_np[train_idx])

            cutvals = np.arange(start=0.001, stop=0.999, step=0.001)
            # find optimal thresholds that maximize accuracy and power for each classifier
            for name, clf in classifiers.items():
                accs = []
                pwrs = []
                if name == 'NN':
                    X_nn = MatConvert(X, device, dtype)
                    dataset = torch.utils.data.TensorDataset(X_nn[tr_idx], y[tr_idx])
                    dataloader = torch.utils.data.DataLoader(dataset, batch_size = batch_size, shuffle=True)
                    clf.fit(dataloader, lr, n_epochs)
                    y_val_prob = clf.predict_proba(X_nn[val_idx])[:, 1]
                else:
                    clf.fit(X[tr_idx], y_np[tr_idx])
                    y_val_prob = clf.predict_proba(X[val_idx])[:, 1]
                for cutval in cutvals:
                    y_val_pred = (y_val_prob >= cutval).astype(int)
                    accs.append(np.mean(y_val_pred == y_np[val_idx]))
                    y0_val = y_np[val_idx][y_np[val_idx] == 0]
                    y1_val = y_np[val_idx][y_np[val_idx] == 1]
                    tau0 = np.mean((y_val_pred[y_np[val_idx] == 0] == 1))
                    tau1 = np.mean((y_val_pred[y_np[val_idx] == 1] == 0))
                    pwrs.append((1 - tau0 - tau1) / np.sqrt((tau0*(1-tau0)/len(y0_val)) + (tau1*(1-tau1)/len(y1_val))))
                
                accs = list(np.nan_to_num(accs))
                pwrs = list(np.nan_to_num(pwrs))
                
                acc_opt_cut = cutvals[accs.index(max(accs))]                    
                pwr_opt_cut = cutvals[pwrs.index(max(pwrs))]
            
                classifiers[name] = (clf, acc_opt_cut, pwr_opt_cut)
            
            # test on test set based on optimal cutoffs
            for name, (clf, acc_opt_cut, pwr_opt_cut) in classifiers.items():
                if name == 'NN':
                    dataset = torch.utils.data.TensorDataset(X_nn[train_idx], y[train_idx])
                    dataloader = torch.utils.data.DataLoader(dataset, batch_size = batch_size, shuffle=False)
                    clf.fit(dataloader, lr, n_epochs)
                else:
                    clf.fit(X[train_idx], y_np[train_idx])
                    y_test_prob = clf.predict_proba(X[test_idx])
                
                y_test_pred_acc = (y_test_prob[:, 1] >= acc_opt_cut).astype(int)
                y_test_pred_pwr = (y_test_prob[:, 1] >= pwr_opt_cut).astype(int)
                y_test_pred_d = (y_test_prob[:, 1] > y_test_prob[:, 0]).astype(int)
                
                tau0_a = np.mean(y_test_pred_acc[y_np[test_idx] == 0] == 1)
                tau1_a = np.mean(y_test_pred_acc[y_np[test_idx] == 1] == 0)
                tau0_p = np.mean(y_test_pred_pwr[y_np[test_idx] == 0] == 1)
                tau1_p = np.mean(y_test_pred_pwr[y_np[test_idx] == 1] == 0)
                tau0_d = np.mean(y_test_pred_d[y_np[test_idx] == 0] == 1)
                tau1_d = np.mean(y_test_pred_d[y_np[test_idx] == 1] == 0)
                n0 = np.sum(y_np[test_idx] == 0)
                n1 = np.sum(y_np[test_idx] == 1)
                stat_a = (1 - tau0_a - tau1_a) / np.sqrt((tau0_a*(1-tau0_a)/n0) + (tau1_a*(1-tau1_a)/n1))
                stat_p = (1 - tau0_p - tau1_p) / np.sqrt((tau0_p*(1-tau0_p)/n0) + (tau1_p*(1-tau1_p)/n1))
                stat_d = (1 - tau0_d - tau1_d) / np.sqrt((tau0_d*(1-tau0_d)/n0) + (tau1_d*(1-tau1_d)/n1))
                stats_a[name].append(stat_a)
                stats_p[name].append(stat_p)
                stats_d[name].append(stat_d)
                print('classifier: ', name)
                print('stat_a: ', stat_a, 'stat_p: ', stat_p, 'stat_d: ', stat_d)
        stats_a_dict[r] = stats_a 
        stats_p_dict[r] = stats_p 
        stats_d_dict[r] = stats_d 
        
    
    with open(acc_path, 'wb') as f:
        pickle.dump(stats_a_dict, f)

    with open(pwr_path, 'wb') as f:
        pickle.dump(stats_p_dict, f)

    with open(d_path, 'wb') as f:
        pickle.dump(stats_d_dict, f)
    
    
               
                
if __name__ == '__main__':
    
    # Default directory containing the results 
    DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent.joinpath('results')
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-c', '--config',
        help='Path to the config file',
        type=str,
        default = 'config.yml'
    )
    parser.add_argument(
        '-dn', '--data_name',
        help= "The name of dataset",
        type=str,
        required=True
    )
    parser.add_argument(
        '-n', '--exp_number',
        help='The parameter that indexes the experiment',
        type=int,
        required=True)
    
    parser.add_argument(
        '-d', '--dir',
        help='Directory containing the results',
        type=str,
        default = DEFAULT_DATA_DIR)
    
    parser.add_argument(
        '-e', '--n_epochs',
        help='The number of epochs for training NeuralNet',
        type=int,
        default=1000
    )
    
    parser.add_argument(
        '-b', '--batch_size',
        help='Batch size of NeuralNet',
        type=int,
        default=128
    )
    args = parser.parse_args()
    
    # Read config file 
    with open(args.config, 'r') as fd:
        config = yaml.safe_load(fd)
    
    # Create data directory 
    data_dir = Path(args.dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    
    # Run the experiments 
    calc_power(**config, exp_number=args.exp_number, n_epochs=args.n_epochs, batch_size=args.batch_size, device=device, dtype=dtype, data_dir = data_dir, data_name=args.data_name)