import argparse
from pathlib import Path 
import pickle 
import re

from matplotlib.font_manager import FontProperties
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm 
import pandas as pd
import seaborn as sns

font = {'size' : 15}

plt.rc('font', **font)
plt.rc('lines', linewidth=2)
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

fontP = FontProperties()
fontP.set_size('xx-small')

style = {'kNN' : '-.', 'Random Forest' : '--', 'LDA' : '-', 'Logistic Regression': '-',
         'XGBoost' : '--', 'Neural Network' : ':'}
labels = {'kNN' : 'kNN', 'Random Forest' : 'RF', 'LDA' : 'LDA', 'Logistic Regression' : 'LR',
          'XGBoost' : 'XGB', 'Neural Network' : 'NN'}
markers = {'kNN': 'x', 'Random Forest' : 'D', 'LDA' : '', 'Logistic Regression' : '*',
           'XGBoost' : '^', 'Neural Network' : 'o'}
ir_list = [1, 5, 9, 13, 17, 21]
classifiers= [
            'kNN',
            'Random Forest',
            'LDA',
            'Logistic Regression',
            'XGBoost',
            'NN'
]

for i in range(4):
    with open('/home/oldrain123/results/acc_results_{}.data'.format(i+1), 'rb') as f:
        data =pickle.load(f)
        
    data_a = []
    for r in ir_list:
        for name in classifiers:
            pwr = np.mean(data[r][name] >= (1-norm.ppf(0.05)))
            data_a.append({'imbalance_ratio': r, 'classifier': name, 'power': pwr})
    df_a = pd.DataFrame(data_a)
    # Create the plot
    sns.set_style("whitegrid")
    sns.lineplot(data=df_a, x="imbalance_ratio", y="power", hue="classifier")

    # Add axis labels and a title
    plt.xlabel("Imbalance Ratio")
    plt.ylabel("Power")
    plt.title("Maximizing Accuracy")

    # Show the plot
    plt.savefig('/home/oldrain123/imbalanced_c2st/Results/acc_results_{}.png'.format(i+1), box_inches='tight')
    plt.clf()  # clear the current plot
    plt.close()  # close the current figure

    with open('/home/oldrain123/results/pwr_results_{}.data'.format(i+1), 'rb') as f:
        data =pickle.load(f)
        
    data_p = []
    for r in ir_list:
        for name in classifiers:
            pwr = np.mean(data[r][name] >= (1-norm.ppf(0.05)))
            data_p.append({'imbalance_ratio': r, 'classifier': name, 'power': pwr})
    df_p = pd.DataFrame(data_p)
    # Create the plot
    sns.set_style("whitegrid")
    sns.lineplot(data=df_p, x="imbalance_ratio", y="power", hue="classifier")

    # Add axis labels and a title
    plt.xlabel("Imbalance Ratio")
    plt.ylabel("Power")
    plt.title("Maximizing Power")

    # Show the plot
    plt.savefig('/home/oldrain123/imbalanced_c2st/Results/pwr_results_{}.png'.format(i+1), box_inches='tight')
    plt.clf()  # clear the current plot
    plt.close()  # close the current figure
    
    with open('/home/oldrain123/results/d_results_{}.data'.format(i+1), 'rb') as f:
        data =pickle.load(f)
        
    data_d = []
    for r in ir_list:
        for name in classifiers:
            pwr = np.mean(data[r][name] >= (1-norm.ppf(0.05)))
            data_d.append({'imbalance_ratio': r, 'classifier': name, 'power': pwr})
    df_d = pd.DataFrame(data_d)
    # Create the plot
    sns.set_style("whitegrid")
    sns.lineplot(data=df_d, x="imbalance_ratio", y="power", hue="classifier")

    # Add axis labels and a title
    plt.xlabel("Imbalance Ratio")
    plt.ylabel("Power")
    plt.title("Default")

    # Show the plot
    plt.savefig('/home/oldrain123/imbalanced_c2st/Results/d_results_{}.png'.format(i+1), box_inches='tight')
    plt.clf()  # clear the current plot
    plt.close()  # close the current figure