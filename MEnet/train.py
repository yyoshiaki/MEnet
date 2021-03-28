import os
import yaml
import sys 

from pickle import dump
from pickle import load

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.impute import SimpleImputer
# from sklearn.model_selection import KFold
# from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math
from torch.nn.modules.loss import _WeightedLoss

import optuna
import joblib

import MEnet

'''
python MEnet_tune_CV.py ../params/210228_optuna_CV.yaml

input : yaml
'''


def train(args):
    dict_input = yaml.load(f)

    f_selected = dict_input['reference']
    f_integrated = dict_input['integrated']
    f_pickle = dict_input['pickle']
    f_ref = dict_input['ref_table']
    dir_output = dict_input['output_dir']
    n_splits = dict_input['n_splits']
    fill = dict_input['fill']
    EPOCHS = dict_input['n_epochs']
    N_TRIALS = dict_input['n_trials']
    patience = dict_input['patience']
    seed = dict_input['seed']
    np.random.seed(seed)

    verbose = True

    os.makedirs(dir_output, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    df_ref = pd.read_csv(f_ref)
    # df_ref.head()

    df_ref_unassigned = df_ref[df_ref.MinorGroup.isna()]
    df_ref_assigned = df_ref[~df_ref.MinorGroup.isna()]


    try:
        df = pd.read_pickle(f_pickle)
        print('pickle loaded')

    except:
        print('no processed pickle file. Now generating inputs...')
        df_selected = \
            pd.read_csv(f_selected, index_col=0)
        df_selected = df_selected.drop_duplicates()
        # df_selected.head()

        df_all = pd.read_csv(f_integrated, index_col = 0)
        # df_all.head()

        df_all.index = df_all['chr'] + ':' + df_all['start'].astype(str) + '-' + df_all['end'].astype(str)
        df = df_all.loc[df_selected.index]
        df_all = None

        df = df[['rate_' + x for x in df_ref_assigned.FileID]]
        df.columns = df_ref_assigned.FileID
        df.to_pickle(f_pickle)

    labels = pd.get_dummies(df_ref_assigned.MinorGroup)
            


    # https://scikit-learn.org/stable/modules/cross_validation.html
    ss = ShuffleSplit(n_splits=n_splits, test_size=1/n_splits, random_state=seed)
    X = np.array(df).T

    def objective(trial):

        n_layers = trial.suggest_int("n_layers", 2, 10)
        hidden_dim = trial.suggest_int("hidden_dim", 100, 3000, 100)
        dropout_rate = trial.suggest_float("dropout_rate", 0, 1)
        activation = trial.suggest_categorical("activation", ["relu", 'tanh', 'leakyrelu'])

        # Generate the optimizers.
        optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
        lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
        
        
        cv = 0
        for fold, (train_index, test_index) in enumerate(ss.split(X)):
    #         print('FOLD : {}'.format(fold))
            x_train = X[train_index]
            y_train = np.array(labels)[train_index]
            x_test = X[test_index]
            y_test = np.array(labels)[test_index]

            if str(fill).isdigit():
                imp = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=fill)
            elif fill in ['median', 'mean', 'most_frequent']:
                imp = SimpleImputer(missing_values=np.nan, strategy='median')

            imp.fit(x_train)

    #         x_train = torch.FloatTensor(x_train).to(device)
    #         y_train = torch.FloatTensor(y_train).to(device)
            # y_train = torch.tensor(y_train, dtype=torch.long).to(device)
            x_test = torch.FloatTensor(imp.transform(x_test)).to(device)
            y_test = torch.FloatTensor(y_test).to(device)
            # y_test = torch.tensor(y_test, dtype=torch.long).to(device)

            data_set = MEnet.Mixup_dataset(x_train, y_train, transform='mix', imputation=imp,
                                        noise=0.01, n_choise=10, dropout=0.4, device=device)
            dataloader = torch.utils.data.DataLoader(data_set, batch_size=100, shuffle=True)

                # Generate the model.
            model = MEnet.MEnet(x_test.shape[1], hidden_dim, dropout_rate, 
                        n_layers, activation, labels.shape[1]).to(device)
            optimizer = getattr(torch.optim, optimizer_name)(model.parameters(), lr=lr)
            criterion = MEnet.OneHotCrossEntropy()
    
            # initialize the early_stopping object
            early_stopping = MEnet.EarlyStopping(patience=patience, verbose=False)
        
            list_loss = []
            list_valloss = []
            list_stopepoch = []
            for e in range(EPOCHS):
                for i, (data, y_target) in enumerate(dataloader):
                    optimizer.zero_grad()
                    y_pred = model(data)
                    loss = criterion(y_pred, y_target)
                    if i == 0:
                        list_loss.append(loss.item())
    #                     writer.add_scalar("Loss/train", loss, e)
                    loss.backward()
                    optimizer.step()

                model.eval()
                with torch.no_grad():
                    y_pred = model(x_test)
                    valloss = criterion(y_pred, y_test) 
                    list_valloss.append(valloss.item())
    #                 writer.add_scalar("Loss/validation", valloss, e)
                
                if (verbose) & (e % 1000 == 0) & (fold == 0):
                    print('Epoch {}: train loss: {} validation loss: {}'.format(e, loss.item(), valloss.item()))
                    
                if fold == 0:
                    trial.report(valloss, e)

                    # Handle pruning based on the intermediate value.
                    if trial.should_prune():
                        raise optuna.exceptions.TrialPruned()
                        
                early_stopping(valloss)
            
                if early_stopping.early_stop:
    #                 print("Early stopping")
                    break
            
            list_stopepoch.append(e)
    #     print(list_stopepoch)
        trial.set_user_attr('best_epoch', int(np.mean(list_stopepoch)-patience))
        cv += min(list_valloss) / n_splits
        return cv


    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=N_TRIALS)

    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial
    trial.params['best_epoch'] = trial.user_attrs['best_epoch']

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
        
    joblib.dump(study, '{}/study.pkl'.format(dir_output))
        
    with open('{}/CV_best_params.yaml'.format(dir_output), 'w') as file:
        yaml.dump(trial.params, file)
              
    fig = optuna.visualization.plot_intermediate_values(study)
    fig.update_yaxes(range=(0,10))
    fig.write_image('{}/CV_int.pdf'.format(dir_output))

    fig = optuna.visualization.plot_parallel_coordinate(study)
    fig.write_image('{}/CV_cood.pdf'.format(dir_output))
    
    fig = optuna.visualization.plot_param_importances(study)
    fig.write_image('{}/CV_importance.pdf'.format(dir_output))
    
    fig = optuna.visualization.plot_optimization_history(study)
    fig.write_image('{}/CV_history.pdf'.format(dir_output))    

if __name__ == "__main__":
    train(sys.argv[1])