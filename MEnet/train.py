import os
import yaml
import sys
import argparse

# from pickle import dump
# from pickle import load
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.impute import SimpleImputer
# from sklearn.model_selection import KFold
# from sklearn.model_selection import train_test_split
# from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import StratifiedShuffleSplit

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math
from torch.nn.modules.loss import _WeightedLoss

import optuna
import joblib

from MEnet import models, utils, _version

'''
python MEnet_tune_CV.py ../params/210228_optuna_CV.yaml

input : yaml
'''


def train(args):
    with open(args.input_yaml, 'r') as f:
        dict_input = yaml.load(f)
    # print(dict_input)

    f_selected = dict_input['reference']
    f_integrated = dict_input['integrated']
    f_pickle = dict_input['pickle']
    f_ref = dict_input['ref_table']
    f_category = dict_input['category']
    dir_output = dict_input['output_dir']
    n_splits = dict_input['n_splits']
    fill = dict_input['fill']
    EPOCHS = dict_input['n_epochs']
    N_TRIALS = dict_input['n_trials']
    patience = dict_input['patience']
    seed = dict_input['seed']
    batch_size = dict_input['batch_size']
    np.random.seed(seed)

    verbose = True

    os.makedirs(dir_output, exist_ok=True)
    os.makedirs(dir_output + '/model_params', exist_ok=True)

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print('device : ', device)
    df_cat = pd.read_csv(f_category)

    df_ref = pd.read_csv(f_ref)
    # df_ref.head()

    df_ref_assigned = df_ref[df_ref.MinorGroup.isin(
        list(df_cat['MinorGroup']))]

    if df_ref_assigned.MinorGroup.value_counts()[
            df_ref_assigned.MinorGroup.value_counts() >= 2].shape[0] > 0:
        print('dropped labels (<2 samples) : ', df_ref_assigned.MinorGroup.value_counts()[
            df_ref_assigned.MinorGroup.value_counts() < 2].index)
    labels = df_ref_assigned.MinorGroup.value_counts(
    )[df_ref_assigned.MinorGroup.value_counts() >= 2].index
    df_ref_assigned = df_ref_assigned[df_ref_assigned.MinorGroup.isin(labels)]
    df_cat = df_cat[df_cat['MinorGroup'].isin(labels)]

    try:
        df = pd.read_pickle(f_pickle)
        print('pickle loaded')

    except:
        print('no processed pickle file. Now generating inputs...')
        df_selected = \
            pd.read_csv(f_selected, index_col=0)
        df_selected = df_selected.drop_duplicates()
        # df_selected.head()

        df_all = pd.read_csv(f_integrated, index_col=0)
        # df_all.head()

        df_all.index = df_all['chr'] + ':' + \
            df_all['start'].astype(str) + '-' + df_all['end'].astype(str)
        df = df_all.loc[df_selected.index]
        df_all = None

        df = df[['rate_' + x for x in df_ref_assigned.FileID]]
        df.columns = df_ref_assigned.FileID
        df.to_pickle(f_pickle)

    labels = pd.get_dummies(df_ref_assigned.MinorGroup)[df_cat.MinorGroup]

    # if df.shape[0] != df_ref_assigned.shape[0]:
    #     raise ValueError('The input file is incompatible form. Try to delete the picke file.')

    # https://scikit-learn.org/stable/modules/cross_validation.html
    # ss = ShuffleSplit(n_splits=n_splits, test_size=1/n_splits, random_state=seed)
    ss = StratifiedShuffleSplit(
        n_splits=n_splits, test_size=1/n_splits, random_state=seed)
    X = np.array(df).T

    def objective(trial):

        n_layers = trial.suggest_int("n_layers", 2, 10)
        hidden_dim = trial.suggest_int("hidden_dim", 100, 3000, 100)
        dropout_rate = trial.suggest_float("dropout_rate", 0, 0.8)
        activation = trial.suggest_categorical(
            "activation", ["relu", 'tanh', 'leakyrelu'])

        # Generate the optimizers.
        optimizer_name = trial.suggest_categorical(
            "optimizer", ["Adam", "RMSprop", "SGD"])
        lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)

        cv = 0
        list_best_models_states = []
        list_best_epoch = []
        list_imp = []
        # for fold, (train_index, test_index) in enumerate(ss.split(X)):
        for fold, (train_index, test_index) in enumerate(ss.split(X, labels)):
            #         print('FOLD : {}'.format(fold))
            x_train = X[train_index]
            y_train = np.array(labels)[train_index]
            x_test = X[test_index]
            y_test = np.array(labels)[test_index]

            if str(fill).isdigit():
                imp = SimpleImputer(missing_values=np.nan,
                                    strategy='constant', fill_value=fill)
            elif fill in ['median', 'mean', 'most_frequent']:
                imp = SimpleImputer(missing_values=np.nan, strategy='median')

            imp.fit(x_train)
            list_imp.append(imp)

            dataset = utils.Mixup_dataset(x_train, y_train, transform='mix', imputation=imp,
                                          noise=0.01, n_choise=10, dropout=0.4)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True,
                                                     num_workers=os.cpu_count(), worker_init_fn=utils.worker_init_fn)

            dataset_test = utils.Mixup_dataset(x_test, y_test, transform='unmix', imputation=imp,
                                               noise=None, dropout=None)
            dataloader_test = torch.utils.data.DataLoader(
                dataset_test, batch_size=batch_size, shuffle=False)

            y_test = torch.FloatTensor(y_test).to(device)

            # Generate the model.
            model = models.MEnet(x_test.shape[1], hidden_dim, dropout_rate,
                                 n_layers, activation, labels.shape[1]).to(device)
            optimizer = getattr(torch.optim, optimizer_name)(
                model.parameters(), lr=lr)
            criterion = utils.OneHotCrossEntropy()

            # initialize the early_stopping object
            early_stopping = utils.EarlyStopping(
                patience=patience, verbose=False)

            list_loss = []
            list_valloss = []
            # list_stopepoch = []
            best_loss = np.inf
            best_state = None
            best_epoch = None
            for e in range(EPOCHS):
                model.train()
                for i, (data, y_target) in enumerate(dataloader):
                    data = data.to(device)
                    y_target = y_target.to(device)

                    optimizer.zero_grad()
                    y_pred = model(data)
                    loss = criterion(y_pred, y_target)
                    if i == 0:
                        list_loss.append(loss.item())
    #                     writer.add_scalar("Loss/train", loss, e)
                    loss.backward()
                    optimizer.step()

                model.eval()
                l_pred_test = []
                with torch.no_grad():
                    for i, (data_test, _) in enumerate(dataloader_test):
                        data_test = data_test.to(device)
                        l_pred_test.append(model(data_test).cpu().numpy())
                y_pred_test = np.concatenate(l_pred_test)
                y_pred_test = torch.FloatTensor(y_pred_test).to(device)
                valloss = criterion(y_pred_test, y_test)
                list_valloss.append(valloss.item())
    #                 writer.add_scalar("Loss/validation", valloss, e)

                if valloss < best_loss:
                    best_state = {k: v.cpu()
                                  for k, v in model.state_dict().items()}
                    best_loss = valloss
                    best_epoch = e

                # if (verbose) & (e % 1000 == 0) & (fold == 0):
                #     print('Epoch {}: train loss: {} validation loss: {}'.format(e, loss.item(), valloss.item()))

                if fold == 0:
                    trial.report(valloss, e)

                    # Handle pruning based on the intermediate value.
                    if trial.should_prune():
                        raise optuna.exceptions.TrialPruned()

                early_stopping(valloss)

                if early_stopping.early_stop:
                    #                 print("Early stopping")
                    break

            # list_stopepoch.append(e)
    #     print(list_stopepoch)

            cv += min(list_valloss) / n_splits

            list_best_models_states.append(best_state)
            list_best_epoch.append(best_epoch)
            if verbose:
                print('Best Epoch : ', best_epoch, 'Best Loss : ', best_loss)

        trial.set_user_attr('best_epoch', list_best_epoch)

        with open("{d}/model_params/{n}.pickle".format(d=dir_output, n=trial.number), "wb") as f_out:
            pickle.dump([list_imp, list_best_models_states], f_out)

        return cv

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=N_TRIALS)

    pruned_trials = [t for t in study.trials if t.state ==
                     optuna.trial.TrialState.PRUNED]
    complete_trials = [t for t in study.trials if t.state ==
                       optuna.trial.TrialState.COMPLETE]

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial
    trial.params['best_epoch'] = trial.user_attrs['best_epoch']

    print("  Best CV: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    joblib.dump(study, '{}/study.pkl'.format(dir_output))

    with open('{}/CV_best_params.yaml'.format(dir_output), 'w') as file:
        yaml.dump(trial.params, file)

    with open("{d}/model_params/{n}.pickle".format(d=dir_output, n=study.best_trial.number), "rb") as fin:
        list_imp, list_best_models_states = pickle.load(fin)

    model_params = [[X.shape[1], trial.params['hidden_dim'],
                     trial.params['dropout_rate'], trial.params['n_layers'],
                     trial.params['activation'], labels.shape[1]],
                    list_best_models_states,
                    list(df.index), list(labels.columns), list_imp, df_cat]

    with open("{d}/best_model.pickle".format(d=dir_output), mode='wb') as f:
        pickle.dump(model_params, f)

    fig = optuna.visualization.plot_intermediate_values(study)
    fig.update_yaxes(range=(0, 10))
    fig.write_image('{}/CV_int.pdf'.format(dir_output))

    fig = optuna.visualization.plot_parallel_coordinate(study)
    fig.write_image('{}/CV_cood.pdf'.format(dir_output))

    fig = optuna.visualization.plot_param_importances(study)
    fig.write_image('{}/CV_importance.pdf'.format(dir_output))

    fig = optuna.visualization.plot_optimization_history(study)
    fig.write_image('{}/CV_history.pdf'.format(dir_output))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.input_yaml = 'test/train/210228_optuna_CV.yaml'
    args.device = 'cpu'
    train(args)
