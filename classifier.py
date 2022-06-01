import numpy as np
import pandas as pd
import json
import argparse

import catboost

from catboost import CatBoostClassifier, Pool, metrics, cv
from catboost.utils import get_roc_curve, get_confusion_matrix, eval_metric
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

def build_model(**kwargs):
    model = CatBoostClassifier(
        custom_loss=[metrics.Accuracy()], 
        **kwargs
    )
    return model

def plot_roc_curves(model_list, X, Y, labels, cat_features_indices):
    k = len(model_list)
    
    assert(k <= 3)
        
    color_list = ['blue', 'green', 'red']
    plt.title('ROC for models')
    
    for m, color, label in zip(model_list, color_list[:k], labels[:k]):
        fpr, tpr, _ = get_roc_curve(m, Pool(X, Y, cat_features=cat_features_indices))
        plt.plot(fpr, tpr, color=color, label=label, linewidth=0.5)

    plt.legend()
    plt.grid(True, linewidth=0.75)
    plt.savefig('roc_curve_result.png', dpi=150)

def stats(model_list, X_test, Y_test, cat_features_indices):
    auc_scores = []
    print('Models info:')
    
    for k in range(0, len(model_list)):
        m = model_list[k]
        
        pr_prob = m.predict_proba(X_test)
        pr = m.predict(X_test)
        ans = Y_test.to_numpy()
                
        check = [True if i==j else False for i, j in zip(ans, pr)]
        cm = get_confusion_matrix(m, Pool(X_test, Y_test, cat_features=cat_features_indices))
        auc_scores.append(eval_metric(ans, pr_prob, 'AUC')[0])
        
        print(f'\nModel {k} confusion_matrix:\n', cm)
        print('AUC:', auc_scores[k])
        print(f'Correct predictions: {check.count(True)}/{Y_test.shape[0]}\n')
        print(m.get_feature_importance(prettified=True))
        
    return auc_scores
        
def cv_models(model_list, X, Y, cat_features_indices):
    print('CV:')
        
    for k in range(0, len(model_list)):
        m = model_list[k]
        
        cv_params = m.get_params()    
        cv_params.update(
            {'loss_function': metrics.Logloss()
        })
    
        cv_data = cv(Pool(X, Y, cat_features = cat_features_indices), cv_params, logging_level='Silent')
    
        best_step = np.argmax(cv_data['test-Accuracy-mean'])
    
        print('- Mean: ', cv_data['test-Accuracy-mean'][best_step])
        print('- Std: ', cv_data['test-Accuracy-std'][best_step])
        print('- Best step: ', best_step)    

def split_dataset(df):
    seed = 123    
    Y = df['class']
    X = df.drop(['class','score'], axis=1)
   
    X_train, X_, Y_train, Y_ = train_test_split(X, Y, train_size=0.8, random_state=seed)
    X_val, X_test, Y_val, Y_test = train_test_split(X_, Y_, train_size=0.5, random_state=seed)

    return X_train, X_val, X_test, Y_train, Y_val, Y_test  

def build_dataset(dataset_filename):
    df = pd.read_csv(dataset_filename, index_col=0)
    threshold_score = 50   
    df['class'] = df['score']//threshold_score
    return df
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_file', type=str, required=True)
    parser.add_argument('--model_params_file', type=str, required=True)
    
    args = parser.parse_args()
    
    dataset_filename = args.dataset_file
    models_params_filename = args.model_params_file
    
    df = build_dataset(dataset_filename)

    X_train, X_val, X_test, Y_train, Y_val, Y_test = split_dataset(df)
    cat_features_indices = np.where(X_test.dtypes != float)[0]
    
    labels, params, models = [], {}, []
    
    with open(models_params_filename, 'r') as f:
        params = json.load(f)
        labels = list(params.keys())
                
    for label in labels:
        models.append(build_model(**params[label]))

    for m in models:
        m.fit(X_train, Y_train, eval_set=(X_val, Y_val), cat_features=cat_features_indices)  
    
    auc_scores = stats(models, X_test, Y_test, cat_features_indices)
    
    for i in range(0, len(labels)):
        labels[i] = labels[i] + f': AUC={auc_scores[i]:.3f}'    
    plot_roc_curves(models, X_test, Y_test, labels, cat_features_indices)
    cv_models(models, X_train, Y_train, cat_features_indices)
    
