import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.models.iforest import IForest
from pyod.models.ocsvm import OCSVM
from pyod.models.deep_svdd import DeepSVDD
from pyod.models.auto_encoder import AutoEncoder
from pyod.models.lunar import LUNAR
from pyod.models.ecod import ECOD
from pyod.models.inne import INNE
from pyod.models.dif import DIF
from pyod.models.copod import COPOD
from TextCore import TextCore
import time
from tqdm import tqdm
import warnings
import json
import gc

warnings.filterwarnings('ignore')

def evaluate_model(clf, X_train, X_test, y_test, sentence_info=None):
    """
    Evaluate anomaly detection model with both ROC AUC and PR AUC metrics
    Now supports both token-level and sentence-level evaluation
    
    Args:
        clf: anomaly detector
        X_train: training data
        X_test: test data (token-level)
        y_test: test labels (token-level)
        sentence_info: dict with 'lengths' (list of token counts per sentence) 
                       and 'labels' (sentence-level labels), optional
    
    Returns:
        If sentence_info is None: (roc_auc, pr_auc)
        If sentence_info is provided: (token_roc_auc, token_pr_auc, sent_roc_auc, sent_pr_auc)
    """
    # Train model
    clf.fit(X_train)
    
    # Predict anomaly scores (token-level)
    scores = clf.decision_function(X_test)
    
    # Calculate token-level metrics
    token_roc_auc = roc_auc_score(y_test, scores)
    precision, recall, _ = precision_recall_curve(y_test, scores)
    token_pr_auc = auc(recall, precision)
    
    # If sentence_info is provided, calculate sentence-level metrics
    if sentence_info is not None:
        sentence_lengths = sentence_info['lengths']
        sentence_labels = sentence_info['labels']
        
        # Aggregate token scores to sentence scores
        sentence_scores = []
        start = 0
        for length in sentence_lengths:
            end = start + length
            # Use max aggregation (you can change to mean, median, etc.)
            sentence_scores.append(np.max(scores[start:end]))
            start = end
        
        sentence_scores = np.array(sentence_scores)
        
        # Calculate sentence-level metrics
        sent_roc_auc = roc_auc_score(sentence_labels, sentence_scores)
        precision, recall, _ = precision_recall_curve(sentence_labels, sentence_scores)
        sent_pr_auc = auc(recall, precision)
        
        return token_roc_auc, token_pr_auc, sent_roc_auc, sent_pr_auc
    
    return token_roc_auc, token_pr_auc

def run_algorithm_with_param_search(X_train, X_test, y_test, detector_class, param_grid, 
                                     n_runs=5, result_file=None, name="Algorithm", sentence_info=None):
    """
    Run algorithm with parameter search, multiple times per parameter set
    
    Args:
        sentence_info: dict with 'lengths' and 'labels' for sentence-level evaluation
    """
    use_sentence_eval = sentence_info is not None
    
    # Handle case with no parameters to search
    if not param_grid:
        clf = detector_class()
        token_roc_scores = []
        token_pr_scores = []
        sent_roc_scores = [] if use_sentence_eval else None
        sent_pr_scores = [] if use_sentence_eval else None
        
        for _ in range(1):
            if use_sentence_eval:
                token_roc, token_pr, sent_roc, sent_pr = evaluate_model(
                    clf, X_train, X_test, y_test, sentence_info)
                token_roc_scores.append(token_roc)
                token_pr_scores.append(token_pr)
                sent_roc_scores.append(sent_roc)
                sent_pr_scores.append(sent_pr)
            else:
                token_roc, token_pr = evaluate_model(clf, X_train, X_test, y_test)
                token_roc_scores.append(token_roc)
                token_pr_scores.append(token_pr)
        
        token_roc_mean = np.mean(token_roc_scores)
        token_roc_std = np.std(token_roc_scores)
        token_pr_mean = np.mean(token_pr_scores)
        token_pr_std = np.std(token_pr_scores)
        
        if result_file:
            with open(result_file, 'a') as f:
                f.write("*" * 70 + "\n")
                f.write(f"{name} - Using default parameters\n")
                f.write(f"Token-level: ROC AUC: {token_roc_mean:.4f} ± {token_roc_std:.4f}, "
                       f"PR AUC: {token_pr_mean:.4f} ± {token_pr_std:.4f}\n")
                
                if use_sentence_eval:
                    sent_roc_mean = np.mean(sent_roc_scores)
                    sent_roc_std = np.std(sent_roc_scores)
                    sent_pr_mean = np.mean(sent_pr_scores)
                    sent_pr_std = np.std(sent_pr_scores)
                    f.write(f"Sentence-level: ROC AUC: {sent_roc_mean:.4f} ± {sent_roc_std:.4f}, "
                           f"PR AUC: {sent_pr_mean:.4f} ± {sent_pr_std:.4f}\n")
                f.write("\n")
        
        if use_sentence_eval:
            return {}, token_roc_mean, token_roc_std, token_pr_mean, token_pr_std, \
                   sent_roc_mean, sent_roc_std, sent_pr_mean, sent_pr_std
        return {}, token_roc_mean, token_roc_std, token_pr_mean, token_pr_std
    
    # Parameter search
    grid = ParameterGrid(param_grid)
    results = []
    
    if result_file:
        with open(result_file, 'a') as f:
            f.write("*" * 70 + "\n")
            f.write(f"{name} - Parameter search\n")
            f.write(f"Grid search parameters: {param_grid}\n")
            if use_sentence_eval:
                f.write(f"{'Parameters':<40} {'Token ROC AUC':<18} {'Token PR AUC':<18} "
                       f"{'Sent ROC AUC':<18} {'Sent PR AUC':<18}\n")
                f.write("-" * 112 + "\n")
            else:
                f.write(f"{'Parameters':<40} {'ROC AUC':<15} {'PR AUC':<15}\n")
                f.write("-" * 70 + "\n")
    
    for params in tqdm(grid, desc=f"{name}"):
        token_roc_scores = []
        token_pr_scores = []
        sent_roc_scores = [] if use_sentence_eval else None
        sent_pr_scores = [] if use_sentence_eval else None
        
        for _ in range(n_runs):
            clf = detector_class(**params)
            
            if use_sentence_eval:
                token_roc, token_pr, sent_roc, sent_pr = evaluate_model(
                    clf, X_train, X_test, y_test, sentence_info)
                token_roc_scores.append(token_roc)
                token_pr_scores.append(token_pr)
                sent_roc_scores.append(sent_roc)
                sent_pr_scores.append(sent_pr)
            else:
                token_roc, token_pr = evaluate_model(clf, X_train, X_test, y_test)
                token_roc_scores.append(token_roc)
                token_pr_scores.append(token_pr)

        del clf
        gc.collect()

        token_roc_mean = np.mean(token_roc_scores)
        token_roc_std = np.std(token_roc_scores)
        token_pr_mean = np.mean(token_pr_scores)
        token_pr_std = np.std(token_pr_scores)
        
        if use_sentence_eval:
            sent_roc_mean = np.mean(sent_roc_scores)
            sent_roc_std = np.std(sent_roc_scores)
            sent_pr_mean = np.mean(sent_pr_scores)
            sent_pr_std = np.std(sent_pr_scores)
        
        if result_file:
            with open(result_file, 'a') as f:
                if use_sentence_eval:
                    f.write(f"{str(params):<40} "
                           f"{token_roc_mean:.4f}±{token_roc_std:.4f}    "
                           f"{token_pr_mean:.4f}±{token_pr_std:.4f}    "
                           f"{sent_roc_mean:.4f}±{sent_roc_std:.4f}    "
                           f"{sent_pr_mean:.4f}±{sent_pr_std:.4f}\n")
                else:
                    f.write(f"{str(params):<40} {token_roc_mean:.4f} ± {token_roc_std:.4f} "
                           f"{token_pr_mean:.4f} ± {token_pr_std:.4f}\n")
        
        if use_sentence_eval:
            results.append((params, token_roc_mean, token_roc_std, token_pr_mean, token_pr_std,
                          sent_roc_mean, sent_roc_std, sent_pr_mean, sent_pr_std))
        else:
            results.append((params, token_roc_mean, token_roc_std, token_pr_mean, token_pr_std))
    
    # Select best parameters (based on token-level ROC AUC)
    best_result = max(results, key=lambda x: x[1])
    
    if result_file:
        with open(result_file, 'a') as f:
            f.write("\nBest parameters (selected by token ROC AUC):\n")
            f.write(f"{str(best_result[0])}\n")
            if use_sentence_eval:
                f.write(f"Token-level: ROC AUC: {best_result[1]:.4f} ± {best_result[2]:.4f}, "
                       f"PR AUC: {best_result[3]:.4f} ± {best_result[4]:.4f}\n")
                f.write(f"Sentence-level: ROC AUC: {best_result[5]:.4f} ± {best_result[6]:.4f}, "
                       f"PR AUC: {best_result[7]:.4f} ± {best_result[8]:.4f}\n\n")
            else:
                f.write(f"Best ROC AUC: {best_result[1]:.4f} ± {best_result[2]:.4f}, "
                       f"PR AUC: {best_result[3]:.4f} ± {best_result[4]:.4f}\n\n")
    
    return best_result

def run_anomaly_detection_benchmark(X_train, X_test, y_test, sentence_info=None, 
                                     result_file="anomaly_detection_results.txt"):
    """
    Run anomaly detection benchmark with multiple algorithms
    
    Args:
        X_train: training data (token-level)
        X_test: test data (token-level)
        y_test: test labels (token-level)
        sentence_info: dict with 'lengths' (list of token counts per sentence) 
                       and 'labels' (sentence-level labels), optional
        result_file: output file path
    """
    use_sentence_eval = sentence_info is not None
    
    # Create result file
    with open(result_file, 'w') as f:
        f.write("Anomaly Detection Benchmark Results\n")
        f.write("=" * 80 + "\n")
        f.write(f"Dataset: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples (token-level)\n")
        f.write(f"Features: {X_train.shape[1]} binary features\n")
        f.write(f"Anomaly ratio in test set (token-level): {np.mean(y_test):.4f}\n")
        
        if use_sentence_eval:
            f.write(f"Number of sentences: {len(sentence_info['lengths'])}\n")
            f.write(f"Anomaly ratio in test set (sentence-level): {np.mean(sentence_info['labels']):.4f}\n")
        
        f.write("=" * 80 + "\n\n")
    
    # Define detectors with parameters to search
    n_features = X_train.shape[1]
    
    detectors = {
        'LOF': {
            'class': LOF,
            'params': {
                'n_neighbors': [5, 10, 20, 40],
            }
        },
        'IForest': {
            'class': IForest,
            'params': {
                'max_samples': [16, 32, 64, 128, 256, 512],
                'n_estimators': [200]
            }
        },
        'ECOD': {
            'class': ECOD,
            'params': {}
        },
        'DeepSVDD': {
            'class': DeepSVDD,
            'params': {
                'hidden_neurons': [[128, 64], [64, 32], [32, 16]],
                'n_features': [n_features]
            }
        },
        'AutoEncoder': {
            'class': AutoEncoder,
            'params': {
                'hidden_neuron_list': [[128, 64], [64, 32], [32, 16]],
                'device':['cuda']
            }
        },
        'LUNAR': {
            'class': LUNAR,
            'params': {
                'n_neighbours': [5, 10, 20, 40],
                'device':['cuda']
            }
        },
            'TextCore':{
        'class': TextCore,
        'params': {
        'n_neighbors':[1]
        }
        }
        # 'DIF': {
        #     'class': DIF,
        #     'params': {
        #         'hidden_neurons': [[128, 64], [64, 32], [32, 16]],
        #     }
        # }
    }
    
    # Summary table header
    with open(result_file, 'a') as f:
        if use_sentence_eval:
            f.write(f"{'Algorithm':<12} {'Token ROC AUC':<18} {'Token PR AUC':<18} "
                   f"{'Sent ROC AUC':<18} {'Sent PR AUC':<18} {'Time':<10}\n")
            f.write("-" * 94 + "\n")
        else:
            f.write(f"{'Algorithm':<12} {'ROC AUC':<15} {'PR AUC':<15} {'Time':<10}\n")
            f.write("-" * 55 + "\n")
    
    results = {}
    
    for name, config in detectors.items():
        print(f"\nEvaluating {name}...")
        start_time = time.time()
        
        best_result = run_algorithm_with_param_search(
            X_train, X_test, y_test, 
            config['class'], 
            config['params'],
            n_runs=5,
            result_file=result_file,
            name=name,
            sentence_info=sentence_info
        )
        
        end_time = time.time()
        runtime = end_time - start_time
        
        if use_sentence_eval:
            best_params, token_roc_mean, token_roc_std, token_pr_mean, token_pr_std, \
                sent_roc_mean, sent_roc_std, sent_pr_mean, sent_pr_std = best_result
            
            results[name] = {
                'best_params': best_params,
                'token_roc_auc': f"{token_roc_mean:.4f} ± {token_roc_std:.4f}",
                'token_pr_auc': f"{token_pr_mean:.4f} ± {token_pr_std:.4f}",
                'sent_roc_auc': f"{sent_roc_mean:.4f} ± {sent_roc_std:.4f}",
                'sent_pr_auc': f"{sent_pr_mean:.4f} ± {sent_pr_std:.4f}",
                'time': f"{runtime:.2f}s"
            }
            
            print(f"{name} Token ROC AUC: {token_roc_mean:.4f} ± {token_roc_std:.4f}")
            print(f"{name} Token PR AUC: {token_pr_mean:.4f} ± {token_pr_std:.4f}")
            print(f"{name} Sentence ROC AUC: {sent_roc_mean:.4f} ± {sent_roc_std:.4f}")
            print(f"{name} Sentence PR AUC: {sent_pr_mean:.4f} ± {sent_pr_std:.4f}")
            print(f"{name} runtime: {runtime:.2f}s")
            
            with open(result_file, 'a') as f:
                f.write(f"{name:<12} {results[name]['token_roc_auc']:<18} "
                       f"{results[name]['token_pr_auc']:<18} "
                       f"{results[name]['sent_roc_auc']:<18} "
                       f"{results[name]['sent_pr_auc']:<18} "
                       f"{results[name]['time']:<10}\n")
        else:
            best_params, token_roc_mean, token_roc_std, token_pr_mean, token_pr_std = best_result
            
            results[name] = {
                'best_params': best_params,
                'roc_auc': f"{token_roc_mean:.4f} ± {token_roc_std:.4f}",
                'pr_auc': f"{token_pr_mean:.4f} ± {token_pr_std:.4f}",
                'time': f"{runtime:.2f}s"
            }
            
            print(f"{name} ROC AUC: {token_roc_mean:.4f} ± {token_roc_std:.4f}")
            print(f"{name} PR AUC: {token_pr_mean:.4f} ± {token_pr_std:.4f}")
            print(f"{name} runtime: {runtime:.2f}s")
            
            with open(result_file, 'a') as f:
                f.write(f"{name:<12} {results[name]['roc_auc']:<15} "
                       f"{results[name]['pr_auc']:<15} {results[name]['time']:<10}\n")

    # Final summary
    with open(result_file, 'a') as f:
        f.write("\n\nFinal Results Summary:\n")
        f.write("=" * 80 + "\n")
        for name, result in results.items():
            f.write(f"{name}:\n")
            f.write(f"  Best parameters: {result['best_params']}\n")
            if use_sentence_eval:
                f.write(f"  Token ROC AUC: {result['token_roc_auc']}\n")
                f.write(f"  Token PR AUC: {result['token_pr_auc']}\n")
                f.write(f"  Sentence ROC AUC: {result['sent_roc_auc']}\n")
                f.write(f"  Sentence PR AUC: {result['sent_pr_auc']}\n")
            else:
                f.write(f"  ROC AUC: {result['roc_auc']}\n")
                f.write(f"  PR AUC: {result['pr_auc']}\n")
            f.write(f"  Runtime: {result['time']}\n\n")
    
    print(f"\nResults saved to {result_file}")
    return results


def split_anomaly_data(X, y, z, train_ratio=0.5, random_state=42):
    y = np.array(y)
    n_train = int(len(X) * train_ratio)
    
    normal_indices = np.where(y == 0)[0]
    
    np.random.seed(random_state)
    train_indices = np.random.choice(normal_indices, size=n_train, replace=False)
    
    test_indices = np.setdiff1d(np.arange(len(X)), train_indices)
    
    X_train = [X[i] for i in train_indices]
    y_train = y[train_indices]
    X_test = [X[i] for i in test_indices]
    y_test = y[test_indices]
    z_train = [z[i] for i in train_indices]
    z_test = [z[i] for i in test_indices]
    return X_train, y_train, X_test, y_test, z_train, z_test