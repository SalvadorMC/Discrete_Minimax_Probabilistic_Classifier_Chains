"""
script_experiments.py
---------------------
Different functions to execute cross-fold validation and run experiments:
1) `run_all_experiments`: Loop over all datasets and models, executing training, prediction, and metrics evaluation.
2) `run_fold_cv`: Performs cross-validation for a given model and dataset, using multiple prediction strategies (Subset, Hamming, F1).
3) `experiment`: Executes cross-validation for a model over multiple seeds, collecting metrics.

"""



import warnings
import numpy as np
import pandas as pd
from skmultilearn.model_selection import IterativeStratification
warnings.filterwarnings('ignore')
import os
from tqdm import tqdm
from joblib import Parallel, delayed

def run_fold_cv(model, X, y, random_seed, metric_functions, k):
    """
    Executes cross-validation for a specific model and dataset.

    inputs:
        model (object): The machine learning model to be trained.
        X (np.ndarray): Features of the dataset.
        y (np.ndarray): Labels of the dataset.
        random_seed (int): Seed for reproducibility.
        metric_functions (list): List of metric functions to evaluate the model.
        k (int): Number of cross-validation folds.

    Returns:
        list: List of dictionaries containing the metrics for each fold and prediction type.
    """
    kfold = IterativeStratification(n_splits=k, order=1, random_state=random_seed)

    import copy

    def run_single_fold(train_idx, test_idx):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        clf = copy.deepcopy(model)
        clf.fit(X_train, y_train)
        general_results = []

        if hasattr(clf, "predict_subset"):
            # Subset
            y_pred = clf.predict_subset(X_test)
            fold_result = {}
            for metric in metric_functions:
                score = metric["func"](y_test, y_pred)
                fold_result[metric["name"]] = score
            general_results.append(fold_result)

            # Hamming
            y_pred = clf.predict_hamming(X_test)
            fold_result = {}
            for metric in metric_functions:
                score = metric["func"](y_test, y_pred)
                fold_result[metric["name"]] = score
            general_results.append(fold_result)

            # F1
            y_pred = clf.predict_f1(X_test)
            fold_result = {}
            for metric in metric_functions:
                score = metric["func"](y_test, y_pred)
                fold_result[metric["name"]] = score
            general_results.append(fold_result)

        else:
            # Fallback a predict estándar: tres copias del mismo resultado
            y_pred = clf.predict(X_test)
            fold_result = {}
            for metric in metric_functions:
                score = metric["func"](y_test, y_pred)
                fold_result[metric["name"]] = score
            general_results.append(fold_result.copy())
            general_results.append(fold_result.copy())
            general_results.append(fold_result.copy())
        return general_results


    results = Parallel(n_jobs=1)(
        delayed(run_single_fold)(train_idx, test_idx)
        for train_idx, test_idx in kfold.split(X, y)
    )

    # Agrupar resultados por métrica y por tipo de predicción
    metric_results_by_prediction = []
    for pred_type in range(3):  # Subset, Hamming, F1
        metric_results = {metric["name"]: [] for metric in metric_functions}
        for fold_result in results:
            for metric_name, score in fold_result[pred_type].items():
                metric_results[metric_name].append(score)
        metric_results_by_prediction.append(metric_results)

    return metric_results_by_prediction


def experiment(model, X, y, metric_functions, seeds, k):
    """
    Executes cross-validation for a given model on multiple seeds.

    Input:
        model (object): The model to be trained and evaluated.
        X (np.ndarray): Features of the dataset.
        y (np.ndarray): Labels of the dataset.
        metric_functions (list): List of metric functions.
        seeds (list): List of seeds for reproducibility.
        k (int): Number of folds for cross-validation.

    Returns:
        list: Summary of metrics for each prediction type (Subset, Hamming, F1).
    """
    all_metrics_by_prediction = [
        {metric["name"]: [] for metric in metric_functions}
        for _ in range(3)  # Subset, Hamming, F1
    ]

    for seed in seeds:
        fold_scores = run_fold_cv(model, X, y, seed, metric_functions, k)
        for i in range(3):
            for metric_name, scores in fold_scores[i].items():
                all_metrics_by_prediction[i][metric_name].extend(scores)

    summaries = []
    for metric_scores in all_metrics_by_prediction:
        summary = {}
        for metric_name, values in metric_scores.items():
            summary[f"{metric_name} Mean"] = np.mean(values)
            summary[f"{metric_name} Std"] = np.std(values)
        summaries.append(summary)

    return summaries


def run_all_experiments(models, datasets, metric_functions, seeds, k):
    """
    Main loop to run all experiments over multiple datasets and models.

    Input:
        models (dict): Dictionary of model names and instances.
        datasets (dict): Dictionary of dataset names and (X, y) tuples.
        metric_functions (list): List of metric functions.
        seeds (list): List of seeds for reproducibility.
        k (int): Number of folds for cross-validation.
    """
    for dataset_name, (X, y) in datasets.items():
        dataset_results = []
        bandera = True
        # Normalizar
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        X = (X - mean) / std

        for model_name, model in tqdm(models.items(), desc=f"Models on {dataset_name}", leave=False):
            try:
                scores_list = experiment(model, X, y, metric_functions, seeds, k)

                scores_list[0]["model"] = model_name + "-Subset"
                dataset_results.append(scores_list[0])
                scores_list[1]["model"] = model_name + "-Hamming"
                dataset_results.append(scores_list[1])
                scores_list[2]["model"] = model_name + "-F1"
                dataset_results.append(scores_list[2])

            except Exception as e:
                print(f"❌ Error with model {model_name} on dataset {dataset_name}: {e}")
                bandera = False

        df = pd.DataFrame(dataset_results)

        output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results'))
        output_path = os.path.join(output_dir, f"{dataset_name}_metrics.csv")
        df.to_csv(output_path, index=False, float_format="%.4f")

        if bandera:
            print(f"✅✅✅ Saved {dataset_name}_metrics.csv")
        else:
            print(f"✅❌ Saved with some models missing: {dataset_name}_metrics.csv")
