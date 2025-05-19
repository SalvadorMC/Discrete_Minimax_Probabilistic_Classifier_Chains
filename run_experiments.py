"""
run_experiments.py
-------------------
Script to run the complete experiment pipeline

"""

from sklearn.tree import DecisionTreeClassifier
from src.scripts_metrics_loadData import load_data
from src.script_experiments import run_all_experiments
from src.scripts_metrics_loadData import EvaluationMetrics
from sklearn.linear_model import LogisticRegression
from lib.DMC import DBC, DiscreteMinimaxClassifier, DBC_KmeansU, DMC_KmeansU, DMC_Tree, DMC_logistic, DBC_logistic, \
    BinaryRelevance

datasets = {
    "CHD_49aaaa": load_data(dataset_name="CHD_49.arff"),
    #"Emotions": load_data(dataset_name="Emotions.arff"),
    #"GnegativePseAAC": load_data(dataset_name="GnegativePseAAC.arff"),
    #"GpositivePseAAC": load_data(dataset_name="GpositivePseAAC.arff"),
    #"Scene": load_data(dataset_name="Scene.arff"),
    #"VirusPseAAC": load_data(dataset_name="VirusPseAAC.arff"),
    #"Image": load_data(dataset_name="Image.arff"),
    #"Flags": load_data(dataset_name="Flags.arff"),
    #"Foodtruck": load_data(dataset_name="Foodtruck.arff"),
    #"PlantPseAAC": load_data(dataset_name="PlantPseAAC.arff"),sssddds
}

# Models
clf_kmeans = DBC()
clf_DMC_kmeans = DiscreteMinimaxClassifier(N=10000)

clf_kmeansU = DBC_KmeansU()
clf_DMC_KmeansU = DMC_KmeansU()

DT = DecisionTreeClassifier()
WDT = DecisionTreeClassifier(class_weight="balanced")
clf_DMC_DT = DMC_Tree(weighted=False)

LR = LogisticRegression(max_iter=10000)
WLR = LogisticRegression(class_weight='balanced',max_iter=10000)
clf_lr = DBC_logistic()
clf_DMC_LR = DMC_logistic(weighted=False)
models = {
    #BR DMC
    #"BR_DMC_Original": BinaryRelevance(classifier=clf_DMC_kmeans)

    # Kmeans
    #"Kmeans": ProbabilisticClassifierChainBase(base_estimator=clf_kmeans),
    #"Kmeans_ROS": MLROS_ProbabilisticClassifierChainBase(base_estimator=clf_kmeans),
    #"DMC_Kmeans": ProbabilisticClassifierChainBase(base_estimator=clf_DMC_kmeans),

    # Kmeans Union
    #"KmeansU": ProbabilisticClassifierChainBase(base_estimator=DBC_KmeansU()),
    #"KmeansU_ROS": MLROS_ProbabilisticClassifierChainBase(base_estimator=DBC_KmeansU()),
    #"DMC_Union": ProbabilisticClassifierChainBase(base_estimator=clf_DMC_KmeansU),

    # DT
    #"DT": ProbabilisticClassifierChainBase(base_estimator=DT),
    #"WDT": ProbabilisticClassifierChainBase(base_estimator=WDT),
    #"DT_ROS": MLROS_ProbabilisticClassifierChainBase(base_estimator=DT),
    #"DMC_DT2": ProbabilisticClassifierChainBase(base_estimator=clf_DMC_DT),
    #"BR_DT": BinaryRelevance(classifier=DT),

    # LR
    #"LR": ProbabilisticClassifierChainBase(base_estimator=LR),
    #"WLR": ProbabilisticClassifierChainBase(base_estimator=WLR),
    #"LR_ROS": MLROS_ProbabilisticClassifierChainBase(base_estimator=LR),
    #"DMC_LR": ProbabilisticClassifierChainBase(base_estimator=clf_DMC_LR),
    "BR_LR": BinaryRelevance(classifier=LR),
}




# Metrics
metric_functions = [
    {"name": "Zero one", "func": EvaluationMetrics.subset_zero_one_loss},
    {"name": "Hamming", "func": EvaluationMetrics.hamming_loss},
    {"name": "F1", "func": EvaluationMetrics.f1_sample},
    {"name": "Phi avg", "func": EvaluationMetrics.phi_avg},
    {"name": "Phi max", "func": EvaluationMetrics.phi_max},
    {"name": "Psi avg", "func": EvaluationMetrics.psi_avg},
    {"name": "Psi max", "func": EvaluationMetrics.psi_max},
]

# Number of folds
k=5
# Seed
seeds = [117]

if __name__ == '__main__':
    print("Running....")
    run_all_experiments(models, datasets, metric_functions,seeds,k)