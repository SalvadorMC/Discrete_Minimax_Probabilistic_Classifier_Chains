from sklearn.metrics import confusion_matrix
from sklearn.metrics import zero_one_loss,hamming_loss,f1_score,jaccard_score,precision_score,recall_score

import numpy as np
import pandas as pd
from scipy.io import arff
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

def load_from_arff(path, label_count, label_location="end",
                   input_feature_type='float'):
    """
    Method for loading ARFF files as numpy arrays.
    Parameters
    ----------
    path : str
        Path to the ARFF file.
    label_count : int
        Number of labels in the ARFF file.
    label_location : str {"start", "end"} (default is "end")
        Whether the ARFF file contains labels at the beginning of the
        attributes list ("start", MEKA format) or at the end ("end", MULAN format).
    input_feature_type : numpy.type as string (default is "float")
        The desired type of the contents of the return 'X' arrays.

    Returns
    -------
    X : numpy.ndarray of `input_feature_type`, shape=(n_samples, n_features)
        Input feature matrix.
    y : numpy.ndarray of `{0, 1}`, shape=(n_samples, n_labels)
        Binary indicator matrix with label assignments.
    feature_names : List[str]
        List of feature attribute names from the ARFF file.
    label_names : List[str]
        List of label attribute names from the ARFF file.
    """
    # Load data and metadata from the ARFF file
    data, meta = arff.loadarff(path)
    #print(type(data))
    # Convert the data to a numpy array
    data = np.array(data.tolist(), dtype=input_feature_type)

    # Extract attribute names

    # Split data into features (X) and labels (y)
    if label_location == "start":
        X = data[:, label_count:].astype(input_feature_type)
        y = data[:, :label_count].astype(int)
    elif label_location == "end":
        X = data[:, :-label_count].astype(input_feature_type)
        y = data[:, -label_count:].astype(int)
    else:
        raise ValueError("Unknown label_location: must be 'start' or 'end'.")


    return X, y.astype(int)

def load_data(dataset_name="CHD_49.arff",path="Datasets/"):
    datasets = {
        "GpositivePseAAC.arff": 4,
        "Image.arff": 5,
        "Scene.arff": 6,
        "VirusPseAAC.arff": 6,
        "Emotions.arff": 6,
        "CHD_49.arff": 6,
        "Flags.arff": 7,
        "GnegativePseAAC.arff": 8,
        "PlantPseAAC.arff": 12,
        "Foodtruck.arff": 12,
    }

    if dataset_name == "Foodtruck.arff":
        base_path = os.path.abspath(path)
        name = os.path.join(base_path, dataset_name)
        # Leer el archivo .arff
        data, meta = arff.loadarff(name)

        # Convertir a DataFrame
        df = pd.DataFrame(data)

        # Decodificar columnas que están como bytes (categóricas)
        for col in df.select_dtypes([object]):
            df[col] = df[col].str.decode('utf-8')

        # Aplicar LabelEncoder a todas las columnas categóricas
        label_encoders = {}  # opcional, por si querés guardar los codificadores

        for col in df.select_dtypes(include=['object', 'category']):
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le  # guardar por si necesitás decodificar después

        # Convertir a numpy array si querés
        data = df.to_numpy()
        data = np.array(data.tolist(), dtype="float")

        numero_labels = datasets[dataset_name]
        X = data[:, :-numero_labels].astype(float)
        y = data[:, -numero_labels:].astype(int)
        return X, y

    if dataset_name == "ABPM.arff":
        base_path = os.path.abspath(path)
        name = os.path.join(base_path, dataset_name)

        numero_labels = datasets[dataset_name]
        X, y = load_from_arff(name, label_count=numero_labels,label_location="start")
        return X, y

    base_path = os.path.abspath(path)
    name = os.path.join(base_path, dataset_name)

    numero_labels = datasets[dataset_name]
    X, y = load_from_arff(name, label_count=numero_labels)
    return X, y


def compute_prios(y):
    """

    :param y: matrix (n_samples,n_labels)
    :return: (n_labels,2) prios per label
    """
    # Total samples
    n_samples = y.shape[0]

    # Count occurrences of each label (1s)
    label_counts = np.sum(y, axis=0)

    # Compute priors
    priors = np.zeros((y.shape[1], 2))
    priors[:, 1] = label_counts / n_samples  # P(label = 1)
    priors[:, 0] = 1 - priors[:, 1]  # P(label = 0)
    return priors

def IRLbl(y):
    """
    Calcula el desequilibrio por etiqueta (IRLbl) en un dataset multi-label.

    Parameters
    ----------
    y : numpy.ndarray
        Matriz binaria de etiquetas de tamaño (n_samples, n_labels).

    Returns
    -------
    numpy.ndarray
        Un array con el desequilibrio por etiqueta (IRLbl) para cada etiqueta.
    """
    # Dimensiones del dataset
    num_instances, num_labels = y.shape

    # Sumar positivos por etiqueta
    sum_list = np.sum(y, axis=0)

    # Valor máximo de positivos en todas las etiquetas
    max_IRLbl = max(sum_list)

    # Calcular IRLbl para cada etiqueta (manejo de etiquetas sin instancias positivas)
    irlbl = [
        max_IRLbl / s if s > 0 else np.inf  # Evita la división por 0
        for s in sum_list
    ]

    return np.array(irlbl)



def compute_conditional_risk(y_true: np.ndarray, y_pred: np.ndarray, K: int, L: np.ndarray):
    '''
    Function to compute the class-conditional risks.
    Parameters
    ----------
    YR : DataFrame
        Real labels.
    Yhat : Array
        Predicted labels.
    K : int
        Number of classes.
    L : Array
        Loss Function.

    Returns
    -------
    R : Array of floats
        Conditional risks.
    confmat : Matrix
        Confusion matrix.
    '''
    Labels = [i for i in range(K)]
    confmat = confusion_matrix(np.array(y_true), np.array(y_pred), normalize='true', labels=Labels)
    R = np.sum(np.multiply(L, confmat), axis=1)
    return R, confmat

class EvaluationMetrics:
    @staticmethod
    def hamming_loss(y_true,y_pred):
        return hamming_loss(y_true,y_pred)

    @staticmethod
    def subset_zero_one_loss(y_true,y_pred):
        return zero_one_loss(y_true,y_pred)

    @staticmethod
    def f1_sample(y_true,y_pred):
        return f1_score(y_true, y_pred, average='samples',zero_division=1)

    @staticmethod
    def phi_avg(y_true, y_pred):
        phi_values = []

        for i in range(y_true.shape[1]):
            R, _ = compute_conditional_risk(
                y_true[:, i],
                y_pred[:, i],
                2,
                np.array([[0, 1], [1, 0]]) #return R in this case is false positive and false negative
            )
            phi_values.append(np.abs(R[0] - R[1]))
        return np.mean(phi_values)

    @staticmethod
    def phi_max(y_true,y_pred):
        phi_values = []

        for i in range(y_true.shape[1]):
            R, _ = compute_conditional_risk(
                y_true[:, i],
                y_pred[:, i],
                2,
                np.array([[0, 1], [1, 0]])  # return R in this case is false positive and false negative
            )
            phi_values.append(np.abs(R[0] - R[1]))
        return np.max(phi_values)

    @staticmethod
    def psi_avg(y_true, y_pred):
        max_errors = []

        for i in range(y_true.shape[1]):
            R, _ = compute_conditional_risk(
                y_true[:, i],
                y_pred[:, i],
                2,
                np.array([[0, 1], [1, 0]])  # FP = R[0], FN = R[1]
            )
            max_errors.append(max(R[0], R[1]))

        return np.mean(max_errors)

    @staticmethod
    def psi_max(y_true, y_pred):
        max_errors = []

        for i in range(y_true.shape[1]):
            R, _ = compute_conditional_risk(
                y_true[:, i],
                y_pred[:, i],
                2,
                np.array([[0, 1], [1, 0]])  # FP = R[0], FN = R[1]
            )
            max_errors.append(max(R[0], R[1]))

        return np.max(max_errors)
