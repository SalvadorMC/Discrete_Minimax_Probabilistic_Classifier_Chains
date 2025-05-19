import numpy as np
from sklearn.neighbors import NearestNeighbors
from skmultilearn.problem_transform import BinaryRelevance,ClassifierChain,LabelPowerset


def calculate_IRLbl(Y):
    """
    Calculate the Imbalance Ratio (IR) for each label.

    IR(lbl) = max(posNumsPerLabel) / posNumsPerLabel(lbl)

    If a label has 0 positives, returns IR = 0 by default to avoid division by zero.
    Args:
        Y (np.ndarray): Binary matrix of labels, shape (n_samples, n_labels).
    Returns:
        np.ndarray: Array with IR for each label, shape (n_labels,).
    """
    posNumsPerLabel = np.sum(Y, axis=0)
    maxPosNums = np.max(posNumsPerLabel)
    IR = []
    for p in posNumsPerLabel:
        if p == 0:
            IR.append(0.0)  # Could use float('inf') if you prefer
        else:
            IR.append(maxPosNums / p)
    return np.array(IR)


def calculate_meanIR(Y):
    """
    Calculate the mean imbalance ratio across all labels.

    Args:
        Y (np.ndarray): Binary matrix of labels, shape (n_samples, n_labels).
    Returns:
        float: Mean IR.
    """
    IRLbl = calculate_IRLbl(Y)
    return np.mean(IRLbl)


def get_minBag(Y):
    """
    Identify minority labels based on their IR compared to the mean IR.

    A label i is considered minority if IR(i) > mean(IR).

    Args:
        Y (np.ndarray): Binary matrix of labels, shape (n_samples, n_labels).
    Returns:
        list: Indices of minority labels.
    """
    IRLbl = calculate_IRLbl(Y)
    meanIR = calculate_meanIR(Y)
    return [i for i in range(Y.shape[1]) if IRLbl[i] > meanIR]


def NN_index(X, k=5):
    """
    Compute the k nearest neighbors for each instance in X.

    Args:
        X (np.ndarray): Feature matrix, shape (n_samples, n_features).
        k (int): Number of neighbors to find.
    Returns:
        np.ndarray: Indices of the k nearest neighbors for each instance, shape (n_samples, k).
    """
    nn = NearestNeighbors(n_neighbors=k + 1, metric='euclidean', algorithm='auto').fit(X)
    _, indices = nn.kneighbors(X)
    # Exclude the instance itself -> we keep the last k neighbors
    return indices[:, 1:]


def MLSMOTE(X, Y, k=2, random_state=None):
    """
    Perform MLSMOTE (Synthetic Minority Oversampling Technique for multi-label data).

    Args:
        X (np.ndarray): Feature matrix, shape (n_samples, n_features).
        Y (np.ndarray): Binary label matrix, shape (n_samples, n_labels).
        k (int): Number of nearest neighbors to use.
        random_state (int or None): Seed for random generator (optional).
    Returns:
        (X_new, Y_new): Tuple of oversampled feature and label matrices.
    """
    if random_state is not None:
        np.random.seed(random_state)

    if not isinstance(X, np.ndarray) or not isinstance(Y, np.ndarray):
        raise ValueError("X and Y must be NumPy arrays.")
    if X.shape[0] != Y.shape[0]:
        raise ValueError("X and Y must have the same number of instances.")
    if k >= X.shape[0]:
        raise ValueError("k must be less than the number of samples.")

    minority_labels = get_minBag(Y)
    X_new, Y_new = X.copy(), Y.copy()

    for label in minority_labels:
        X_minor = X[Y[:, label] == 1]
        Y_minor = Y[Y[:, label] == 1]

        # Si no hay muestras que contengan esta etiqueta (Y[:, label] == 1 da array vacío)
        # se salta automáticamente, aunque no es común que pase luego del filtro:
        if X_minor.shape[0] == 0:
            continue

        nn_indices = NN_index(X_minor, k)

        for i in range(len(X_minor)):
            sample = X_minor[i]
            # Elige un vecino aleatorio de los k vecinos
            ref_idx = np.random.choice(nn_indices[i])
            ref_sample = X_minor[ref_idx]

            # Generamos la nueva instancia sintética
            diff = ref_sample - sample
            new_sample = sample + diff * np.random.uniform(0, 1)

            # Votación mayoritaria entre los k vecinos
            combined_labels = np.vstack([Y_minor[i].reshape(1, -1), Y_minor[nn_indices[i]]])

            #nn_labels = Y_minor[nn_indices[i]]
            new_label = (np.sum(combined_labels, axis=0) >= (k + 1) / 2).astype(int)

            X_new = np.vstack([X_new, new_sample])
            Y_new = np.vstack([Y_new, new_label])
            # === CAMBIO 2: Actualización dinámica del desequilibrio ===
            # Recalcular IRLbl y MeanIR usando Y_new actualizado para la etiqueta actual.
            current_IR = calculate_IRLbl(Y_new)[label]
            current_meanIR = calculate_meanIR(Y_new)
            # Si la etiqueta ya no es minoritaria (IR <= MeanIR), se detiene la generación para ella.
            if current_IR <= current_meanIR:
                break

    return X_new, Y_new


def MLROS(X, Y, ratio=0.1, random_state=None):
    """
    Perform Minority Label Random Oversampling (MLROS) on a multilabel dataset.

    Args:
        X (np.ndarray): Feature matrix, shape (n_samples, n_features).
        Y (np.ndarray): Binary matrix of labels, shape (n_samples, n_labels).
        ratio (float): Proportion of new samples to add relative to the dataset size (N).
        random_state (int or None): Seed for random generator (optional).
    Returns:
        (X_new, Y_new): Oversampled feature and label matrices.
    """
    if random_state is not None:
        np.random.seed(random_state)

    if not isinstance(X, np.ndarray) or not isinstance(Y, np.ndarray):
        raise ValueError("X and Y must be NumPy arrays.")
    if X.shape[0] != Y.shape[0]:
        raise ValueError("X and Y must have the same number of instances.")

    N = X.shape[0]
    samplesToClone = int(N * ratio)
    minBag = get_minBag(Y)

    X_new, Y_new = X.copy(), Y.copy()
    skipInd = set()

    while samplesToClone > 0 and len(skipInd) < len(minBag):
        for i in minBag:
            if i in skipInd:
                continue

            # Instancias que poseen la etiqueta i
            minBagInd = np.where(Y_new[:, i] == 1)[0]
            if len(minBagInd) == 0:
                # Si ya no hay instancias con esa etiqueta (quizás en un dataset muy pequeño),
                # no podremos clonar más -> descartar
                skipInd.add(i)
                continue

            # Elige al azar una de las instancias que poseen la etiqueta
            instanceIdx = np.random.choice(minBagInd)
            instance = X_new[instanceIdx].reshape(1, -1)
            target = Y_new[instanceIdx].reshape(1, -1)

            # Añadir una copia
            X_new = np.vstack([X_new, instance])
            Y_new = np.vstack([Y_new, target])
            samplesToClone -= 1

            # Recalcular IR para ver si la etiqueta i sigue siendo minoritaria
            IRLbl_new = calculate_IRLbl(Y_new)
            meanIR_new = np.mean(IRLbl_new)
            if IRLbl_new[i] <= meanIR_new:
                skipInd.add(i)

            if samplesToClone <= 0:
                break

    return X_new, Y_new
























