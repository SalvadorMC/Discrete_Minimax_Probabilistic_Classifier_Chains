import matplotlib.pyplot as plt
from itertools import combinations
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.utils.validation import check_X_y
from sklearn.base import BaseEstimator, ClassifierMixin
import warnings
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import numpy as np
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings('ignore')


class DBC(BaseEstimator, ClassifierMixin):
    """
      Discrete Bayesian Classifier (DBC)
      -----------------------------------
      A classification model that discretizes the feature space using KMeans clustering
      and applies a Bayesian approach for prediction.

      Parameters:
      -----------
      T : int or "auto", default="auto"
          Number of discrete profiles (clusters) to generate. If set to "auto", the classifier
          will attempt to determine the optimal number of profiles automatically.

    Attributes:
    -----------
    pHat : np.ndarray
        Estimated probabilities for each discrete profile after fitting the model.

    piStar : np.ndarray
        The posterior probabilities for each class after fitting.

    """
    def __init__(
            self,
            T="auto",
            random_state=None,
    ):
        self.T = T
        self.random_state = random_state
        self.pHat = None
        self.piStar = None

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        K = len(self.classes_)
        self.L = np.ones((K, K)) - np.eye(K)
        if self.T == "auto":
            self.T = self.get_T_optimal(X, y)['T']
        self.discretize(X,y)
        discrete_profiles = self.clf_discretizer.labels_
        self.pHat = compute_pHat(discrete_profiles, y, K, self.T)
        self.piStar = compute_pi(y, K)
        return self

    def predict(self, X):
        discrete_profiles = self.clf_discretizer.predict(X)
        y_pred =predict_profile_label(self.piStar, self.pHat, self.L)[discrete_profiles]
        return y_pred

    def predict_proba(self,X):
        lambd = (self.piStar.reshape(-1, 1) * self.L).T @ self.pHat
        prob = 1 - (lambd / np.sum(lambd, axis=0))
        return prob[:, self.clf_discretizer.predict(X)].T

    def discretize(self,X,y):
        self.clf_discretizer = KMeans(n_clusters=self.T)
        self.clf_discretizer.fit(X)

    def get_T_optimal(self, X, y, T_start=None, T_end=None, Num_t_Values=15):
        if T_start is None:
            T_start = int(len(X) / 30)
        if T_end is None:
            T_end = int(len(X) / 4)
        t_values = np.unique(np.linspace(T_start, T_end, Num_t_Values, dtype=int))
        param_grid = {
            'T': t_values
        }
        grid_search = GridSearchCV(estimator=DBC(), param_grid=param_grid, cv=3, scoring="accuracy")
        grid_search.fit(X, y)
        results = grid_search.cv_results_
        mean_scores = results['mean_test_score']
        params = results['param_T'].data  # 'param_T' es una columna de objetos

        max_score = np.max(mean_scores)
        best_Ts = [params[i] for i, score in enumerate(mean_scores) if score == max_score]
        best_T = max(best_Ts)  # elegir el T más grande entre los mejores
        return {'T': best_T}


class DiscreteMinimaxClassifier(DBC):
    """
    Discrete Minimax Classifier (DMC)
    -----------------------------------
    A classification model that discretizes the feature space using KMeans clustering
      and applies a Minimax approach for prediction.

    Inherits:
    ---------
    DBC : Discrete Bayesian Classifier
        This class extends the DBC with additional parameters for minimax optimization.

    Parameters:
    -----------
    T : int or "auto", default="auto"
        Number of discrete profiles (clusters) to generate. If set to "auto", the classifier
        will attempt to determine the optimal number of profiles automatically.
    N: int
        Number of iterations to find the least favorable priors.
    L: array, default=None
        Loss matrix. If `L=None`, a Zero-One loss is assumed.

    Attributes:
    -----------
    pHat : np.ndarray
        Estimated probabilities for each discrete profile after fitting the model.

    piTrain : np.ndarray
        The posterior probabilities for each class after fitting
    piStar : np.ndarray
        The least favorables priors.

    """
    def __init__(
            self,
            N=10000,
            L=None,
            T = "auto",
    ):
        super().__init__(T=T)
        #self.T = T
        self.N = N
        self.L = L

    def fit(self, X, y):
        self.piStar = None
        self.piTrain = None
        self.pHat = None

        K = len(np.unique(y))
        if self.L is None:
            self.L = np.ones((K, K)) - np.eye(K)
        if self.T == "auto":
            self.T = self.get_T_optimal(X, y)['T']
        self.discretize(X,y)
        self.discrete_profiles = self.clf_discretizer.labels_
        #self.T = self.clf_discretizer.T
        self.pHat = compute_pHat(self.discrete_profiles, y, K, self.T)
        self.piTrain = compute_pi(y, K)

        self.piStar, rStar, self.RStar, V_iter, stockpi = compute_piStar(self.pHat, y, K, self.L, self.T,
                                                                         self.N, 0, self.box)
        self.classes_ = np.unique(y)
        return self

    def predict(self, X, pi=None):
        if pi is None:
            pi = self.piStar

        discrete_profiles = self.clf_discretizer.predict(X)

        return predict_profile_label(pi, self.pHat, self.L)[discrete_profiles]

class KMeansUnionDiscretizer:
    """
    Class-Aware Discretizer via KMeans:
      - Divides each class into a number of clusters proportional to its frequency.
      - Merges all centroids and assigns each point to its nearest center.

    Parameters
    ----------
    T : int
        Total number of desired discrete regions.
    """

    def __init__(self, T=10, random_state=None):
        self.T = T
        self.random_state = random_state
        self.centers_ = None

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        classes, counts = np.unique(y, return_counts=True)
        total = len(y)

        n_clusters = {}
        for c, cnt in zip(classes, counts):
            n_clusters[c] = max(1, round(cnt / total * self.T))
        delta = self.T - sum(n_clusters.values())
        for _ in range(abs(delta)):
            key = classes[np.argmax(counts)] if delta < 0 else classes[np.argmin(counts)]
            n_clusters[key] += 1 if delta < 0 else -1

        centroids = []
        for c in classes:
            Xc = X[y == c]
            k = n_clusters[c]
            kmeans = KMeans(n_clusters=k, random_state=self.random_state)
            centroids.append(kmeans.fit(Xc).cluster_centers_)
        self.centers_ = np.vstack(centroids)
        distances = cdist(X, self.centers_)
        self.labels_ = np.argmin(distances, axis=1)
        centers_clean = self._cleanup_centers(
            self.centers_, X, self.labels_
        )

        self.centers_ = centers_clean
        self.T = len(centers_clean)
        return self

    def predict(self, X):
        if self.centers_ is None:
            raise ValueError("Error")
        X = np.asarray(X)
        dist = cdist(X, self.centers_)
        return np.argmin(dist, axis=1)

    def _cleanup_centers(self,centers, X, labels, min_size=1):

        counts = np.bincount(labels, minlength=len(centers))

        for idx, cnt in enumerate(counts):
            if cnt < min_size:
                dists = np.min(cdist(X, np.delete(centers, idx, axis=0)), axis=1)
                far_idx = np.argmax(dists)
                centers[idx] = X[far_idx]
                counts[idx] = 1
        return centers

class DBC_KmeansU(DBC):
    """
    Discrete Bayesian Classifier using Kmeans Union discretization
    Inherits:
    DBC

    """
    def discretize(self,X,y):
        self.clf_discretizer = KMeansUnionDiscretizer(T=self.T)
        self.clf_discretizer.fit(X,y)
        self.T =self.clf_discretizer.T

class DMC_KmeansU(DiscreteMinimaxClassifier):
    """
    Discrete Minimax Classifier using Kmeans Union discretization

    inherits:
    DiscreteMinimaxClassifier
    """
    def discretize(self,X,y):
        self.clf_discretizer = KMeansUnionDiscretizer(T=self.T)
        self.clf_discretizer.fit(X,y)
        self.T =self.clf_discretizer.T

    def get_T_optimal(self, X, y, T_start=None, T_end=None, Num_t_Values=15):
        #todo check this values
        if T_start is None:
            T_start = int(len(X) / 30)
        if T_end is None:
            T_end = int(len(X) / 4)
        t_values = np.unique(np.linspace(T_start, T_end, Num_t_Values, dtype=int))
        #print(t_values)
        param_grid = {
            'T': t_values
        }
        grid_search = GridSearchCV(estimator=DBC_KmeansU(), param_grid=param_grid, cv=3, scoring="accuracy")
        grid_search.fit(X, y)
        # Extraer todos los resultados
        results = grid_search.cv_results_
        mean_scores = results['mean_test_score']
        params = results['param_T'].data  # 'param_T' es una columna de objetos

        # Encuentra el mayor T entre los mejores scores
        max_score = np.max(mean_scores)
        best_Ts = [params[i] for i, score in enumerate(mean_scores) if score == max_score]
        best_T = max(best_Ts)  # elegir el T más grande entre los mejores
        return {'T': best_T}

# Logistic regression dicretizer ############
class LogisticDiscretizer:
    """
    Discretizes a dataset into T regions based on contour curves
    of the probability output of a logistic regression classifier,
    using 0.5 as the baseline and distributing thresholds on both sides.

    Parameters
    ----------
    T : int
        Number of desired discrete regions.
     """

    def __init__(self, T=10, random_state=None,weighted=False):
        self.T = T
        self.random_state = random_state
        self.clf_ = None
        self.thresholds_ = None
        self.labels_ = None
        self.weighted = weighted

    def fit(self, X, y):
        """
        Fits the model with X and y, calculates thresholds distributed
        around 0.5 (relative quartiles), and assigns labels.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix.
        y : array-like, shape (n_samples,)
            Target vector.

        Returns
        -------
        self : object
            The fitted instance of the classifier.
        """
        X = np.array(X)
        y = np.array(y)

        # Entrenar regresión logística y obtener probabilidades
        if self.weighted:
            self.clf_ = LogisticRegression(random_state=self.random_state,class_weight="balanced")
        else:
            self.clf_ = LogisticRegression(random_state=self.random_state)
        self.clf_.fit(X, y)
        probs = self.clf_.predict_proba(X)[:, 1]

        # Número de thresholds interiores y cómo repartirlos
        # T regiones generan T-1 cortes; uno se fija en 0.5
        n_inner = self.T - 1
        remaining = n_inner - 1
        lower_cnt = remaining // 2
        upper_cnt = remaining - lower_cnt

        # Subconjuntos de probabilidades bajo y sobre 0.5
        below = probs[probs < 0.5]
        above = probs[probs > 0.5]

        # Cálculo de thresholds para cada lado
        thr_b = np.array([])
        if lower_cnt > 0 and len(below) >= lower_cnt:
            p_b = np.linspace(0, 100, lower_cnt + 2)[1:-1]
            thr_b = np.percentile(below, p_b)

        thr_a = np.array([])
        if upper_cnt > 0 and len(above) >= upper_cnt:
            p_a = np.linspace(0, 100, upper_cnt + 2)[1:-1]
            thr_a = np.percentile(above, p_a)

        # Construir array de thresholds incluyendo extremos y 0.5
        self.thresholds_ = np.concatenate([
            [probs.min()],
            thr_b,
            [0.5],
            thr_a,
            [probs.max()]
        ])

        # Asignar cada muestra a su región
        self.labels_ = np.digitize(probs, self.thresholds_[1:-1], right=True)
        return self

    def predict(self, X):
        X = np.array(X)
        probs = self.clf_.predict_proba(X)[:, 1]
        return np.digitize(probs, self.thresholds_[1:-1], right=True)

class DBC_logistic(DBC):
    """
    Discrete Bayesian Classifier using logistic discretization
    inherints:
        BDC
    """
    def __init__(self,T="auto",weighted=False):
        super().__init__(T=T)
        self.weighted = weighted
    def discretize(self,X,y):
        self.clf_discretizer = LogisticDiscretizer(T=self.T,weighted=self.weighted)
        self.clf_discretizer.fit(X,y)


class DMC_logistic(DiscreteMinimaxClassifier):
    """
    Discrete Minimax Classfier Classifier using logistic discretization
    inherints:
        Discrete Minimax Classifier
    """
    def __init__(self,weighted=False,T="auto"):
        super().__init__(T=T)
        self.weighted = weighted

    def discretize(self,X,y):
        self.clf_discretizer = LogisticDiscretizer(T=self.T,weighted=self.weighted)
        self.clf_discretizer.fit(X,y)
    def get_T_optimal(self, X, y, T_start=None, T_end=None, Num_t_Values=15):
        if T_start is None:
            #T_start = int(len(X) / 25)
            T_start = 10
        if T_end is None:
            T_end = int(len(X) / 5)
            #T_end = 20
        t_values = np.unique(np.linspace(T_start, T_end, Num_t_Values, dtype=int))
        #print(t_values)
        param_grid = {
            'T': t_values
        }
        grid_search = GridSearchCV(estimator=DBC_logistic(weighted=self.weighted), param_grid=param_grid, cv=3, scoring="accuracy")
        grid_search.fit(X, y)
        # Extraer todos los resultados
        results = grid_search.cv_results_
        mean_scores = results['mean_test_score']
        params = results['param_T'].data  # 'param_T' es una columna de objetos

        # Encuentra el mayor T entre los mejores scores
        max_score = np.max(mean_scores)
        best_Ts = [params[i] for i, score in enumerate(mean_scores) if score == max_score]
        best_T = max(best_Ts)  # elegir el T más grande entre los mejores
        return {'T': best_T}


class DecisionTreeDiscretizer:
    """
    Discretizes data into regions using the leaves of a decision tree.

    Parameters
    ----------
    T : int or None
        Maximum number of desired leaves (regions).
    min_samples_leaf : int
        Minimum number of samples required for each leaf.
    random_state : int or None
        Seed for reproducibility.
    """

    def __init__(self, T=None, min_samples_leaf=1,random_state=None,weighted=False):
        self.max_leaf_nodes = T
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state

        self.tree_ = None
        self.leaf_mapping = None  # dict {leaf_id: región_index}
        self.T = None
        self.weighted = weighted

    def fit(self, X, y):
        if self.weighted:
            self.tree_ = DecisionTreeClassifier(
            max_leaf_nodes=self.max_leaf_nodes, #maximo numero de hojas
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
            class_weight='balanced'
            )
        else:
            self.tree_ = DecisionTreeClassifier(
            max_leaf_nodes=self.max_leaf_nodes, #maximo numero de hojas
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
            )
        self.tree_.fit(X, y)
        leaf_ids = self.tree_.apply(X)
        unique_leaves = np.unique(leaf_ids)

        self.leaf_mapping = {leaf: idx for idx, leaf in enumerate(unique_leaves)}
        self.T = len(unique_leaves)
        self.labels_ = self.predict(X)

        return self

    def predict(self, X):
        if self.tree_ is None or self.leaf_mapping is None:
            raise ValueError("Discretizer not fitted. Call `fit` first.")

        leaf_ids = self.tree_.apply(X)
        try:
            return np.array([self.leaf_mapping[leaf] for leaf in leaf_ids])
        except KeyError as e:
            raise ValueError(f"Leaf ID desconocido: {e}")


class DBC_Tree(DBC):
    """
    Discrete Bayesian Classifier using a Tree discretization

    """
    def __init__(self,T="auto",weighted=False):
        super().__init__(T=T)
        self.weighted = weighted
    def discretize(self,X,y):
        self.clf_discretizer = DecisionTreeDiscretizer(T=self.T,weighted=self.weighted)
        self.clf_discretizer.fit(X, y)
        self.T = self.clf_discretizer.T


class DMC_Tree(DiscreteMinimaxClassifier):
    def __init__(self,weighted=False,T="auto"):
        super().__init__(T=T)
        self.weighted = weighted
    def discretize(self,X,y):
        self.clf_discretizer = DecisionTreeDiscretizer(T=self.T,weighted=self.weighted)
        self.clf_discretizer.fit(X,y)
        self.T =self.clf_discretizer.T

    def get_T_optimal(self, X, y,T_start=None, T_end=None,Num_t_Values=15):

        if T_start is None:
            T_start = int(len(X) / 30)
        if T_end is None:
            T_end = int(len(X) / 4)
        t_values = np.unique(np.linspace(T_start, T_end, Num_t_Values, dtype=int))
        #print(t_values)
        param_grid = {
            'T': t_values
        }
        grid_search = GridSearchCV(estimator=DBC_Tree(), param_grid=param_grid, cv=3, scoring="accuracy")
        grid_search.fit(X, y)
        # Extraer todos los resultados
        results = grid_search.cv_results_
        mean_scores = results['mean_test_score']
        params = results['param_T'].data  # 'param_T' es una columna de objetos

        # Encuentra el mayor T entre los mejores scores
        max_score = np.max(mean_scores)
        best_Ts = [params[i] for i, score in enumerate(mean_scores) if score == max_score]
        best_T = max(best_Ts)  # elegir el T más grande entre los mejores
        return {'T': best_T}




def compute_pi(y, K):
    """
    Parameters
    ----------
    y : ndarray of shape (n_samples,)
        Labels

    K : int
        Number of classes

    Returns
    -------
    pi : ndarray of shape (K,)
        Proportion of classes
    """
    pi = np.zeros(K)
    total_count = len(y)

    for k in range(K):
        pi[k] = np.sum(y == k) / total_count
    return pi

def compute_pHat(XD: np.ndarray, y, K, T):
    """
    Parameters
    ----------
    XD : ndarray of shape (n_samples,)
        Labels of profiles for each data point

    y : ndarray of shape (n_samples,)
        Labels

    K : int
        Number of classes

    T : int
        Number of profiles

    Returns
    -------
    pHat : ndarray of shape(K, n_profiles)
    Modificado
    """
    #y = check_array(y, ensure_2d=False)
    pHat = np.zeros((K, T))
    for k in range(K):
        Ik = np.where(y == k)[0]
        mk = len(Ik)
        if mk != 0:
            pHat[k] = np.bincount(XD[Ik], minlength=T) / mk
        else:
            pHat[k] = np.zeros(T)
        # Count number of occurrences of each value in array of non-negative ints.
    return pHat

def delta_proba_U(U, pHat, pi, L, methode='before', temperature=0):
    '''
    Parameters
    ----------
    U : Array

    pHat : Array of floats
        Probability estimate of observing the features profile.
    pi : Array of floats
        Real class proportions.
    L : Array
        Loss function.

    Returns
    -------
    Yhat : Vector
        Predicted labels.
    '''

    def softmin_with_temperature(X, temperature=1.0, axis=1):
        X = -X
        X_max = np.max(X, axis=axis, keepdims=True)
        X_adj = X - X_max

        exp_X_adj = np.exp(X_adj / temperature)
        softmax_output = exp_X_adj / np.sum(exp_X_adj, axis=axis, keepdims=True)

        return softmax_output

    lambd = U.T @ ((pi.T * L).T @ pHat).T

    if methode == 'softmin':
        prob = softmin_with_temperature(lambd, temperature)

    elif methode == 'argmin':
        prob = np.zeros_like(lambd)
        rows = np.arange(lambd.shape[0])
        cols = np.argmin(lambd, axis=1)
        prob[rows, cols] = 1

    elif methode == 'proportion':
        prob = 1 - np.divide(lambd, np.sum(lambd, axis=1)[:, np.newaxis])

    elif methode == 'before':
        prob = 1 - np.divide(lambd, np.sum(lambd, axis=1)[:, np.newaxis])

    elif methode == 'after':
        prob_init = 1 - np.divide(lambd, np.sum(lambd, axis=1)[:, np.newaxis])
        index = np.argmax(prob_init, axis=1)
        prob = np.zeros_like(prob_init)
        prob[np.arange(index.shape[0]), index] = 1
    return prob

def compute_conditional_risk(y_true, y_pred, K=2, L=None):
    '''
    Function to compute the class-conditional risks.
    Parameters
    ----------
    y_true : DataFrame
        Real labels.
    y_pred : Array
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
    if L is None:
        L = np.ones((K, K)) - np.eye(K)
    Labels = [i for i in range(K)]
    confmat = confusion_matrix(np.array(y_true), np.array(y_pred), normalize='true', labels=Labels)
    R = np.sum(np.multiply(L, confmat), axis=1)

    return R

def max_risk(y_true, y_pred):
    k = len(np.unique(y_true))
    L = np.ones((k, k)) - np.eye(k)
    pi = compute_pi(y_true, k)
    R, M = compute_conditional_risk(y_true, y_pred, k, L)

def compute_global_risk(R, pi):
    """
    Parameters
    ----------
    R : ndarray of shape (K,)
        Conditional risk
    pi : ndarray of shape (K,)
        Proportion of classes

    Returns
    -------
    r : float
        Global risk.
    """

    r = np.sum(R * pi)

    return r

def predict_profile_label(pi, pHat, L):
    lambd = (pi.reshape(-1, 1) * L).T @ pHat
    lbar = np.argmin(lambd, axis=0)
    return lbar

def proj_simplex_Condat(K, pi):
    """
    This function is inspired from the article: L.Condat, "Fast projection onto the simplex and the
    ball", Mathematical Programming, vol.158, no.1, pp. 575-585, 2016.
    Parameters
    ----------
    K : int
        Number of classes.
    pi : Array of floats
        Vector to project onto the simplex.

    Returns
    -------
    piProj : List of floats
        Priors projected onto the simplex.

    """

    linK = np.linspace(1, K, K)
    piProj = np.maximum(pi - np.max(((np.cumsum(np.sort(pi)[::-1]) - 1) / (linK[:]))), 0)
    piProj = piProj / np.sum(piProj)
    return piProj

def graph_convergence(V_iter):
    '''
    Parameters
    ----------
    V_iter : List
        List of value of V at each iteration n.

    Returns
    -------
    Plot
        Plot of V_pibar.

    '''

    figConv = plt.figure(figsize=(8, 4))
    plt_conv = figConv.add_subplot(1, 1, 1)
    V = V_iter.copy()
    V.insert(0, np.min(V))
    font = {'weight': 'normal', 'size': 16}
    plt_conv.plot(V, label='V(pi(n))')
    plt_conv.set_xscale('log')
    plt_conv.set_ylim(np.min(V), np.max(V) + 0.01)
    plt_conv.set_xlim(10 ** 0)
    plt_conv.set_xlabel('Interation n', fontdict=font)
    plt_conv.set_title('Maximization of V over U', fontdict=font)
    plt_conv.grid(True)
    plt_conv.grid(which='minor', axis='x', ls='-.')
    plt_conv.legend(loc=2, shadow=True)

def num2cell(a):
    if type(a) is np.ndarray:
        return [num2cell(x) for x in a]
    else:
        return a

def proj_onto_polyhedral_set(pi, Box, K):
    '''
    Parameters
    ----------
    pi : Array of floats
        Vector to project onto the box-constrained simplex.
    Box : Array
        {'none', matrix} : Box-constraint on the priors.
    K : int
        Number of classes.

    Returns
    -------
    piStar : Array of floats
            Priors projected onto the box-constrained simplex.

    '''

    # Verification of constraints
    for i in range(K):
        for j in range(2):
            if Box[i, j] < 0:
                Box[i, j] = 0
            if Box[i, j] > 1:
                Box[i, j] = 1

    # Generate matrix G:
    U = np.concatenate((np.eye(K), -np.eye(K), np.ones((1, K)), -np.ones((1, K))))
    eta = Box[:, 1].tolist() + (-Box[:, 0]).tolist() + [1] + [-1]

    n = U.shape[0]

    G = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            G[i, j] = np.vdot(U[i, :], U[j, :])

    # Generate subsets of {1,...,n}:
    M = (2 ** n) - 1
    I = num2cell(np.zeros((1, M)))

    i = 0
    for l in range(n):
        T = list(combinations(list(range(n)), l + 1))
        for p in range(i, i + len(T)):
            I[0][p] = T[p - i]
        i = i + len(T)

    # Algorithm

    for m in range(M):
        Im = I[0][m]

        Gmm = np.zeros((len(Im), len(Im)))
        ligne = 0
        for i in Im:
            colonne = 0
            for j in Im:
                Gmm[ligne, colonne] = G[i, j]
                colonne += 1
            ligne += 1

        if np.linalg.det(Gmm) != 0:

            nu = np.zeros((2 * K + 2, 1))
            w = np.zeros((len(Im), 1))
            for i in range(len(Im)):
                w[i] = np.vdot(pi, U[Im[i], :]) - eta[Im[i]]

            S = np.linalg.solve(Gmm, w)

            for e in range(len(S)):
                nu[Im[e]] = S[e]

            if np.any(nu < -10 ** (-10)) == False:
                A = G.dot(nu)
                z = np.zeros((1, 2 * K + 2))
                for j in range(2 * K + 2):
                    z[0][j] = np.vdot(pi, U[j, :]) - eta[j] - A[j]

                if np.all(z <= 10 ** (-10)) == True:
                    pi_new = pi
                    for i in range(2 * K + 2):
                        pi_new = pi_new - nu[i] * U[i, :]

    piStar = pi_new

    # Remove noisy small calculus errors:
    piStar = piStar / piStar.sum()

    return piStar

def proj_onto_U(pi, Box, K):
    '''
    Parameters
    ----------
    pi : Array of floats
        Vector to project onto the box-constrained simplex..
    Box : Matrix
        {'none', matrix} : Box-constraint on the priors.
    K : int
        Number of classes.

    Returns
    -------
    pi_new : Array of floats
            Priors projected onto the box-constrained simplex.

    '''

    check_U = 0
    if pi.sum() == 1:
        for k in range(K):
            if (pi[0][k] >= Box[k, 0]) & (pi[0][k] <= Box[k, 1]):
                check_U = check_U + 1

    if check_U == K:
        pi_new = pi

    if check_U < K:
        pi_new = proj_onto_polyhedral_set(pi, Box, K)

    return pi_new

def compute_piStar(pHat, y_train, K, L, T, N, optionPlot, Box):
    """
    Parameters
    ----------
    pHat : Array of floats
        Probability estimate of observing the features profile in each class.
    y_train : Dataframe
        Real labels of the training set.
    K : int
        Number of classes.
    L : Array
        Loss Function.
    T : int
        Number of discrete profiles.
    N : int
        Number of iterations in the projected subgradient algorithm.
    optionPlot : int {0,1}
        1 plots figure,   0: does not plot figure.
    Box : Array
        {'none', matrix} : Box-constraints on the priors.

    Returns
    -------
    piStar : Array of floats
        Least favorable priors.
    rStar : float
        Global risks.
    RStar : Array of float
        Conditional risks.
    V_iter : Array
        Values of the V function at each iteration.
    stockpi : Array
        Values of pi at each iteration.

    """
    # IF BOX-CONSTRAINT == NONE (PROJECTION ONTO THE SIMPLEX)
    if Box is None:
        pi = compute_pi(y_train, K).reshape(1, -1)
        rStar = 0
        piStar = pi
        RStar = 0

        V_iter = []
        stockpi = np.zeros((K, N))

        for n in range(1, N + 1):
            # Compute subgradient R at point pi (see equation (21) in the paper)
            lambd = np.dot(L, pi.T * pHat)
            R = np.zeros((1, K))

            mu_k = np.sum(L[:, np.argmin(lambd, axis=0)] * pHat, axis=1)
            R[0, :] = mu_k
            stockpi[:, n - 1] = pi[0, :]

            r = compute_global_risk(R, pi)
            V_iter.append(r)
            if r > rStar:
                #print("aa", np.abs(r - rStar),n)
                rStar = r
                piStar = pi
                RStar = R
                # Update pi for iteration n+1
            gamma = 1 / n
            eta = np.maximum(float(1), np.linalg.norm(R))
            w = pi + (gamma / eta) * R
            pi = proj_simplex_Condat(K, w)

        # Check if pi_N == piStar
        lambd = np.dot(L, pi.T * pHat)
        R = np.zeros((1, K))

        mu_k = np.sum(L[:, np.argmin(lambd, axis=0)] * pHat, axis=1)
        R[0, :] = mu_k
        stockpi[:, n - 1] = pi[0, :]

        r = compute_global_risk(R, pi)
        if r > rStar:

            rStar = r
            piStar = pi
            RStar = R

        if optionPlot == 1:
            print("si")
            graph_convergence(V_iter)

    # IF BOX-CONSTRAINT
    if Box is not None:
        pi = compute_pi(y_train, K).reshape(1, -1)
        rStar = 0
        piStar = pi
        RStar = 0

        V_iter = []
        stockpi = np.zeros((K, N))

        for n in range(1, N + 1):
            # Compute subgradient R at point pi (see equation (21) in the paper)
            lambd = np.dot(L, pi.T * pHat)
            R = np.zeros((1, K))

            mu_k = np.sum(L[:, np.argmin(lambd, axis=0)] * pHat, axis=1)
            R[0, :] = mu_k
            stockpi[:, n - 1] = pi[0, :]

            r = compute_global_risk(R, pi)
            V_iter.append(r)
            if r > rStar:
                rStar = r
                piStar = pi
                RStar = R
                # Update pi for iteration n+1
            gamma = 1 / n
            eta = np.maximum(float(1), np.linalg.norm(R))
            w = pi + (gamma / eta) * R
            pi = proj_onto_U(w, Box, K)

        # Check if pi_N == piStar
        lambd = np.dot(L, pi.T * pHat)
        R = np.zeros((1, K))

        mu_k = np.sum(L[:, np.argmin(lambd, axis=0)] * pHat, axis=1)
        R[0, :] = mu_k
        stockpi[:, n - 1] = pi[0, :]

        r = compute_global_risk(R, pi)
        if r > rStar:
            rStar = r
            piStar = pi
            RStar = R

        if optionPlot == 1:
            print("AQU")
            graph_convergence(V_iter)

    return piStar, rStar, RStar, V_iter, stockpi



class BinaryRelevance:
    def __init__(self, classifier):
        """
        classifier: un estimador de scikit-learn con métodos fit(X, y) y predict(X)
        """
        self.base_clf = classifier

    def fit(self, X, Y):
        """
        Ajusta un clasificador independiente por cada etiqueta.

        Parámetros
        ----------
        X : array-like, shape (n_samples, n_features)
        Y : array-like, shape (n_samples, n_labels)
        """
        X = np.asarray(X)
        Y = np.asarray(Y)
        n_labels = Y.shape[1]
        self.classifiers_ = []

        for i in range(n_labels):
            # Creamos un clon limpio del clasificador base
            clf = clone(self.base_clf)
            # Ajustamos en la i-ésima columna de Y
            clf.fit(X, Y[:, i])
            self.classifiers_.append(clf)
        return self

    def predict(self, X):
        """
        Predice todas las etiquetas generando un array (n_samples, n_labels) de 0/1.

        Parámetros
        ----------
        X : array-like, shape (n_samples, n_features)

        Devuelve
        -------
        Y_pred : array, shape (n_samples, n_labels)
        """
        X = np.asarray(X)
        # Para cada clasificador, obtenemos su predicción binaria
        preds = [clf.predict(X) for clf in self.classifiers_]
        # Transponer para obtener forma (n_samples, n_labels)
        return np.vstack(preds).T


