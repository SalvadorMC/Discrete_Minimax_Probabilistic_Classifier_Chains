#import uuid
import numpy as np
from sklearn.linear_model import LogisticRegression

from lib.PCC.skmultiflow.meta.classifier_chains import ClassifierChain
from lib.resampling import MLROS


#from functools import lru_cache


# @lru_cache(maxsize=None)  # Use 'lru_cache' for memoization
def P(y, x, cc, payoff=np.prod):
    """Payoff function, P(Y=y|X=x)

    What payoff do we get for predicting y | x, under model cc.

    Parameters
    ----------
    x: input instance
    y: its true labels
    cc: a classifier chain
    payoff: payoff function. Default is np.prod
            example np.prod([0.1, 0.2, 0.3]) = 0.006 (np.prod returns the product of array elements over a given axis.)


    Returns
    -------
    A single number; the payoff of predicting y | x.
    """
    D = len(x)
    # D is the number of features
    L = len(y)
    # L is the number of labels

    p = np.zeros(L)

    # xy is the concatenation of x and y
    # e.g., x = [1, 2, 3], y = [0, 1, 0], xy = [1, 2, 3, 0, 1, 0]
    xy = np.zeros(D + L)

    xy[0:D] = x.copy()

    # For each label j, compute P_j(y_j | x, y_1, ..., y_{j-1})
    for j in range(L):
        # reshape(1,-1) is needed because predict_proba expects a 2D array
        # example: cc.ensemble[j].predict_proba(xy[0:D+j].reshape(1,-1)) = [[0.9, 0.1]]

        P_j = cc.ensemble[j].predict_proba(xy[0 : D + j].reshape(1, -1))[0]
        # e.g., [0.9, 0.1] wrt 0, 1

        xy[D + j] = y[j]  # e.g., 1
        p[j] = P_j[y[j]]
        # e.g., 0.1 or, y[j] = 0 is predicted with probability p[j] = 0.9

    # The more labels we predict incorrectly, the higher the penalty of the payoff
    # p = [0.99055151 0.00709076 0.99999978]
    # y_ [0 1 0]
    # w_ = 0.007
    return payoff(p)


class ProbabilisticClassifierChainBase(ClassifierChain):
    """Probabilistic Classifier Chains for multi-label learning.

    Published as 'PCC'

    Parameters
    ----------
    base_estimator: skmultiflow or sklearn model (default=LogisticRegression)
        This is the ensemble classifier type, each ensemble classifier is going
        to be a copy of the base_estimator.

    order : str (default=None)
        `None` to use default order, 'random' for random order.

    random_state: int, RandomState instance or None, optionalseed used by the random number genera (default=None)
        If int, random_state is the tor;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used by `np.random`.

    Examples
    --------

    TRUE:
    [[1. 0. 1.]
     [1. 1. 0.]
     [0. 0. 0.]
     [1. 1. 0.]]
    vs
    PCC
    [[1. 0. 1.]
     [1. 1. 0.]
     [0. 0. 0.]
     [1. 1. 0.]]
    """

    def __init__(
        self, base_estimator=LogisticRegression(), order=None, random_state=None
    ):
        super().__init__(
            base_estimator=base_estimator, order=order, random_state=random_state
        )
        self.store_key: str | None = None
        self.predicted_store = {}


    def set_store_key(self, key):
        print(f"🐠 - Set store key: {key}")
        self.store_key = key
        self.predicted_store[key] = None  # type: ignore

    def predict_aux(
        self, X, marginal=False, pairwise=False
    ) -> tuple[np.ndarray, np.ndarray, dict]:
        """Predict classes for the passed data.

        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features)
            The set of data samples to predict the labels for.

        Returns
        -------
        A numpy.ndarray with all the predictions for the samples in X.

        Notes
        -----
        Explores all possible branches of the probability tree
        (i.e., all possible 2^L label combinations).
        """
        N, D = X.shape

        Yp = np.zeros((N, self.L))

        P_margin_yi_1 = np.zeros((N, self.L))

        P_pair_wise = np.zeros((N, self.L, self.L + 1))
        P_pair_wise0 = np.zeros((N, 1))
        P_pair_wise1 = np.zeros((N, 1))

        if (
            self.predicted_store is not None
            and self.store_key is not None
            and self.store_key in self.predicted_store
            and self.predicted_store[self.store_key] is not None
        ):
            print(f"🐠 Cached [{self.store_key}]")
            return self.predicted_store[self.store_key]

        #print(f"🐠 Predicting... [{self.store_key}]")

        # for each instance
        for n in range(N):
            w_max = 0.0

            # s is the number of labels that are 1
            s = 0
            # for each and every possible label combination
            # initialize a list of $L$ elements which encode the $L$ marginal probability masses
            # initialize a $L \times (L+1)$ matrix which encodes the pairwise probability masses
            # (i.e., all possible 2^L label combinations) [0, 1, ..., 2^L-1]
            for b in range(2**self.L):
                # put together a label vector
                # e.g., b = 3, self.L = 3, y_ = [0, 0, 1] | b = 5, self.L = 3, y_ = [0, 1, 0]
                y_ = np.array(list(map(int, np.binary_repr(b, width=self.L))))

                # ... and gauge a probability for it (given x)
                w_ = P(y_, X[n], self)

                # All values of y_ are 0
                if np.sum(y_) == 0:
                    P_pair_wise0[n] = w_

                # All values of y_ are 1
                if np.sum(y_) == self.L:
                    P_pair_wise1[n] = w_

                if pairwise:
                    # is number [0-K]
                    s = np.sum(y_)

                if marginal or pairwise:
                    for label_index in range(self.L):
                        if y_[label_index] == 1:
                            P_margin_yi_1[n, label_index] += w_
                            P_pair_wise[n, label_index, s] += w_

                # Use y_ to check which marginal probability masses and pairwise
                # probability masses should be updated (by adding w_)
                # if it performs well, keep it, and record the max
                if w_ > w_max:
                    Yp[n, :] = y_[:].copy()
                    w_max = w_

                # P(y_1 = 1 | X) = P(y_1 = 1 | X, y_2 = 0) * P(y_2 = 0 | X) + P(y_1 = 1 | X, y_2 = 1) * P(y_2 = 1 | X)

        self.predicted_store = {
            self.store_key: (
                Yp,
                P_margin_yi_1,
                {
                    "P_pair_wise": P_pair_wise,
                    "P_pair_wise0": P_pair_wise0,
                    "P_pair_wise1": P_pair_wise1,
                },
            )
        }
        return self.predicted_store[self.store_key]

    def predict(self, X):
        self.predictions, self.P_margin_yi_1, self.P_pair_wise_obj = self.predict_aux(X,marginal=True, pairwise=True)


    def predict_subset(self, X):
        """
        Retorna la combinación de etiquetas que maximiza la probabilidad (Subset).
        Simplemente utiliza predict_aux sin parámetros adicionales.
        """
        return self.predictions
    def predict_hamming(self,X):
        return np.where(self.P_margin_yi_1 > 0.5, 1, 0)
    def predict_f1(self, X, beta=1):
        N, _ = X.shape
        #_, _, P_pair_wise_obj = self.predict_aux(X, pairwise=True)
        P_pair_wise_obj = self.P_pair_wise_obj

        P_pair_wise, P_pair_wise0, P_pair_wise1 = (
            P_pair_wise_obj["P_pair_wise"],
            P_pair_wise_obj["P_pair_wise0"],
            P_pair_wise_obj["P_pair_wise1"],
        )

        # E[0] , E[L-1], E[L]
        P = np.zeros((N, self.L))

        for i in range(N):  # for each instance
            # q_f_measure[i][top_ranked_label][label]
            q_f_measure = np.zeros((self.L, self.L))
            indices_q_f_measure_desc = []

            expectation_values = np.zeros(self.L + 1)

            # line 9 in the algorithm for F-measure
            expectation_value_0 = P_pair_wise0[i][0]

            # rank label = L -> L + 1 top ranked labels
            for top_ranked_label in range(self.L):
                # l = top ranked labels \bar{y}_{(k)} = 1
                for label in range(self.L):  # for each label
                    for s in range(self.L):
                        # for group of vectors with s relevant labels (label = 1)
                        # + 2 because iterate from 1 to L
                        q_f_measure[top_ranked_label][label] += (1 + beta**2) * (
                            P_pair_wise[i][label][s]
                            / (
                                beta**2 * (s + 1) + top_ranked_label + 1
                            )  # revised indices
                        )
                # sort by descending order indices_q_f_measure_desc[top_ranked_label]
                indices_q_f_measure_desc.append(
                    np.argsort(q_f_measure[top_ranked_label])[::-1].tolist()
                )

                # max q_f_measure
                # q_f_measure_max = q_f_measure[i][top_ranked_label][indices_q_f_measure[i][0]]

                # Expectation value at top_ranked_label = sum max q_f_measure from 0 to top_ranked_label
                for i_ in range(top_ranked_label + 1):
                    expectation_values[top_ranked_label] += q_f_measure[
                        top_ranked_label
                    ][int(indices_q_f_measure_desc[top_ranked_label][i_])]

            # Determine ˆy which is ˆyl with the highest E(f (y, ˆyl) where l ∈ [K]0
            # Case 1: Expectation value of 0 > max(expectation_values)
            if expectation_value_0 > np.max(expectation_values):
                P[i] = np.zeros(self.L)
            else:
                # Case 2: Expectation value of 0 <= max(expectation_values)
                # max_expectation_value_index = L_optimal -> optimal top ranked label
                L_optimal_index = np.argmax(expectation_values)
                for _l in range(L_optimal_index + 1):
                    P[i][int(indices_q_f_measure_desc[L_optimal_index][_l])] = 1
        return P


class MLROS_ProbabilisticClassifierChainBase(ProbabilisticClassifierChainBase):
    def fit(self, X, y):
        X_resampled, y_resampled = MLROS(X, y, ratio=0.1, random_state=None)
        super().fit(X_resampled, y_resampled)
        self.L = y.shape[1]
        return self


