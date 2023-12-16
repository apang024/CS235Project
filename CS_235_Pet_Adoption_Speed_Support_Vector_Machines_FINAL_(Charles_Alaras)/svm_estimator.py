# Citations
# Initial gradient descent code provided by Mariam Salloum
# Some implementation details constructed from libsvm: https://www.csie.ntu.edu.tw/~cjlin/papers/libsvm.pdf
# CVXPY implementation derived from: https://stackoverflow.com/questions/67102058/svm-implementation-using-cvxpy
# Some details referenced from: https://domino.ai/blog/fitting-support-vector-machines-quadratic-programming

"""
This is a module to be used as a reference for building other modules
"""
import numpy as np
import cvxpy as cp
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances

# Binary classifier
class SVC:
    kernel = "linear"
    c = 1.0
    w = None
    b = None
    def __init__(self, c=1.0, kernel='linear', gamma='scale', coef0=0.0, degree=-1):
        if c < 0:
            raise Exception("SVCError: Regularization parameter cannot be less than 0")
        self.c = c
        if kernel != "linear":
            raise Exception("SVCError: Unknown kernel function definition")
        self.kernel = kernel
    
    def train_one(self, data, classes):
        # Ensure classes are bound in -1 and 1.
        # Get num features, and num points
        dataPoints, features = data.shape
        w = cp.Variable((features, 1))
        b = cp.Variable()
        xi = cp.Variable((dataPoints, 1))
        objective = cp.Minimize(0.5 * cp.norm(w,2) + self.c * cp.sum(xi))
        constraint = [cp.multiply(classes, ((data @ w) + b)) >= 1 - xi, xi >= 0]

        problem = cp.Problem(objective, constraint)
        problem.solve(solver=cp.ECOS)

        self.w = w.value
        self.b = b.value
        pass

    def predict_one(self, data):
        if self.w is None or self.b is None:
            raise Exception("SVCError: Call train_one() before calling predict_one()")
        return (np.dot(self.w.T, data) + self.b)[0]
        pass

class SVM(ClassifierMixin, BaseEstimator):
    """ Custom implementation of multi-class SVM

    For more information regarding how to build your own classifier, read more
    in the :ref:`User Guide <user_guide>`.

    Parameters
    ----------
    demo_param : str, default='demo'
        A parameter used for demonstation of how to pass and store paramters.

    Attributes
    ----------
    X_ : ndarray, shape (n_samples, n_features)
        The input passed during :meth:`fit`.
    y_ : ndarray, shape (n_samples,)
        The labels passed during :meth:`fit`.
    classes_ : ndarray, shape (n_classes,)
        The classes seen at :meth:`fit`.
    """
    clfs = []
    def __init__(self, C=1.0):
        self.C = C

    def fit(self, X, y):
        """A reference implementation of a fitting function for a classifier.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,)
            The target values. An array of int.

        Returns
        -------
        self : object
            Returns self.
        """
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)

        self.clfs.clear()
        
        coef_ = []
        intercept_ = []
        for i in self.classes_:
            currClasses = np.where(y == i, 1, -1)
            currClasses = np.reshape(currClasses, (currClasses.shape[0], 1))
            currSVC = SVC(self.C)
            currSVC.train_one(X, currClasses)
            self.clfs.append(currSVC)
            coef_.append(currSVC.w)
            intercept_.append(currSVC.b)
            pass
        self.coef_ = np.array(coef_)
        self.intercept_ = np.array(intercept_)
        self.X_ = X
        self.y_ = y
        # Return the classifier
        return self

    def predict(self, X):
        """ A reference implementation of a prediction for a classifier.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y : ndarray, shape (n_samples,)
            The label for each sample is the label of the closest sample
            seen during fit.
        """
        # Check is fit had been called
        check_is_fitted(self, ['X_', 'y_'])

        # Input validation
        X = check_array(X)
        
        y = []
        for i in range(X.shape[0]):
            confidences = []
            for j in range(len(self.clfs)):
                confidences.append(self.clfs[j].predict_one(X[i]))
                pass
            result = np.argmax(np.array(confidences))
            y.append(result)
            
        return np.array(y)
