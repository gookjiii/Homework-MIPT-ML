import numpy as np
from sklearn.base import BaseEstimator


def entropy(y):
    """
    Computes entropy of the provided distribution. Use log(value + eps) for numerical stability

    Parameters
    ----------
    y : np.array of type float with shape (n_objects, n_classes)
        One-hot representation of class labels for corresponding subset

    Returns
    -------
    float
        Entropy of the provided subset
    """
    EPS = 0.0005

    # YOUR CODE HERE
    probs = np.sum(y, axis=0) / y.shape[0]
    return - np.sum(probs * np.log(probs + EPS))


def gini(y):
    """
    Computes the Gini impurity of the provided distribution

    Parameters
    ----------
    y : np.array of type float with shape (n_objects, n_classes)
        One-hot representation of class labels for corresponding subset

    Returns
    -------
    float
        Gini impurity of the provided subset
    """

    # YOUR CODE HERE
    probs = np.sum(y, axis=0) / y.shape[0]
    return 1 - np.sum(probs ** 2)


def variance(y):
    """
    Computes the variance the provided target values subset

    Parameters
    ----------
    y : np.array of type float with shape (n_objects, 1)
        Target values vector

    Returns
    -------
    float
        Variance of the provided target vector
    """

    # YOUR CODE HERE
    return np.mean((y - np.mean(y)) ** 2)


def mad_median(y):
    """
    Computes the mean absolute deviation from the median in the
    provided target values subset

    Parameters
    ----------
    y : np.array of type float with shape (n_objects, 1)
        Target values vector

    Returns
    -------
    float
        Mean absolute deviation from the median in the provided vector
    """

    # YOUR CODE HERE

    return np.mean(np.abs(y - np.median(y)))


def one_hot_encode(n_classes, y):
    y_one_hot = np.zeros((len(y), n_classes), dtype=float)
    y_one_hot[np.arange(len(y)), y.astype(int)[:, 0]] = 1.
    return y_one_hot


def one_hot_decode(y_one_hot):
    return y_one_hot.argmax(axis=1)[:, None]


class Node:
    """
    This class is provided "as is" and it is not mandatory to it use in your code.
    """

    def __init__(self, feature_index, threshold, proba=0):
        self.feature_index = feature_index
        self.value = threshold
        self.proba = proba
        self.is_leaf = False
        self.label = None
        self.left_child = None
        self.right_child = None


class DecisionTree(BaseEstimator):
    all_criterions = {
        'gini': (gini, True),  # (criterion, classification flag)
        'entropy': (entropy, True),
        'variance': (variance, False),
        'mad_median': (mad_median, False)
    }

    def __init__(self, n_classes=None, max_depth=np.inf, min_samples_split=2,
                 criterion_name='gini', debug=False):

        assert criterion_name in self.all_criterions.keys(), 'Criterion name must be on of the following: {}'.format(
            self.all_criterions.keys())

        self.n_classes = n_classes
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion_name = criterion_name

        self.depth = 0
        self.root = None  # Use the Node class to initialize it later
        self.debug = debug

    def make_split(self, feature_index, threshold, X_subset, y_subset):
        """
        Makes split of the provided data subset and target values using provided feature and threshold

        Parameters
        ----------
        feature_index : int
            Index of feature to make split with

        threshold : float
            Threshold value to perform split

        X_subset : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the selected subset

        y_subset : np.array of type float with shape (n_objects, n_classes) in classification
                   (n_objects, 1) in regression
            One-hot representation of class labels for corresponding subset

        Returns
        -------
        (X_left, y_left) : tuple of np.arrays of same type as input X_subset and y_subset
            Part of the providev subset where selected feature x^j < threshold
        (X_right, y_right) : tuple of np.arrays of same type as input X_subset and y_subset
            Part of the providev subset where selected feature x^j >= threshold
        """

        # YOUR CODE HERE
        X_left = X_subset[X_subset[:, feature_index] < threshold]
        X_right = X_subset[X_subset[:, feature_index] >= threshold]

        y_left, y_right = self.make_split_only_y(feature_index, threshold, X_subset, y_subset)

        return (X_left, y_left), (X_right, y_right)

    def make_split_only_y(self, feature_index, threshold, X_subset, y_subset):
        """
        Split only target values into two subsets with specified feature and threshold

        Parameters
        ----------
        feature_index : int
            Index of feature to make split with

        threshold : float
            Threshold value to perform split

        X_subset : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the selected subset

        y_subset : np.array of type float with shape (n_objects, n_classes) in classification
                   (n_objects, 1) in regression
            One-hot representation of class labels for corresponding subset

        Returns
        -------
        y_left : np.array of type float with shape (n_objects_left, n_classes) in classification
                   (n_objects, 1) in regression
            Part of the provided subset where selected feature x^j < threshold

        y_right : np.array of type float with shape (n_objects_right, n_classes) in classification
                   (n_objects, 1) in regression
            Part of the provided subset where selected feature x^j >= threshold
        """

        # YOUR CODE HERE
        y_left = y_subset[X_subset[:, feature_index] < threshold]
        y_right = y_subset[X_subset[:, feature_index] >= threshold]

        return y_left, y_right

    def choose_best_split(self, X_subset, y_subset):
        """
        Greedily select the best feature and best threshold w.r.t. selected criterion

        Parameters
        ----------
        X_subset : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the selected subset

        y_subset : np.array of type float with shape (n_objects, n_classes) in classification
                   (n_objects, 1) in regression
            One-hot representation of class labels or target values for corresponding subset

        Returns
        -------
        feature_index : int
            Index of feature to make split with

        threshold : float
            Threshold value to perform split

        """
        # YOUR CODE HERE
        current_criterion, is_classification = self.all_criterions[self.criterion_name]
        H_lst = []

        for feature_idx in range(X_subset.shape[1]):
            threshold_lst = sorted(list(set(X_subset[:, feature_idx])))
            if len(threshold_lst) <= 1:
                continue

            for threshold_idx in range(len(threshold_lst) - 1):
                threshold = (threshold_lst[threshold_idx] + threshold_lst[threshold_idx + 1]) / 2
                H = current_criterion(y_subset)
                (X_left, y_left), (X_right, y_right) = self.make_split(feature_idx, threshold, X_subset, y_subset)
                _, is_classification = self.all_criterions[self.criterion_name]

                if is_classification:
                    left_unique_labels = len(set(np.argmax(y_left, axis=1)))
                    right_unique_labels = len(set(np.argmax(y_right, axis=1)))
                else:
                    left_unique_labels = len(set(y_left.reshape(-1)))
                    right_unique_labels = len(set(y_right.reshape(-1)))

                if left_unique_labels == 0 or right_unique_labels == 0:
                    continue
                left_criterion = current_criterion(y_left)
                right_criterion = current_criterion(y_right)

                H -= X_left.shape[0] / X_subset.shape[0] * left_criterion
                H -= X_right.shape[0] / X_subset.shape[0] * right_criterion
                H_lst.append((H, feature_idx, threshold, X_left.shape[0], X_right.shape[0]))

        best_values = sorted(H_lst, key=(lambda x: x[0]))[-1]
        feature_index = best_values[1]
        threshold = best_values[2]

        return (feature_index, threshold)

    def make_tree(self, X_subset, y_subset, current_depth = 1):
        """
        Recursively builds the tree

        Parameters
        ----------
        X_subset : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the selected subset

        y_subset : np.array of type float with shape (n_objects, n_classes) in classification
                   (n_objects, 1) in regression
            One-hot representation of class labels or target values for corresponding subset

        Returns
        -------
        root_node : Node class instance
            Node of the root of the fitted tree
        """

        # YOUR CODE HERE
        _, is_classification = self.all_criterions[self.criterion_name]
        if is_classification:
            unique_labels = len(set(np.argmax(y_subset, axis=1)))
        else:
            unique_labels = len(set(y_subset.reshape(-1)))

        if X_subset.shape[0] <= self.min_samples_split or unique_labels == 1 or current_depth >= (self.max_depth):
            root_node = Node(None, None)
            root_node.is_leaf = True

            if self.classification:
                counts = np.argmax(y_subset, axis=1)
                probas = np.zeros((self.n_classes))
                for i in counts:
                    probas[i] += 1
                probas /= y_subset.shape[0]
                root_node.label = np.argmax(probas)
                root_node.proba = probas
            else:
                root_node.label = np.mean(y_subset)
                root_node.proba = None

            return root_node

        feature_idx, threshold = self.choose_best_split(X_subset, y_subset)
        (X_left, y_left), (X_right, y_right) = self.make_split(feature_idx, threshold, X_subset, y_subset)
        root_node = Node(feature_idx, threshold)

        root_node.left_child = self.make_tree(X_left, y_left, current_depth + 1)
        root_node.right_child = self.make_tree(X_right, y_right, current_depth + 1)

        return root_node

    def fit(self, X, y):
        """
        Fit the model from scratch using the provided data

        Parameters
        ----------
        X : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the data to train on

        y : np.array of type int with shape (n_objects, 1) in classification
                   of type float with shape (n_objects, 1) in regression
            Column vector of class labels in classification or target values in regression

        """
        assert len(y.shape) == 2 and len(y) == len(X), 'Wrong y shape'
        self.criterion, self.classification = self.all_criterions[self.criterion_name]
        if self.classification:
            if self.n_classes is None:
                self.n_classes = len(np.unique(y))
            y = one_hot_encode(self.n_classes, y)

        self.root = self.make_tree(X, y)

    def predict(self, X):
        """
        Predict the target value or class label  the model from scratch using the provided data

        Parameters
        ----------
        X : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the data the predictions should be provided for

        Returns
        -------
        y_predicted : np.array of type int with shape (n_objects, 1) in classification
                   (n_objects, 1) in regression
            Column vector of class labels in classification or target values in regression

        """

        # YOUR CODE HERE
        y_predicted = np.zeros((X.shape[0], 1))

        for obj_idx in range(X.shape[0]):
            node = self.root
            while not node.is_leaf:
                feature_idx = node.feature_index
                threshold = node.value

                if X[obj_idx, feature_idx] < threshold:
                    node = node.left_child
                else:
                    node = node.right_child

            y_predicted[obj_idx] = node.label

        return y_predicted

    def predict_proba(self, X):
        """
        Only for classification
        Predict the class probabilities using the provided data

        Parameters
        ----------
        X : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the data the predictions should be provided for

        Returns
        -------
        y_predicted_probs : np.array of type float with shape (n_objects, n_classes)
            Probabilities of each class for the provided objects

        """
        assert self.classification, 'Available only for classification problem'

        # YOUR CODE HERE
        y_predicted = np.zeros((X.shape[0], self.n_classes))

        for obj_idx in range(X.shape[0]):
            node = self.root
            while not node.is_leaf:
                feature_idx = node.feature_index
                threshold = node.value

                if X[obj_idx, feature_idx] < threshold:
                    node = node.left_child
                else:
                    node = node.right_child

            y_predicted[obj_idx, :] = node.proba

        return y_predicted

        return y_predicted_probs
