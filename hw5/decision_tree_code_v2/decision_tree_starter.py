from collections import Counter

import io
import numpy as np
from numpy import genfromtxt
import pandas as pd
from pydot import graph_from_dot_data
import scipy.io
from scipy import stats
from sklearn.tree import DecisionTreeClassifier, export_graphviz, plot_tree
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

eps = 1e-5  # a small number


class DecisionTree:

    def __init__(self, max_depth=3, feature_labels=None):
        self.max_depth = max_depth
        self.features = feature_labels
        self.left, self.right = None, None  # for non-leaf nodes
        self.split_idx, self.thresh = None, None  # for non-leaf nodes
        self.data, self.pred = None, None  # for leaf nodes

    @staticmethod
    def entropy(y):
        # TODO implement entropy function
        if y.shape[0] == 0:
            return 0
        p = np.count_nonzero(y) / y.shape[0]
        if p == 0 or p == 1:
            return 0
        return -p * np.log(p) - (1-p) * np.log(1-p)

    @staticmethod
    def information_gain(X, y, thresh):
        # TODO implement information gain function
        before = DecisionTree.entropy(y)
        mask = X < thresh
        lY = y[mask]
        rY = y[~mask]
        p = lY.shape[0] / (lY.shape[0] + rY.shape[0])
        after = p * DecisionTree.entropy(lY) + (1-p) * DecisionTree.entropy(rY)
        return before - after

    def split(self, X, y, idx, thresh):
        X0, idx0, X1, idx1 = self.split_test(X, idx=idx, thresh=thresh)
        y0, y1 = y[idx0], y[idx1]
        return X0, y0, X1, y1

    def split_test(self, X, idx, thresh):
        idx0 = np.where(X[:, idx] < thresh)[0]
        idx1 = np.where(X[:, idx] >= thresh)[0]
        X0, X1 = X[idx0, :], X[idx1, :]
        return X0, idx0, X1, idx1

    def fit(self, X, y):
        if self.max_depth > 0:
            # compute entropy gain for all single-dimension splits,
            # thresholding with a linear interpolation of 10 values
            gains = []
            # The following logic prevents thresholding on exactly the minimum
            # or maximum values, which may not lead to any meaningful node
            # splits.
            thresh = np.array([
                np.linspace(
                    np.min(X[:, i]) + eps, np.max(X[:, i]) - eps, num=10)
                for i in range(X.shape[1])
            ])
            for i in range(X.shape[1]):
                gains.append([
                    self.information_gain(X[:, i], y, t) for t in thresh[i, :]
                ])

            gains = np.nan_to_num(np.array(gains))
            self.split_idx, thresh_idx = np.unravel_index(
                np.argmax(gains), gains.shape)
            self.thresh = thresh[self.split_idx, thresh_idx]
            X0, y0, X1, y1 = self.split(
                X, y, idx=self.split_idx, thresh=self.thresh)
            if X0.size > 0 and X1.size > 0:
                self.left = DecisionTree(
                    max_depth=self.max_depth - 1, feature_labels=self.features)
                self.left.fit(X0, y0)
                self.right = DecisionTree(
                    max_depth=self.max_depth - 1, feature_labels=self.features)
                self.right.fit(X1, y1)
            else:
                self.max_depth = 0
                self.data, self.labels = X, y
                self.pred = stats.mode(y).mode
        else:
            self.data, self.labels = X, y
            self.pred = stats.mode(y).mode
        return self

    def predict(self, X):
        if self.max_depth == 0:
            return self.pred * np.ones(X.shape[0])
        else:
            X0, idx0, X1, idx1 = self.split_test(
                X, idx=self.split_idx, thresh=self.thresh)
            yhat = np.zeros(X.shape[0])
            yhat[idx0] = self.left.predict(X0)
            yhat[idx1] = self.right.predict(X1)
            return yhat

    def __repr__(self):
        if self.max_depth == 0:
            return "%s (%s)" % (self.pred, self.labels.size)
        else:
            return "[%s < %s: %s | %s]" % (self.features[self.split_idx],
                                           self.thresh, self.left.__repr__(),
                                           self.right.__repr__())


class BaggedTrees(BaseEstimator, ClassifierMixin):

    def __init__(self, params=None, n=200):
        if params is None:
            params = {}
        self.params = params
        self.n = n
        self.decision_trees = [
            DecisionTreeClassifier(random_state=i, **self.params)
            for i in range(self.n)
        ]

    def fit(self, X, y):
        # TODO implement function
        for tree in self.decision_trees:
            mask = np.random.choice(X.shape[0], size=X.shape[0])
            x_sample = X[mask,:]
            y_sample = y[mask]
            tree.fit(x_sample, y_sample)
        return self

    def predict(self, X):
        # TODO implement function
        predict = []
        for tree in self.decision_trees:
            predict.append(tree.predict(X))
        predict = np.array(predict)
        return stats.mode(predict, axis=0).mode

class RandomForest(BaggedTrees):

    def __init__(self, params=None, n=200, m=1):
        if params is None:
            params = {}
        # TODO implement function
        super().__init__(params, n)
        self.m = m
        self.selectF = None

    def fit(self, X, y):
        if self.selectF == None:
            print('Initialize selectF for RT')
        else:
            print('Reset selectF for RT')
        self.selectF = []
        for tree in self.decision_trees:
            mask = np.random.choice(X.shape[0], size=X.shape[0])
            feature_mask = np.random.choice(X.shape[1], size = self.m)
            x_sample = X[mask, :][:, feature_mask]
            self.selectF.append(feature_mask)
            y_sample = y[mask]
            tree.fit(x_sample, y_sample)
        return self
    
    def predict(self, X):
        predict = []
        for i in range(0, self.n):
            idx = np.asarray(self.selectF)[i]
            predict.append(self.decision_trees[i].predict(X[:,idx]))
        predict = np.array(predict)
        return stats.mode(predict, axis=0).mode


def preprocess(data, fill_mode=True, min_freq=10, onehot_cols=[]):
    # Temporarily assign -1 to missing data
    data[data == b''] = '-1'

    # Hash the columns (used for handling strings)
    onehot_encoding = []
    onehot_features = []
    for col in onehot_cols:
        counter = Counter(data[:, col])
        for term in counter.most_common():
            if term[0] == b'-1':
                continue
            if term[-1] <= min_freq:
                break
            onehot_features.append(term[0])
            onehot_encoding.append((data[:, col] == term[0]).astype(float))
        data[:, col] = '0'
    onehot_encoding = np.array(onehot_encoding).T
    data = np.hstack(
        [np.array(data, dtype=float),
         np.array(onehot_encoding)])

    # Replace missing data with the mode value. We use the mode instead of
    # the mean or median because this makes more sense for categorical
    # features such as gender or cabin type, which are not ordered.
    if fill_mode:
        for i in range(data.shape[-1]):
            mode = stats.mode(data[((data[:, i] < -1 - eps) +
                                    (data[:, i] > -1 + eps))][:, i]).mode
            data[(data[:, i] > -1 - eps) *
                 (data[:, i] < -1 + eps)][:, i] = mode

    return data, onehot_features


def evaluate(clf, num_splits=3):
    print("Cross validation", cross_val_score(clf, X, y, cv=num_splits))
    if hasattr(clf, "decision_trees"):
        counter = Counter([t.tree_.feature[0] for t in clf.decision_trees])
        first_splits = [
            (features[term[0]], term[1]) for term in counter.most_common()
        ]
        print("First splits", first_splits)


def generate_submission(testing_data, predictions):
    # This code below will generate the predictions.csv file.
    if isinstance(predictions, np.ndarray):
        predictions = predictions.astype(int)
    else:
        predictions = np.array(predictions, dtype=int)
    assert predictions.shape == (len(testing_data),), "Predictions were not the correct shape"
    df = pd.DataFrame({'Category': predictions})
    df.index += 1  # Ensures that the index starts at 1.
    df.to_csv('predictions.csv', index_label='Id')

    # Now download the predictions.csv file to submit.`

def calc_train_accuracy(clf, X, y):
    predict = clf.predict(X)
    assert(predict.shape == y.shape)
    return np.sum(predict == y) / y.shape[0]

if __name__ == "__main__":
    dataset = "titanic"
    params = {
        "max_depth": 5,
        "min_samples_leaf": 10,
    }
    N = 200

    if dataset == "titanic":
        # Load titanic data
        path_train = 'datasets/titanic/titanic_training.csv'
        data = genfromtxt(path_train, delimiter=',', dtype=None)
        path_test = 'datasets/titanic/titanic_testing_data.csv'
        test_data = genfromtxt(path_test, delimiter=',', dtype=None)
        y = data[1:, 0]  # label = survived
        class_names = ["Died", "Survived"]

        labeled_idx = np.where(y != b'')[0]
        y = np.array(y[labeled_idx], dtype=int)
        print("\n\nPart (b): preprocessing the titanic dataset")
        X, onehot_features = preprocess(data[1:, 1:], onehot_cols=[1, 5, 7, 8])
        X = X[labeled_idx, :]
        Z, _ = preprocess(test_data[1:, :], onehot_cols=[1, 5, 7, 8])
        assert X.shape[1] == Z.shape[1]
        features = list(data[0, 1:]) + onehot_features

    elif dataset == "spam":
        features = [
            "pain", "private", "bank", "money", "drug", "spam", "prescription",
            "creative", "height", "featured", "differ", "width", "other",
            "energy", "business", "message", "volumes", "revision", "path",
            "meter", "memo", "planning", "pleased", "record", "out",
            "semicolon", "dollar", "sharp", "exclamation", "parenthesis",
            "square_bracket", "ampersand"
        ]
        assert len(features) == 32

        # Load spam data
        path_train = 'datasets/spam/spam_data.mat'
        data = scipy.io.loadmat(path_train)
        X = data['training_data']
        y = np.squeeze(data['training_labels'])
        Z = data['test_data']
        class_names = ["Ham", "Spam"]

    else:
        raise NotImplementedError("Dataset %s not handled" % dataset)

    print("Features", features)
    print("Train/test size", X.shape, Z.shape)

    print("\n\nPart 0: constant classifier")
    print("Accuracy", 1 - np.sum(y) / y.size)

    # Basic decision tree
    print("\n\nPart (a-b): simplified decision tree")
    # TODO
    DT = DecisionTree(max_depth=params["max_depth"], feature_labels=features)
    DT = DT.fit(X, y)
    print(DT.__repr__)

    # Basic decision tree
    print("\n\nPart (c): sklearn's decision tree")
    # Hint: Take a look at the imports!
    clf = DecisionTreeClassifier(**params) # TODO
    clf = clf.fit(X, y)
    figure = plt.figure(figsize=(12,12))
    plot_tree(clf, feature_names=features, class_names=class_names, filled=True)
    plt.show()
    # TODO
    # Visualizing the tree
    # out = io.StringIO()
    # export_graphviz(
    #     clf, out_file=out, feature_names=features, class_names=class_names)
    # # For OSX, may need the following for dot: brew install gprof2dot
    # graph = graph_from_dot_data(out.getvalue())
    # graph_from_dot_data(out.getvalue())[0].write_pdf("%s-basic-tree.pdf" % dataset)

    # Bagged trees
    print("\n\nPart (d-e): bagged trees")
    # TODO
    BT = BaggedTrees(params=params)
    BT = BT.fit(X, y)

    # Random forest
    print("\n\nPart (f-g): random forest")
    # TODO
    RT = RandomForest(params=params, m=8)
    RT = RT.fit(X, y)

    # Generate csv file of predictions on test data
    # TODO
    print('Start evaluation for ' + dataset)
    print('For Basic Decision Tree:')
    evaluate(clf)
    print('Training Accuracy : ', calc_train_accuracy(clf, X, y))
    print('For Bagging Tree:')
    evaluate(BT)
    print('Training Accuracy : ', calc_train_accuracy(BT, X, y))
    print('For Radom Forest:')
    evaluate(RT)
    print('Training Accuracy : ', calc_train_accuracy(RT, X, y))
    prediction = BT.predict(Z)
    generate_submission(Z, prediction)
