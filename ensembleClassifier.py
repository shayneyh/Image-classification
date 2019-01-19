from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import numpy as np

import matplotlib.pyplot as plt
import datetime
import os
import sys


class ensembleClassifier(object):
    def __init__(self,X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def adaBoostKNN(self, n_estimators, n_neighbors):
        #adaBoost with decision tree
        X_train = self.X_train
        y_train = self.y_train
        X_test = self.X_test
        y_test = self.y_test
        real_test_errors = []
        clf = KNeighborsClassifier(n_neighbors=n_neighbors)
        clf.sample_weight = np.ones(len(y_train))/len(y_train)
        bdt_real = AdaBoostClassifier(clf, n_estimators=n_estimators, learning_rate=1)

        bdt_real.fit(X_train, y_train)
        min_error = 1.0
        count = 0
        for real_test_predict in bdt_real.staged_predict(X_test):
            error = 1.0 - accuracy_score(real_test_predict, y_test)
            if error < min_error:
                min_error = error
                min_error_pred = real_test_predict
                min_error_n = count
            real_test_errors.append(error)
            count += 1
        n_trees_real = len(bdt_real)
        real_estimator_errors = bdt_real.estimator_errors_[:n_trees_real]

        plt.figure(figsize=(15, 5))

        plt.subplot(1, 2, 1)
        plt.plot(range(1, n_trees_real + 1),
                 real_test_errors, c='black',
                 linestyle='dashed', label='SAMME.R')
        plt.legend()
        # plt.ylim(0.18, 0.62)
        plt.ylabel('Test Error')
        plt.xlabel('Number of Trees')

        plt.subplot(1, 2, 2)
        plt.plot(range(1, n_trees_real + 1), real_estimator_errors,
                 "r", label='SAMME.R', alpha=.5)
        plt.legend()
        plt.ylabel('Error')
        plt.xlabel('Number of Trees')
        plt.ylim((.2, real_estimator_errors.max() * 1.2))
        plt.xlim((-20, len(bdt_real) + 20))

        # prevent overlapping y-axis y_train
        plt.subplots_adjust(wspace=0.25)
        plt.show()

        return min_error_pred, min_error_n

    def adaBoostSVM(self, C, gamma, n_estimators):
        #adaBoost with SVM
        X_train = self.X_train
        y_train = self.y_train
        X_test = self.X_test
        y_test = self.y_test
        real_test_errors = []

        bdt_real = AdaBoostClassifier(SVC(C=C, gamma=gamma), n_estimators=n_estimators, learning_rate=1, algorithm='SAMME')

        bdt_real.fit(X_train, y_train)
        min_error = 1.0
        count = 0
        for real_test_predict in bdt_real.staged_predict(X_test):
            error = 1.0 - accuracy_score(real_test_predict, y_test)
            if error < min_error:
                min_error = error
                min_error_pred = real_test_predict
                min_error_n = count
            real_test_errors.append(error)
            count += 1
        n_trees_real = len(bdt_real)
        real_estimator_errors = bdt_real.estimator_errors_[:n_trees_real]

        plt.figure(figsize=(15, 5))

        plt.subplot(1, 2, 1)
        plt.plot(range(1, n_trees_real + 1),
                 real_test_errors, c='black',
                 linestyle='dashed', label='SAMME')
        plt.legend()
        # plt.ylim(0.18, 0.62)
        plt.ylabel('Test Error')
        plt.xlabel('Number of Trees')

        plt.subplot(1, 2, 2)
        plt.plot(range(1, n_trees_real + 1), real_estimator_errors,
                 "r", label='SAMME', alpha=.5)
        plt.legend()
        plt.ylabel('Error')
        plt.xlabel('Number of Trees')
        plt.ylim((.2, real_estimator_errors.max() * 1.2))
        plt.xlim((-20, len(bdt_real) + 20))

        # prevent overlapping y-axis y_train
        plt.subplots_adjust(wspace=0.25)
        plt.show()

        return min_error_pred, min_error_n

    def gradientBoost(self,n_estimators, max_depth):
        #gradientBoost
        X_train = self.X_train
        y_train = self.y_train
        X_test = self.X_test
        y_test = self.y_test
        clf = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=1.0, max_depth=max_depth, random_state=0)
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)
        count = 0
        for i in range(700):
            # print("%d: %d | %d" %(i, result[i], y_test[i]))
            if (pred[i] == y_test[i]):
                count += 1
        print("%d out of %d correct" %(count, len(pred)))
        return pred

    def bagKNN(self, n_neighbors, max_samples, max_features):
        X_train = self.X_train
        y_train = self.y_train
        X_test = self.X_test
        y_test = self.y_test
        bagging = BaggingClassifier(KNeighborsClassifier(n_neighbors=n_neighbors),max_samples = max_samples, max_features = max_features)
        bagging.fit(X_train, y_train)
        pred = bagging.predict(X_test)
        return pred

    def bagSVM(self, C, gamma, max_samples, max_features):
        X_train = self.X_train
        y_train = self.y_train
        X_test = self.X_test
        y_test = self.y_test
        bagging = BaggingClassifier(SVC(C=C, gamma=gamma), max_samples=max_samples, max_features=max_features)
        bagging.fit(X_train, y_train)
        pred = bagging.predict(X_test)
        return pred

    def randomForest(self, n_estimators):
        X_train = self.X_train
        y_train = self.y_train
        X_test = self.X_test
        y_test = self.y_test
        clf = RandomForestClassifier(n_estimators=n_estimators)
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)
        return pred

    def main(self):
        print('AdaBoostDecisionTree')
        #adaBoostDT(self, n_estimators, max_depth)
        param_grid = {'n_estimators': [50], 'max_depth': [True, False]}
        ada_result, min_error_n = self.adaBoostKNN(param_grid)

        print('AdaBoostSVC')
        # def adaBoostSVM(self, C, gamma, n_estimators)
        adaSVM_result = self.adaBoostSVM(3, 0.01, 50)

        print('GradientBoost')
        #gradientBoost(self,n_estimators, max_depth)
        grad_result = self.gradientBoost(200, 2)

        print('BagKNN')
        #bagKNN(self, n_neighbors, max_samples, max_features)
        bagKNN_result = self.bagKNN(9, 0.5, 0.5)

        print('BagSVM')
        #bagSVM(self, C, gamma, max_samples, max_features)
        bagSVM_result = self.bagSVM(3, 0.01, 0.5, 0.5)

        print('Random Forest')
        #randomForest(self, n_estimators)
        randForest_result = self.randomForest(100)




