import numpy as np
import pandas as pd
import os
import sys
import datetime


from time import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
import pickle
from sklearn.metrics import log_loss
import ensembleClassifier as ec
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier

class run_knn(object):

    def load_data(self, data_amount):
        f = open('bottleneck_labels', 'rb')
        self.y_train, self.y_test = pickle.load(f)
        f.close()

        # bottleneck_dir = 'D:\Skule\Fifth year\CSC411\Project/411a3/Shayne/bottleneck/'
        #
        # bottleneck_paths = []
        # bottleneck_labels = []
        # train_bottleneck_paths = []
        # train_bottleneck_labels = []
        # test_bottleneck_paths = []
        # test_bottleneck_labels = []
        #
        # labels = os.listdir(bottleneck_dir)
        # for label in labels:
        #     label_path = bottleneck_dir + label + "/"
        #     l = int(len(os.listdir(label_path))*0.1)
        #     for f in os.listdir(label_path):
        #         bottleneck_paths += [label_path + f]
        #         bottleneck_labels += [label]
        #     train_bottleneck_paths += bottleneck_paths[:-l]
        #     train_bottleneck_labels += bottleneck_labels[:-l]
        #     test_bottleneck_paths += bottleneck_paths[-l:]
        #     test_bottleneck_labels += bottleneck_labels[-l:]
        #
        #     bottleneck_paths = []
        #     bottleneck_labels = []
        #
        # train_bottlenecks = []
        # test_bottlenecks = []
        #
        # for bottleneck_path in train_bottleneck_paths:
        #     with open(bottleneck_path, 'r') as bottleneck_file:
        #         bottleneck_string = bottleneck_file.read()
        #         bottleneck_values = [[float(x) for x in bottleneck_string.split(',')]]
        #     train_bottlenecks += bottleneck_values
        #
        # for bottleneck_path in test_bottleneck_paths:
        #     with open(bottleneck_path, 'r') as bottleneck_file:
        #         bottleneck_string = bottleneck_file.read()
        #         bottleneck_values = [[float(x) for x in bottleneck_string.split(',')]]
        #     test_bottlenecks += bottleneck_values
        #
        # self.X_train = np.array(train_bottlenecks)
        # self.y_train = np.array(train_bottleneck_labels)
        #
        # self.X_test = np.array(test_bottlenecks)
        # self.y_test = np.array(test_bottleneck_labels)
        # print(self.X_train.shape, self.X_test.shape)
        #
        # f = open('bottleneck_labels', 'wb')
        # pickle.dump((self.y_train, self.y_test), f)
        # f.close()

    def pca(self, n_components):
        # PCA step
        # print('Extracting the top %d eigenvectors') % n_components
        # t0 = time()
        # pca = PCA(n_components=n_components, svd_solver='randomized',
        #         whiten=True)
        # pca.fit(self.X_train)
        # print("done in %0.3fs" % (time() - t0))
        #
        # print("Projecting the input data on the eigenvectors orthonormal basis")
        # t0 = time()
        # self.X_train = pca.transform(self.X_train)
        # self.X_test = pca.transform(self.X_test)
        #saving files
        # print("done in %0.3fs" % (time() - t0))
        # f = open('bottleneck_PCA.pckl', 'wb')
        # pickle.dump((self.X_train, self.X_test), f)
        # f.close()
        f = open('bottleneck_PCA.pckl', 'rb')
        X_train, X_test = pickle.load(f)
        self.X_train = X_train[:,range(n_components)]
        self.X_test = X_test[:,range(n_components)]
        f.close()
        print(self.X_train.shape)

    def knn(self):
        print("Training kNN...")
        t0 = time()
        knn = KNeighborsClassifier(n_neighbors=9)
        knn.fit(self.X_train, self.y_train)
        print("done in %0.3fs" % (time() - t0))

        print("Predicting...")
        t0 = time()
        prediction = knn.predict(self.X_test)
        print("done in %0.3fs" % (time() - t0))

        print("Results: ")
        correct = 0.0
        for i in range(0, len(prediction)):
            if self.y_test[i] == prediction[i]:
                correct += 1.0
                symb = "+"
            else:
                symb = ""
            print("%s | %s   %s") % (self.y_test[i], prediction[i], symb)
        
        print '%d out of %d, %d%% accuracy' % (correct, len(prediction), correct/len(prediction)*100)

        pd.crosstab(self.y_test, prediction, rownames=['self.y_test'], colnames=['prediction'])
        
        return prediction

    def svm(self):
        print("Fitting the classifier to the training set")
        t0 = time()
        param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
                    'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
        param_grid = {'C': [1, 3, 5]}
        clf = GridSearchCV(SVC(gamma=0.01,kernel='rbf', class_weight='balanced'), param_grid)
        # clf = SVC(C=3,gamma=0.01,kernel='rbf', class_weight='balanced')
        clf.fit(self.X_train, self.y_train)
        print("done in %0.3fs" % (time() - t0))
        print("Best estimator found by grid search:")
        print(clf.best_estimator_)

        print("Predicting on the test set")
        t0 = time()
        y_pred = clf.predict(self.X_test)
        print("done in %0.3fs" % (time() - t0))
        f = open('SVM_train_full.pckl', 'wb')
        pickle.dump(clf, f)
        f.close()
        print("Results: ")
        correct = 0.0
        for i in range(0,len(y_pred)):
            print("%s | %s") % (self.y_test[i], y_pred[i])
            if self.y_test[i] == y_pred[i]:
                correct += 1.0
        
        print '%d out of %d, %d%% accuracy' % (correct, len(y_pred), correct/len(y_pred)*100)

        return y_pred

    def main(self):
        #linear svm
        self.load_data(7000)
        self.pca(150)
        C=100
        gamma=0.01
        clf = SVC(C=C, gamma=gamma, kernel='rbf')
        clf.fit(self.X_train, self.y_train)
        result = clf.predict(self.X_test)
        count = 0
        for i in range(len(self.y_test)):
            # print("%d: %d | %d" %(i, result[i], y_test[i]))
            if (result[i] == self.y_test[i]):
                count += 1
        print("%d out of %d correct" %(count, len(result)))

        # self.load_data(7000)
        # print('Finished loading data')
        # self.pca(150)
        #
        # # SVM (non-linear)
        # print('start')
        # kernal = 'rbf'
        # C = [1,5, 10, 100, 500]
        # gamma = [1e-1, 1e-2, 1e-3, 1e-4]
        # accuracy_train = np.zeros([len(C), len(gamma)])
        # correct_train = np.zeros([len(C), len(gamma)])
        # ce_train = np.zeros([len(C), len(gamma)])
        # accuracy_test = np.zeros([len(C), len(gamma)])
        # correct_test = np.zeros([len(C), len(gamma)])
        # ce_test = np.zeros([len(C), len(gamma)])
        # for j in range(len(C)):
        #     for k in range(len(gamma)):
        #         clf = SVC(C = C[j], kernel=kernal, gamma=gamma[k], probability=True)
        #         clf.fit(self.X_train, self.y_train)
        #         #test set
        #         result = clf.predict(self.X_test)
        #         result_proba = clf.predict_proba(self.X_test)
        #         count = 0
        #         for i in range(len(self.y_test)):
        #             # print("%d: %d | %d" %(i, result[i], y_test[i]))
        #             if (result[i] == self.y_test[i]):
        #                 count += 1
        #         # print("%d out of %d correct" %(count, len(result)))
        #         accuracy_test[j,k] = count * 100.0 / len(self.y_test)
        #         correct_test[j,k] = count#
        #         ce_test[j, k] = log_loss(self.y_test, result_proba)
        #
        #         # test model on training set itself
        #         result = clf.predict(self.X_train)
        #         result_proba = clf.predict_proba(self.X_train)
        #         count = 0
        #         for i in range(len(self.y_train)):
        #             # print("%d: %d | %d" %(i, result[i], y_test[i]))
        #             if (result[i] == self.y_train[i]):
        #                 count += 1
        #         # print("%d out of %d correct" %(count, len(result)))
        #         accuracy_train[j, k] = count * 100.0 / len(self.y_train)
        #         correct_train[j, k] = count  #
        #         ce_train[j, k] = log_loss(self.y_train, result_proba)
        #         print('Done')

        # #save accuracy
        # f = open('bottleneck_SVC_accuracy.pckl', 'wb')
        # pickle.dump((accuracy_train,correct_train, ce_train, accuracy_test,correct_test, ce_test), f)
        # f.close()

    #---------------------------Ensemble methods---------------------------------
        # f = open('KNN_bag.pckl', 'rb')
        # accuracy,correct, ce = pickle.load(f)
        # f.close()
        # print(accuracy)


        # self.load_data(7000)
        # ensemble = ec.ensembleClassifier(self.X_train, self.y_train, self.X_test, self.y_test)
        # n_neighbors = 9

        #AdaBoostKNN
        #KNN: 9 neighbors, no pca
        #KNN doesn't work with adaboost
        # result = ensemble.adaBoostKNN(300, n_neighbors)
        #
        #BagKNN
        # max_samples = [0.5, 0.7, 0.9, 0.95, 1.0]
        # max_features = [0.5, 0.7, 0.9, 0.95, 1.0]
        #
        # accuracy = np.zeros([len(max_samples),len(max_features)])
        # correct = np.zeros([len(max_samples),len(max_features)])
        # ce = np.zeros([len(max_samples),len(max_features)])
        #
        # for sample in range(len(max_samples)):
        #     print(sample)
        #     for feature in range(len(max_features)):
        #         clf = BaggingClassifier(KNeighborsClassifier(n_neighbors=n_neighbors), max_samples=max_samples[sample],
        #                                     max_features=max_features[feature])
        #         clf.fit(self.X_train, self.y_train)
        #         result = clf.predict(self.X_test)
        #         result_proba = clf.predict_proba(self.X_test)
        #         count = 0
        #         for i in range(len(self.y_test)):
        #             # print("%d: %d | %d" %(i, result[i], y_test[i]))
        #             if (result[i] == self.y_test[i]):
        #                 count += 1
        #         # print("%d out of %d correct" %(count, len(result)))
        #         accuracy[sample,feature] = count * 100.0 / len(self.y_test)
        #         correct[sample,feature] = count#
        #         ce[sample, feature] = log_loss(self.y_test,result_proba)
        # #save results
        # f = open('KNN_bag.pckl', 'wb')
        # pickle.dump((accuracy,correct, ce), f)
        # f.close()

        # #SVM:
        # self.pca(150)#use PCA with 150 compoenents as features
        # C = 3
        # gamma = 0.01
        # ensemble = ec.ensembleClassifier(self.X_train, self.y_train, self.X_test, self.y_test)
        # #Adaboost SVM
        # result = ensemble.adaBoostSVM(C, gamma, 15)

        # #BagSVM
        # max_samples = [0.7, 0.8, 0.9]
        # max_features = [1.0]
        #
        # accuracy = np.zeros([len(max_samples),len(max_features)])
        # correct = np.zeros([len(max_samples),len(max_features)])
        # ce = np.zeros([len(max_samples),len(max_features)])
        #
        # for sample in range(len(max_samples)):
        #     print(sample)
        #     for feature in range(len(max_features)):
        #         clf = BaggingClassifier(SVC(kernel='rbf', C=C, gamma=gamma, probability=True, class_weight='balanced'), max_samples=max_samples[sample],
        #                                     max_features=max_features[feature])
        #         clf.fit(self.X_train, self.y_train)
        #         result = clf.predict(self.X_test)
        #         result_proba = clf.predict_proba(self.X_test)
        #         count = 0
        #         for i in range(len(self.y_test)):
        #             # print("%d: %d | %d" %(i, result[i], y_test[i]))
        #             if (result[i] == self.y_test[i]):
        #                 count += 1
        #         # print("%d out of %d correct" %(count, len(result)))
        #         accuracy[sample,feature] = count * 100.0 / len(self.y_test)
        #         correct[sample,feature] = count#
        #         ce[sample, feature] = log_loss(self.y_test,result_proba)
        #
        # #save results
        # f = open('SVM_bag.pckl', 'wb')
        # pickle.dump((accuracy,correct, ce), f)
        # f.close()




    
if __name__ == '__main__':
    knn_run = run_knn()
    knn_run.main()
