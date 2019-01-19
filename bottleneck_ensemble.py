import numpy as np
import pandas as pd
import os
import sys
import datetime
# import tensorflow as tf

from time import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from collections import Counter
from sklearn.externals import joblib


class run_knn(object):

    def load_data(self):
        bottleneck_paths = []
        bottleneck_labels = []
        train_bottleneck_paths = []
        train_bottleneck_labels = []
        test_bottleneck_paths = []
        test_bottleneck_labels = []

        labels = os.listdir(self.bottleneck_dir)
        for label in labels:
            label_path = self.bottleneck_dir + label + "/"
            for f in os.listdir(label_path):
                bottleneck_paths += [label_path + f]
                bottleneck_labels += [label]
            train_bottleneck_paths += bottleneck_paths
            train_bottleneck_labels += bottleneck_labels

            bottleneck_paths = []
            bottleneck_labels = []

        train_bottlenecks = []

        for bottleneck_path in train_bottleneck_paths:
            with open(bottleneck_path, 'r') as bottleneck_file:
                bottleneck_string = bottleneck_file.read()
                bottleneck_values = [[float(x) for x in bottleneck_string.split(',')]]
            train_bottlenecks += bottleneck_values

        self.X_train = np.array(train_bottlenecks)
        self.y_train = np.array(train_bottleneck_labels)

    def pca(self, n_components):
        # PCA step
        print('Extracting the top %d eigenvectors') % n_components
        t0 = time()
        pca = PCA(n_components=n_components, svd_solver='randomized',
                whiten=True).fit(self.X_train)
        print("done in %0.3fs" % (time() - t0))

        print("Projecting the input data on the eigenvectors orthonormal basis")
        t0 = time()
        self.X_train = pca.transform(self.X_train)
        self.X_test = pca.transform(self.X_test)
        print("done in %0.3fs" % (time() - t0))

    def knn(self, neighbours):
        print("Training kNN...")
        t0 = time()
        knn = KNeighborsClassifier(n_neighbors=neighbours)
        knn.fit(self.X_train, self.y_train)
        print("done in %0.3fs" % (time() - t0))

        print("Predicting...")
        t0 = time()
        prediction = knn.predict(self.X_test)
        print("done in %0.3fs" % (time() - t0))
        
        return prediction

    def svm(self):
        try: 
            clf = joblib.load('svm.pkl')
        except:
            print("Fitting the classifier to the training set")
            t0 = time()
            param_grid = {'C': [1, 10, 100, 1000],
                    'gamma': [1e-2, 1e-3, 1e-4, 1e-5], }
            clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
            clf = clf.fit(self.X_train, self.y_train)
            print("done in %0.3fs" % (time() - t0))
            print("Best estimator found by grid search:")
            print(clf.best_estimator_)
            joblib.dump(clf, 'svm.pkl') 

        print("Predicting on the test set")
        t0 = time()
        y_pred = clf.predict(self.X_test)
        print("done in %0.3fs" % (time() - t0))

        return y_pred


    def create_graph(self):
        """Creates a graph from saved GraphDef file and returns a saver."""
        # Creates graph from saved graph_def.pb.
        with tf.gfile.FastGFile(self.modelFullPath, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(graph_def, name='')

    def batch_pool3_features(self):
        self.create_graph()
        with tf.Session() as sess:
            pool3 = sess.graph.get_tensor_by_name('pool_3:0')
            X_pool3 = []
            i = 1
            for image in self.testing_names:
                print 'Iteration %i' % i
                imagePath = self.test_path + image
                print imagePath
                image_data = tf.gfile.FastGFile(imagePath, 'rb').read()
                pool3_features = sess.run(pool3, {'DecodeJpeg/contents:0': image_data})
                X_pool3.append(np.squeeze(pool3_features))
                i += 1
        return np.array(X_pool3)

    def run_inference_on_image_modified(self, testing_names):
        # Creates graph from saved GraphDef.
        self.create_graph()
        i = 1
        total = len(testing_names)
        prediction = []
        with tf.Session() as sess:
            for image in testing_names:
                imagePath = self.test_path + image
                image_data = tf.gfile.FastGFile(imagePath, 'rb').read()
                softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
                predictions = sess.run(softmax_tensor,
                                    {'DecodeJpeg/contents:0': image_data})
                predictions = np.squeeze(predictions)
                f = open(self.labelsFullPath, 'rb')
                lines = f.readlines()
                labels = [str(w).replace("\n", "") for w in lines]
                prediction += labels[np.argmax(predictions)]
                i += 1
                print i
        return prediction

    def adaBoost(self):
        print('AdaBoost')
        X_train = self.X_train
        y_train = self.y_train
        X_test = self.X_test
        real_test_errors = []

        bdt_real = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2), n_estimators=100, learning_rate=1)
        bdt_real.fit(X_train, y_train)
        for real_test_predict in bdt_real.staged_predict(X_test):
            real_test_errors.append(1. - accuracy_score(real_test_predict, y_test))
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

        # prevent overlapping y-axis labels
        plt.subplots_adjust(wspace=0.25)
        plt.show()

        return 1-min(real_test_errors)

    def main(self):
        self.test_path = '/Users/jfl/Documents/CSC2515/A3/test/'
        self.bottleneck_dir = '/tmp/bottleneck/'
        self.modelFullPath = '/tmp/output_graph.pb'
        self.labelsFullPath = '/tmp/output_labels.txt'

        print 'START'
        t0 = time()
        self.testing_names = os.listdir(self.test_path)
        print 'Getting Softmax predictions...'
        sm_results = self.run_inference_on_image_modified(self.testing_names)
        print("done in %0.3fs" % (time() - t0))
        
        t0 = time()
        print 'Extracting features from test set...'
        self.load_data()
        self.X_test = self.batch_pool3_features()
        print("done in %0.3fs" % (time() - t0))
        
        t0 = time()
        print 'Getting KNN predictions...'
        knn_results = self.knn(9)
        print("done in %0.3fs" % (time() - t0))

        t0 = time()
        print 'Performing PCA on features...'
        self.pca(100)
        print("done in %0.3fs" % (time() - t0))
        
        t0 = time()
        print 'Getting SVM predictions...'
        svm_results = self.svm()
        print("done in %0.3fs" % (time() - t0))

        #-------------------------------Ensemble methods---------------------
        t0 = time()
        print 'Getting SVM predictions...'
        ada_results = self.adaBoost()
        print("done in %0.3fs" % (time() - t0))

        t0 = time()
        print 'Getting SVM predictions...'
        ada_results = self.adaBoost()
        print("done in %0.3fs" % (time() - t0))



        #------------------------------End ensemble methods------------------

        voted_predictions = [['id', 'prediction']]
        print np.array(cnn_results).shape
        print np.array(svm_results).shape
        print np.array(knn_results).shape
        for i in range(0,len(self.X_test)):
            vote = Counter([sm_results[i], knn_results[i], svm_results[i]]).most_common(1)[0][0]
            if cnn_results[i] != knn_results[i] or cnn_results[i] != svm_results[i]:
                print('%d: %s | sm: %s, knn: %s, svm: %s') % (i, vote, sm_results[i], knn_results[i], svm_results[i])
            voted_predictions += [[i+1, vote]]

        for a in range(i+2, 2971):
            voted_predictions += [[a,0]]
            a += 1
        np.savetxt("predictions.csv", np.array(voted_predictions), delimiter=",", fmt="%s")        
        
if __name__ == '__main__':
    knn_run = run_knn()
    knn_run.main()
