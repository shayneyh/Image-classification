import numpy as np
import pandas as pd
import os
import sys
import datetime

from time import time
from sklearn.neighbors import KNeighborsClassifier
#from sklearn.model_selection import GridSearchCV
#from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from utils import *
from sklearn.decomposition import PCA
from sklearn.svm import SVC

class run_knn(object):
    def create_array(self, set_type):
        img_dir = "./" + set_type + "/"
        images = [img_dir + f for f in os.listdir(img_dir)]

        data = []
        tot_images = len(images)
        i = 0.0
        for image in images:
            sys.stdout.flush()
            img = img_to_matrix(image)
            img = flatten_image(img)
            data.append(img)

            i += 1.0
            progress = int(i/tot_images*100)
            sys.stdout.write('\r[{0}] {1}%'.format('#'*(progress/10) + ' '*(10 - progress/10), progress))

        data = np.array(data)

        np.savez(set_type, data)

    def load_data(self, data_amount):
        try:
            npzfile = np.load('./train.npz')
            X = npzfile['arr_0']
            y = np.recfromcsv('./train.csv', delimiter=',', names=True)
        except:
            raw_input('Press ENTER to convert the training jpg images to an array: ')
            self.create_array('test')

        try:
            npzfile = np.load('./test.npz')
            valid_data = npzfile['arr_0']
        except:
            raw_input('Press ENTER to convert the test jpg images to an array: ')
            self.create_array('test')
        
        X = X[:data_amount, :]
        y = y['label'][:data_amount]

        # split training and test sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=0.25, random_state=42)

    def pca(self, n_components):
        # PCA step
        print('Extracting the top %d eigenvectors') % n_components
        t0 = time()
        pca = PCA(n_components=n_components, svd_solver='randomized',whiten=True).fit(self.X_train)
        print("done in %0.3fs" % (time() - t0))

        print("Projecting the input data on the eigenvectors orthonormal basis")
        t0 = time()
        self.X_train = pca.transform(self.X_train)
        self.X_test = pca.transform(self.X_test)
        print("done in %0.3fs" % (time() - t0))

    def knn(self):
        print("Training kNN...")
        t0 = time()
        knn = KNeighborsClassifier()
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
            print("%d | %d   %s") % (self.y_test[i], prediction[i], symb)
        
        print '%d out of %d, %d%% accuracy' % (correct, len(prediction), correct/len(prediction)*100)

        pd.crosstab(self.y_test, prediction, rownames=['self.y_test'], colnames=['prediction'])
    
    def svm(self):
        print("Fitting the classifier to the training set")
        t0 = time()
        param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
                    'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
        clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
        clf = clf.fit(self.X_train, self.y_train)
        print("done in %0.3fs" % (time() - t0))
        print("Best estimator found by grid search:")
        print(clf.best_estimator_)

        print("Predicting on the test set")
        t0 = time()
        y_pred = clf.predict(self.X_test)
        print("done in %0.3fs" % (time() - t0))

        print("Results: ")
        correct = 0.0
        for i in y_pred:
            print("%d | %d") % (self.y_test[i], y_pred[i])
            if self.y_test[i] == y_pred[i]:
                correct += 1.0
        
        print '%d out of %d, %d%% accuracy' % (correct, len(y_pred), correct/len(y_pred)*100)

    def main(self):
        self.load_data(7000)
        self.pca(100)
        self.knn()
    
if __name__ == '__main__':
    knn_run = run_knn()
    knn_run.main()
