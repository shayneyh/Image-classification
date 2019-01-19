#!/usr/local/bin/python2.7


import argparse as ap
import cv2
import imutils
import numpy as np
import os
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.externals import joblib
from scipy.cluster.vq import *
from sklearn.preprocessing import StandardScaler
import pickle
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import ensembleClassifier as ec
from sklearn.metrics import log_loss

# train_path = 'D:\Skule\Fifth year\CSC411\Project/411a3/train/'
# y = np.recfromcsv('D:\Skule\Fifth year\CSC411\Project/411a3/train.csv', delimiter=',', names=True)
#
# for i in range(7):
#     image_paths = []
#     y_temp = y[(i * 1000):(i * 1000 + 1000)]
#     image_classes = [x[1] for x in y_temp]
#     for training_name in y_temp:
#         class_path = train_path + '%0*d' % (5,training_name[0]) + '.jpg'
#         image_paths.append(class_path)
#
#     # Create feature extraction and keypoint detector objects
#     fea_det = cv2.FeatureDetector_create("SIFT")
#     des_ext = cv2.DescriptorExtractor_create("SIFT")
#
#     # List where all the descriptors are stored
#     des_list = []
#     for image_path in image_paths:
#         im = cv2.imread(image_path)
#         # normal SIFT
#         # kpts = fea_det.detect(im)
#         # kpts, des = des_ext.compute(im, kpts)
#         # if des == None:
#         #     des = np.zeros([1, 128])
#
#         #dense SIFT (http://stackoverflow.com/questions/33120951/compute-dense-sift-features-in-opencv-3-0/33702400#33702400)
#         gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
#         step_size = 8
#         kpts = [cv2.KeyPoint(a, b, step_size) for b in range(0, gray.shape[0], step_size)
#               for a in range(0, gray.shape[1], step_size)]
#         kpts, des = des_ext.compute(im, kpts)
#
#         des_list.append((image_path, des))
#
#     # Stack all the descriptors vertically in a numpy array
#     descriptors = des_list[0][1]
#     for image_path, descriptor in des_list[1:]:
#         descriptors = np.vstack((descriptors, descriptor))
#
#     f = open('denseSIFT_train' + str(i) + '.pckl', 'wb')
#     pickle.dump(descriptors, f)
#     f.close()
#
#     print(descriptors.shape)

# descriptors = []
# for i in range(7):
#     f = open('SIFT features/denseSIFT_train' + str(i) + '.pckl', 'rb')
#     object = pickle.load(f)
#     if i == 0:
#         descriptors = object
#     else:
#         descriptors = np.append(descriptors, object, axis=0)
#     print(descriptors.shape)
#     f.close()
#
# descriptors_new = np.empty([7000, 256*128]).astype(int)
# for i in range(7000):
#     # descriptors_new.append(descriptors[i:(i+256),:].flatten())
#     ind = i*256
#     descriptors_new[i,:] = descriptors[ind:(ind + 256), :].flatten()
#
# print(descriptors_new.shape)
#------------------------
# #separate into training and validation
# y = np.recfromcsv('D:\Skule\Fifth year\CSC411\Project/411a3/train.csv', delimiter=',', names=True)
# image_classes = [x[1] for x in y]
# image_names = np.array([x[0] for x in y])
# feaures_train = []
# feaures_val = []
#
# for i in range(1, 9):
#     temp_index = np.squeeze([x for x, a in enumerate(y) if a[1] == i])
#     train_index = temp_index[0:round(len(temp_index) * 0.9)]
#     feaures_train = np.append(feaures_train,train_index)
#
#     val_index = temp_index[round(len(temp_index) * 0.9):len(temp_index)]
#     feaures_val = np.append(feaures_val,val_index)
#
# feaures_train = feaures_train.astype(int)
# feaures_val = feaures_val.astype(int)
# print(len(feaures_train))
# print(len(feaures_val))

##seprate training and validation on bottleneck features
# for i in range(len(feaures_train)):
#     bottleneck_path = 'D:\Skule\Fifth year\CSC411\Project/411a3/bottleneck/' + str(image_classes[i]) + '/' + '%0*d' % (5,i) + '.jpg'
#     bottleneck_file = open(bottleneck_path, 'r')
#     bottleneck_string = bottleneck_file.read()
#     bottleneck_values = [[float(x) for x in bottleneck_string.split(',')]]
#-----------------------
##separate training and validation on SIFT
# train_descriptors = descriptors_new[feaures_train,:]
# val_descriptors = descriptors_new[feaures_val,:]
# print(train_descriptors.shape)
# #pca
# n_components = 1000
# ipca = PCA(n_components=n_components, whiten=True, svd_solver='randomized')
# ipca.fit(train_descriptors)
# train_pca = ipca.transform(train_descriptors)
# val_pca = ipca.transform(val_descriptors)
# f = open('denseSIFT_pca.pckl', 'wb')
# pickle.dump((train_pca,val_pca), f)
# f.close()
# #save pca values
# print('saving...')

# f = open('denseSIFT_pca_SVM_accuracy.pckl', 'rb')
# accuracy, correct = pickle.load(f)
# f.close()

#open PCA for SIFT
f = open('denseSIFT_pca.pckl', 'rb')
train_pca, val_pca = pickle.load(f)
f.close()

y = np.recfromcsv('D:\Skule\Fifth year\CSC411\Project/411a3/train.csv', delimiter=',', names=True)
image_classes = [x[1] for x in y]
image_names = np.array([x[0] for x in y])
feaures_train = []
feaures_val = []

for i in range(1, 9):
    temp_index = np.squeeze([x for x, a in enumerate(y) if a[1] == i])
    train_index = temp_index[0:round(len(temp_index) * 0.9)]
    feaures_train = np.append(feaures_train,train_index)

    val_index = temp_index[round(len(temp_index) * 0.9):len(temp_index)]
    feaures_val = np.append(feaures_val,val_index)

feaures_train = feaures_train.astype(int)
feaures_val = feaures_val.astype(int)
y_train = []
y_test = []
for ind in feaures_train:
    y_train.append(image_classes[ind])
for ind in feaures_val:
    y_test.append(image_classes[ind])
#
# num_pca = 50
# X_train = train_pca[:,range(num_pca)]
# X_test = val_pca[:,range(num_pca)]


#------------------------------
#adaboost
# print('AdaBoost')
# ensemble = ec.ensembleClassifier(X_train, y_train, X_test, y_test)
# paramGrid = {'n_estimators': 10, 'max_depth':[2,3]}
# min_error_pred, min_error_n = GridSearchCV(ensemble.adaBoostDT(paramGrid))
# print()
# # print(min_error_pred)
# print(min_error_n)
# count = 0
# for i in range(700):
#     # print("%d: %d | %d" %(i, result[i], y_test[i]))
#     if (min_error_pred[i] == y_test[i]):
#         count += 1
# print("%d out of %d correct" %(count, len(min_error_pred)))

#------------------------------------------
#gradientboost
# print('gradientBoost')
# ensemble = ec.ensembleClassifier(X_train, y_train, X_test, y_test)
# n_estimators = 200
# max_depth = 2
# pred = ensemble.gradientBoost(n_estimators,max_depth)
#
# count = 0
# for i in range(700):
#     # print("%d: %d | %d" %(i, result[i], y_test[i]))
#     if (pred[i] == y_test[i]):
#         count += 1
# print("%d out of %d correct" %(count, len(pred)))

#--------------------------------------------
#run bagKNN
# ensemble = ec.ensembleClassifier(X_train, y_train, X_test, y_test)
# pred = ensemble.bagKNN(9, 0.25, 0.25)
# count = 0
# for i in range(700):
#     # print("%d: %d | %d" %(i, result[i], y_test[i]))
#     if (pred[i] == y_test[i]):
#         count += 1
# print("%d out of %d correct" %(count, len(pred)))

#--------------------------------------------
# #run bagSVM
# ensemble = ec.ensembleClassifier(X_train, y_train, X_test, y_test)
# pred = ensemble.bagSVM(3, 0.01, 0.9, 0.9)
# count = 0
# for i in range(700):
#     # print("%d: %d | %d" %(i, result[i], y_test[i]))
#     if (pred[i] == y_test[i]):
#         count += 1
# print("%d out of %d correct" %(count, len(pred)))
#------------------------------------------------
#run random forest
# ensemble = ec.ensembleClassifier(X_train, y_train, X_test, y_test)
# pred = ensemble.randomForest(250)
# count = 0
# for i in range(700):
#     # print("%d: %d | %d" %(i, result[i], y_test[i]))
#     if (pred[i] == y_test[i]):
#         count += 1
# print("%d out of %d correct" %(count, len(pred)))


#---------------------------------
#run linear SVC
# clf = LinearSVC()
# clf.fit(train_pca[:,range(num_pca)], y_train)
# result = clf.predict(val_pca[:,range(num_pca)])
#--------------------------------------------------------------
# #SVM (non-linear)
# #num_pca tested: 30, 50, 70. 50 yields best result
# num_pca = 50
# kernal = ['linear','rbf']
# C = [1, 3, 5, 7,  10, 20]
# gamma = [1e-1, 1e-2, 1e-3, 1e-4]
# accuracy_train = np.zeros([len(kernal),len(C), len(gamma)])
# correct_train = np.zeros([len(kernal),len(C), len(gamma)])
# ce_train = np.zeros([len(kernal),len(C), len(gamma)])
# accuracy_test = np.zeros([len(kernal),len(C), len(gamma)])
# correct_test = np.zeros([len(kernal),len(C), len(gamma)])
# ce_test = np.zeros([len(kernal),len(C), len(gamma)])
# hist_test = np.zeros([len(kernal),len(C), len(gamma), 8])
# for type in range(len(kernal)):
#     print(type)
#     for j in range(len(C)):
#         for k in range(len(gamma)):
#             clf = SVC(C = C[j], kernel=kernal[type], gamma=gamma[k], probability=True)
#             clf.fit(train_pca[:,range(num_pca)], y_train)
#             #test set
#             result = clf.predict(val_pca[:,range(num_pca)])
#             result_proba = clf.predict_proba(val_pca[:,range(num_pca)])
#             count = 0
#             hist = np.zeros(8)
#             for i in range(len(y_test)):
#                 # print("%d: %d | %d" %(i, result[i], y_test[i]))
#                 if (result[i] == y_test[i]):
#                     count += 1
#                 hist[result[i]-1]  += 1
#             # print("%d out of %d correct" %(count, len(result)))
#             print(hist)
#             accuracy_test[type,j,k] = count * 100.0 / len(y_test)
#             correct_test[type,j,k] = count#
#             ce_test[type, j, k] = log_loss(y_test, result_proba)
#             hist_test[type, j, k, :] = hist
#
#             # test model on training set itself
#             result = clf.predict(train_pca[:, range(num_pca)])
#             result_proba = clf.predict_proba(train_pca[:, range(num_pca)])
#             count = 0
#             for i in range(len(y_train)):
#                 # print("%d: %d | %d" %(i, result[i], y_test[i]))
#                 if (result[i] == y_train[i]):
#                     count += 1
#             # print("%d out of %d correct" %(count, len(result)))
#             accuracy_train[type, j, k] = count * 100.0 / len(y_train)
#             correct_train[type, j, k] = count  #
#             ce_train[type, j, k] = log_loss(y_train, result_proba)
#             print('Done')
#
# # for i in range(len(num_pca)):
# #     print("PCA = %d" %num_pca[i])
# #     print(accuracy_train[i,:,:])
# #     print(correct_train[i,:,:])
# #save accuracy
# f = open('denseSIFT_pca_SVC_accuracy.pckl', 'wb')
# pickle.dump((accuracy_train,correct_train, ce_train, accuracy_test,correct_test, ce_test, hist_test), f)
# f.close()

#-----------------------------------------
#KNN
# num_pca = [5, 10, 15, 20]
#
# # n_neighbors = [1,2]
# n_neighbors = [3, 7, 11,13, 17,21,25, 29, 33, 37, 41, 45, 49, 53, 57, 61, 65, 69, 73, 77, 81, 85, 89, 93, 97, 101]
#
# accuracy = np.zeros([len(num_pca),len(n_neighbors)])
# correct = np.zeros([len(num_pca),len(n_neighbors)])
# ce = np.zeros([len(num_pca),len(n_neighbors)])
# hist_test = np.zeros([len(num_pca),len(n_neighbors),8])#number of labels
#
# for num in range(len(num_pca)):
#     print(num)
#     for j in range(len(n_neighbors)):
#         clf = KNeighborsClassifier(n_neighbors=n_neighbors[j])
#         clf.fit(train_pca[:,range(num_pca[num])], y_train)
#         result = clf.predict(val_pca[:,range(num_pca[num])])
#         result_proba = clf.predict_proba(val_pca[:,range(num_pca[num])])
#         count = 0
#         hist = np.zeros(8)
#         for i in range(len(y_test)):
#             # print("%d: %d | %d" %(i, result[i], y_test[i]))
#             if (result[i] == y_test[i]):
#                 count += 1
#             hist[result[i] - 1] += 1
#         # print("%d out of %d correct" %(count, len(result)))
#         accuracy[num,j] = count * 100.0 / len(y_test)
#         correct[num,j] = count#
#         ce[num, j] = log_loss(y_test,result_proba)
#         hist_test[num, j,:] = hist
#         # print result
# # for i in range(len(num_pca)):
# #     print("PCA = %d" %num_pca[i])
# #     print(accuracy[i,:])
# #     print(correct[i,:])
# #     print(ce[i,:])
# #save accuracy
# f = open('denseSIFT_pca_KNN_accuracy2.pckl', 'wb')
# pickle.dump((accuracy,correct, ce, hist_test), f)
# f.close()
# ----------------------------------------------------------
# #Perform k-means clustering
# k = 50
# voc, variance = kmeans(descriptors, k, 1)
# print(voc.shape)
# # Calculate the histogram of features
# num_images = 6999
# im_features = np.zeros((num_images, k), "float32")
# for i in xrange(num_images):
#     print(descriptors[(i*256):(i*256+256),:].shape)
#     words, distance = vq(descriptors[(i*256):(i*256+256),:],voc)
#     for w in words:
#         im_features[i][w] += 1
#
# # Perform Tf-Idf vectorization
# nbr_occurences = np.sum( (im_features > 0) * 1, axis = 0)
# idf = np.array(np.log((1.0*num_images+1) / (1.0*nbr_occurences + 1)), 'float32')
#
# print('Done')
# # Scaling the words
# stdSlr = StandardScaler().fit(im_features)
# im_features = stdSlr.transform(im_features)
#
# Train the Linear SVM
# print im_features.shape[0]
# print np.array(image_classes)[0]

#
# # Save the SVM
# joblib.dump((clf, image_classes, stdSlr, k, voc), "bof.pkl", compress=3)