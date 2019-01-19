#!/usr/local/bin/python2.7


import argparse as ap
import cv2
import imutils
import numpy as np
import os
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
from scipy.cluster.vq import *
from sklearn.preprocessing import StandardScaler
import pickle

train_path = 'D:\Skule\Fifth year\CSC411\Project/411a3/train/'
labelfile = np.recfromcsv('D:\Skule\Fifth year\CSC411\Project/411a3/train.csv', delimiter=',', names=True)
image_classes = [x[1] for x in labelfile]
image_names = np.array([x[0] for x in labelfile])
image_paths_train = []
image_paths_val = []

for i in range(1,9):
    temp_image_names = np.squeeze([x for x, y in enumerate(labelfile) if y[1] == i])

    training_names = image_names[0:round(len(temp_image_names)*0.9)]
    for training_name in training_names:
        class_path = train_path + '%0*d' % (5, training_name) + '.jpg'
        image_paths_train.append(class_path)

    val_names = image_names[round(len(temp_image_names)*0.9):len(temp_image_names)]
    for val_name in val_names:
        class_path = train_path + '%0*d' % (5,val_name) + '.jpg'
        image_paths_val.append(class_path)

print(len(image_paths_train))
print(len(image_paths_val))
# Create feature extraction and keypoint detector objects
fea_det = cv2.FeatureDetector_create("SIFT")
des_ext = cv2.DescriptorExtractor_create("SIFT")

# List where all the descriptors are stored
des_list = []
for image_path in image_paths_train:
    im = cv2.imread(image_path)
    # normal SIFT
    # kpts = fea_det.detect(im)
    # kpts, des = des_ext.compute(im, kpts)
    # if des == None:
    #     des = np.zeros([1, 128])

    #dense SIFT (http://stackoverflow.com/questions/33120951/compute-dense-sift-features-in-opencv-3-0/33702400#33702400)
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    step_size = 8
    kpts = [cv2.KeyPoint(a, b, step_size) for b in range(0, gray.shape[0], step_size)
          for a in range(0, gray.shape[1], step_size)]
    kpts, des = des_ext.compute(im, kpts)

    des_list.append((image_path, des))

# Stack all the descriptors vertically in a numpy array
descriptors = des_list[0][1]
for image_path, descriptor in des_list[1:]:
    descriptors = np.vstack((descriptors, descriptor))

f = open('denseSIFT_train.pckl', 'wb')
pickle.dump(descriptors, f)
f.close()

print(descriptors.shape)


# y = np.recfromcsv('D:\Skule\Fifth year\CSC411\Project/411a3/train_mod.csv', delimiter=',', names=True)
# # y = y[6000:7000]
# image_classes = [x[1] for x in y]
# # f = open('denseSIFT_train6.pckl', 'rb')
# # descriptors = pickle.load(f)
# # print(descriptors.shape)
#
# descriptors = []
# for i in range(7):
#     f = open('denseSIFT_train' + str(i) + '.pckl', 'rb')
#     object = pickle.load(f)
#     if i == 0:
#         descriptors = object
#     else:
#         descriptors = np.append(descriptors, object, axis=0)
#     print(descriptors.shape)
#     f.close()
#
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
# # Train the Linear SVM
# print im_features.shape[0]
# print np.array(image_classes)[0]
# clf = LinearSVC()
# clf.fit(im_features, np.array(image_classes))
#
# # Save the SVM
# joblib.dump((clf, image_classes, stdSlr, k, voc), "bof.pkl", compress=3)