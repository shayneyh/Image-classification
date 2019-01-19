import pickle
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import axes3d

# #plot SVM for SIFT features
# f = open('denseSIFT_pca_SVC_accuracy.pckl', 'rb')
# accuracy_train,correct_train, ce_train, accuracy_test,correct_test, ce_test, hist_test = pickle.load(f)
# f.close()
#
# # varied params:
# kernal = ['linear','rbf']
# C = [1, 3, 5, 7,  10, 20]
# gamma = [1e-1, 1e-2, 1e-3, 1e-4]
# log_gamma = [-1, -2, -3, -4] #plot log gamma instead
#
#
# x = C
# y = log_gamma
# xv, yv = np.meshgrid(x, y)
#
# #plot accuracy linear kernal
# linear_ind = 0
# fig = plt.figure()
# ax = fig.add_subplot(121, projection='3d')
# ax.plot_wireframe(xv, yv, accuracy_train[linear_ind,:,:].T, rstride=1, cstride=1,color="red")
# ax.plot_wireframe(xv, yv, accuracy_test[linear_ind,:,:].T, rstride=1, cstride=1, color="blue")
# plt.title('Accuracy (Linear)')
# plt.xlabel('C')
# plt.ylabel('log(gamma)')
# plt.yticks((-1, -2, -3, -4), ('-1', '-2', '-3', '-4'))
# ax.set_zlabel('Accuracy')
# plt.legend(['Training','Validation'])
# #plot ce linear kernal
# ax = fig.add_subplot(122, projection='3d')
# ax.plot_wireframe(xv, yv, ce_train[linear_ind,:,:].T, rstride=1, cstride=1,color="red")
# ax.plot_wireframe(xv, yv, ce_test[linear_ind,:,:].T, rstride=1, cstride=1, color="blue")
# plt.title('Cross Entropy (Linear)')
# plt.xlabel('C')
# plt.ylabel('log(gamma)')
# plt.yticks((-1, -2, -3, -4), ('-1', '-2', '-3', '-4'))
# ax.set_zlabel('CE')
# plt.legend(['Training','Validation'])
# plt.subplots_adjust(wspace=0.10, hspace=0.10)
# plt.show()
#
# #plot accuracy rbf kernal
# rbf_ind = 1
# fig = plt.figure()
# ax = fig.add_subplot(121, projection='3d')
# ax.plot_wireframe(xv, yv, accuracy_train[rbf_ind,:,:].T, rstride=1, cstride=1,color="red")
# ax.plot_wireframe(xv, yv, accuracy_test[rbf_ind,:,:].T, rstride=1, cstride=1, color="blue")
# plt.title('Accuracy (RBF)')
# plt.xlabel('C')
# plt.ylabel('log(gamma)')
# plt.yticks((-1, -2, -3, -4), ('-1', '-2', '-3', '-4'))
# ax.set_zlabel('Accuracy')
# plt.legend(['Training','Validation'])
# #plot ce rbf kernal
# ax = fig.add_subplot(122, projection='3d')
# ax.plot_wireframe(xv, yv, ce_train[rbf_ind,:,:].T, rstride=1, cstride=1,color="red")
# ax.plot_wireframe(xv, yv, ce_test[rbf_ind,:,:].T, rstride=1, cstride=1, color="blue")
# plt.title('Cross Entropy (RBF)')
# plt.xlabel('C')
# plt.ylabel('log(gamma)')
# plt.yticks((-1, -2, -3, -4), ('-1', '-2', '-3', '-4'))
# ax.set_zlabel('CE')
# plt.legend(['Training','Validation'])
# plt.subplots_adjust(wspace=0.10, hspace=0.10)
# plt.show()


# #plot KNN for SIFT features
# f = open('denseSIFT_pca_KNN_accuracy2.pckl', 'rb')
# accuracy,correct, ce, hist_test = pickle.load(f)
# f.close()
# #params
# num_pca = [5, 10, 15, 20]
# n_neighbors = [3, 7, 11,13, 17,21,25, 29, 33, 37, 41, 45, 49, 53, 57, 61, 65, 69, 73, 77, 81, 85, 89, 93, 97, 101]
# colors = ['k','g','b','r','y']
# # accuracy
# # plt.figure()
# # line = []
# #
# # line1, = plt.plot(n_neighbors,accuracy[0,:],  'k', label="pca = 5")
# # line2, = plt.plot(n_neighbors,accuracy[1,:],  'g', label="pca = 10")
# # line3, = plt.plot(n_neighbors,accuracy[2,:],  'b', label="pca = 15")
# # line4, = plt.plot(n_neighbors,accuracy[3,:],  'r', label="pca = 20")
# # plt.legend(handles=[line1, line2, line3, line4])
# # plt.legend(bbox_to_anchor=(0.6, 0.5), loc=2, borderaxespad=0.)
#
# fig = plt.figure()
# # accuracy
# fig.add_subplot(121)
# plt.plot(n_neighbors,accuracy[1,:])
# plt.xlabel("# of nearest neighbors")
# plt.ylabel("Accuracy (%)")
# plt.title("Accuracy vs # NN")
# #ce
# fig.add_subplot(122)
# plt.plot(n_neighbors,ce[1,:])
# plt.xlabel("# of nearest neighbors")
# plt.ylabel("CE")
# plt.title("Cross Entropy vs # NN")
# plt.subplots_adjust(wspace=0.20, hspace=0.10)
# plt.show()

# #plot SVM for botleneck features
#optimzation of PCA components
num_pca = [50,75,100,150,175,300]
accuracy = [79,79,80,81, 79, 79]
fig = plt.figure()
plt.plot(num_pca,accuracy)
plt.xlabel("# of PCA components")
plt.ylabel("Accuracy (%)")
plt.title("Accuracy with varying PCA components")
plt.ylim(75, 85)
plt.subplots_adjust(wspace=0.10, hspace=0.10)
plt.show()



# f = open('bottleneck_SVC_accuracy.pckl', 'rb')
# accuracy_train,correct_train, ce_train, accuracy_test,correct_test, ce_test = pickle.load(f)
# f.close()
# print(accuracy_test)
# C = [1,5, 10, 100]
# gamma = [1e-1, 1e-2, 1e-3, 1e-4]
# log_gamma = [-1, -2, -3, -4] #plot log gamma instead
#
# x = C
# y = log_gamma
# xv, yv = np.meshgrid(x, y)
#
# #plot accuracy rbf kernal
# fig = plt.figure()
# ax = fig.add_subplot(121, projection='3d')
# ax.plot_wireframe(xv, yv, accuracy_train[0:4,:].T, rstride=1, cstride=1,color="red")
# ax.plot_wireframe(xv, yv, accuracy_test[0:4,:].T, rstride=1, cstride=1, color="blue")
# plt.title('Accuracy (RBF)')
# plt.xlabel('C')
# plt.ylabel('log(gamma)')
# plt.yticks((-1, -2, -3, -4), ('-1', '-2', '-3', '-4'))
# ax.set_zlabel('Accuracy (%)')
# plt.legend(['Training','Validation'])
# #plot ce rbf kernal
# ax = fig.add_subplot(122, projection='3d')
# ax.plot_wireframe(xv, yv, ce_train[0:4,:].T, rstride=1, cstride=1,color="red")
# ax.plot_wireframe(xv, yv, ce_test[0:4,:].T, rstride=1, cstride=1, color="blue")
# plt.title('Cross Entropy (RBF)')
# plt.xlabel('C')
# plt.ylabel('log(gamma)')
# plt.yticks((-1, -2, -3, -4), ('-1', '-2', '-3', '-4'))
# ax.set_zlabel('CE')
# plt.legend(['Training','Validation'])
# plt.subplots_adjust(wspace=0.10, hspace=0.10)
# plt.show()



#
# #plot KNN for botleneck features
# n_neighbors = [3,5,9,11,13,15]
# accuracy = [73,76,77,75, 74, 74]
# fig = plt.figure()
# plt.plot(n_neighbors,accuracy)
# plt.xlabel("# of nearest neighbors")
# plt.ylabel("Accuracy (%)")
# plt.title("Accuracy vs # NN")
# plt.ylim(70, 80)
# plt.subplots_adjust(wspace=0.10, hspace=0.10)
# plt.show()
