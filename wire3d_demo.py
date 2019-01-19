from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# X, Y, Z = axes3d.get_test_data(0.05)
C = [1, 3, 5, 7,  10, 20]
gamma = [1e-1, 1e-2, 1e-3, 1e-4]
log_gamma = [-1, -2, -3, -4]
Z = np.ones([len(log_gamma), len(C)])
x = C
y = log_gamma
xv, yv = np.meshgrid(x, y)
print(xv.shape)
print(xv)
print(yv.shape)
print(yv)
print(Z)
ax.plot_wireframe(xv, yv, Z, rstride=1, cstride=1)
ax.plot_wireframe(xv, yv, Z*2, rstride=1, cstride=1)

plt.xlabel('C')
plt.ylabel('log(gamma)')
ax.set_zlabel('Accuracy')
plt.show()

