from sklearn.datasets import load_iris, load_boston, load_breast_cancer, load_diabetes, load_wine
import numpy as np


iris = load_iris()
x_iris = iris.data
y_iris = iris.target

boston = load_boston()
x_boston = boston.data
y_boston = boston.target

cancer = load_breast_cancer()
x_cancer = cancer.data
y_cancer = cancer.target

diabetes = load_diabetes()
x_diabetes = diabetes.data
y_diabetes = diabetes.target

wine = load_wine()
x_wine = wine.data
y_wine = wine.target

# 필요한 데이터 loading
x_data = np.load('./_save/NPY/keras55_x_iris.npy')
y_data = np.load('./_save/NPY/keras55_y_iris.npy')

# np.save('./_save/NPY/keras55_x_iris.npy', arr=x_iris)
# np.save('./_save/NPY/keras55_y_iris.npy', arr=y_iris)
# np.save('./_save/NPY/keras55_x_boston.npy', arr=x_boston)
# np.save('./_save/NPY/keras55_y_boston.npy', arr=y_boston)
# np.save('./_save/NPY/keras55_x_cancer.npy', arr=x_cancer)
# np.save('./_save/NPY/keras55_y_cancer.npy', arr=y_cancer)
# np.save('./_save/NPY/keras55_x_diabetes.npy', arr=x_diabetes)
# np.save('./_save/NPY/keras55_y_diabetes.npy', arr=y_diabetes)
# np.save('./_save/NPY/keras55_x_wine.npy', arr=x_wine)
# np.save('./_save/NPY/keras55_y_wine.npy', arr=y_wine)
