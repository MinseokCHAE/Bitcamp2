from sklearn.datasets import load_iris, load_boston, load_breast_cancer, load_diabetes, load_wine
from tensorflow.keras.datasets import mnist, fashion_mnist, cifar10, cifar100
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

(x_mnist_train, y_mnist_train), (x_mnist_test, y_mnist_test) = mnist.load_data()

(x_fashion_train, y_fashion_train), (x_fashion_test, y_fashion_test) = fashion_mnist.load_data()

(x_cifar10_train, y_cifar10_train), (x_cifar10_test, y_cifar10_test) = cifar10.load_data()

(x_cifar100_train, y_cifar100_train), (x_cifar100_test, y_cifar100_test) = cifar100.load_data()

# 필요한 데이터 loading
x_data = np.load('./_save/_npy/keras55_x_iris.npy')
y_data = np.load('./_save/_npy/keras55_y_iris.npy')

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

# np.save('./_save/NPY/keras55_x_mnist_train.npy', arr=x_mnist_train)
# np.save('./_save/NPY/keras55_x_mnist_test.npy', arr=x_mnist_test)
# np.save('./_save/NPY/keras55_y_mnist_train.npy', arr=y_mnist_train)
# np.save('./_save/NPY/keras55_y_mnist_test.npy', arr=y_mnist_test)

# np.save('./_save/NPY/keras55_x_fashion_train.npy', arr=x_fashion_train)
# np.save('./_save/NPY/keras55_x_fashion_test.npy', arr=x_fashion_test)
# np.save('./_save/NPY/keras55_y_fashion_train.npy', arr=y_fashion_train)
# np.save('./_save/NPY/keras55_y_fashion_test.npy', arr=y_fashion_test)

# np.save('./_save/NPY/keras55_x_cifar10_train.npy', arr=x_cifar10_train)
# np.save('./_save/NPY/keras55_x_cifar10_test.npy', arr=x_cifar10_test)
# np.save('./_save/NPY/keras55_y_cifar10_train.npy', arr=y_cifar10_train)
# np.save('./_save/NPY/keras55_y_cifar10_test.npy', arr=y_cifar10_test)

# np.save('./_save/NPY/keras55_x_cifar100_train.npy', arr=x_cifar100_train)
# np.save('./_save/NPY/keras55_x_cifar100_test.npy', arr=x_cifar100_test)
# np.save('./_save/NPY/keras55_y_cifar100_train.npy', arr=y_cifar100_train)
# np.save('./_save/NPY/keras55_y_cifar100_test.npy', arr=y_cifar100_test)
