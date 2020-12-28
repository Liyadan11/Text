
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys,os
from matplotlib import animation, rc
from IPython.display import HTML
from matplotlib.animation import FuncAnimation


def loaddata(filename): #加载数据函数

    dataSet = pd.read_table(filename, header=None)
    dataSet.columns = ['X1', 'X2', 'label']
    dataSet.insert(0, 'X0', 1)
    columns = [i for i in dataSet.columns if i != 'label']
    data_x = dataSet[columns]
    data_y = dataSet[['label']]
    return data_x,data_y


# In[4]:


def sigmoid(y): #sigmoid函数
    s = 1.0/(1.0+np.exp(-y))
    return s

def cost(xMat,weights,yMat):#定义损失函数
    
    m, n = xMat.shape
    hypothesis = sigmoid(np.dot(xMat, weights))  # 预测值
    cost = (-1.0 / m) * np.sum(yMat.T * np.log(hypothesis) + (1 - yMat).T * np.log(1 - hypothesis))  # 损失函数
    return cost


# In[5]:


import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
from IPython.display import HTML



class SimpleAnimation(animation.FuncAnimation):
    """
    Define a animation class.
    Author:Louis Tiao
    Revised by: Gordon Woo
    Email:wuguoning@gmail.com
    """

    def __init__(self, xdata, ydata, label, fig=None, ax=None, frames=None,
                 interval=60, repeat_delay=5, blit=True, **kwargs):

        if fig is None:
            if ax is None:
                fig, ax = plt.subplots()
            else:
                fig = ax.get_figure()
        else:
            if ax is None:
                ax = fig.gca()

        self.fig = fig
        self.ax = ax
        self.xdata = xdata
        self.ydata = ydata

        if frames is None:
            frames = len(xdata)

        self.line = self.ax.plot([], [], label=label, color='red', lw=2)[0]
        self.point = self.ax.plot([], [], 'o', color='red')[0]

        super(SimpleAnimation, self).__init__(fig, self.animate, init_func=self.init_anim,
                                                  frames=frames, interval=interval, blit=blit,
                                                  repeat_delay=repeat_delay, **kwargs)

    def init_anim(self):
        self.line.set_data([], [])
        self.point.set_data([], [])

        return self.line, self.point

    def animate(self, i):
        self.line.set_data(self.xdata[:i], self.ydata[:i])
        self.point.set_data(self.xdata[i-1:i], self.ydata[i-1:i])

        return self.line, self.point


# In[6]:


import numpy as np

class CostFunction(object):
    """
    Cost Function of different types.
    """

    def __init__(self, weight, bias):
        """
        Parameters:
            self.weight:  initial vale of weight
            self.bias:    initial vale of bias
            self.loss_h:  loss history
        """
        self.weight = weight
        self.bias = bias
        self.loss_h = []


    def GD(self, epoch, training_data, output_value, eta):
        """
        Train the neuron using gradient method.
        The "training_data" is a signal data, the eta
        is the learning rate.
        """
        self.loss_h = []
        nabla_b = 0
        nabla_w = 0

        # update the weight and bias.
        for i in range(epoch):
            delta_nabla_w, delta_nabla_b = self.update_gradiet(training_data, output_value)
            nabla_b += delta_nabla_b
            nabla_w += delta_nabla_w
            self.weight = self.weight - eta * nabla_w
            self.bias = self.bias - eta * nabla_b
            self.loss_h.append(self.evaluate(training_data, output_value))

    def update_gradiet(self, x, y):
        """
        Update the neuron's weight by applying gradient descent
        method. The tuples x is the input and y is the supposed
        output values.
        """
        #neuron_input = w*x +b
        neuron_input = self.weight * x + self.bias
        nabla_w = (sigmoid(neuron_input) - y) *             sigmoid_prime(neuron_input) * x
        nabla_b = (sigmoid(neuron_input) - y) *             sigmoid_prime(neuron_input)
        return nabla_w, nabla_b


    def CrossEntropy(self, training_data, output_value, eta):
        """
        The Cross-Entropy cost function for this neuron is:
            C = -1/n*(\sum_x [y \ln a + (1-y)\ln (1-a)])
            where n is the total number of items of training data,
            the sum is over all training inputs, x, and y is the
            corresponding desired output.
            It tells us that the rate at which the weight learns
            is controlled by (\sigma(z) - y), i.e., by the error
            in the output. The larger the error, the faster the
            neuron will learn.
        """
        nabla_b = 0
        nabla_w = 0

        neuron_input = self.weight * training_data + self.bias
        nabla_w = (sigmoid(neuron_input) - output_value) * training_data
        nabla_b = sigmoid(neuron_input) - output_value
        self.weight = self.weight - eta * nabla_w
        self.bias = self.bias - eta * nabla_b

    def evaluate(self, training_data, output_value):
        """
        Evaluate the output of the neuron.
        """
        return sigmoid(self.weight*training_data + self.bias)

# Define the function for output

def sigmoid(x):
    """
    The sigmoid function
    """
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_prime(x):
    """
    Derivative of the sigmoid function
    """
    return sigmoid(x)*(1.0 - sigmoid(x))


# In[7]:


def BGD_LR(data_x,data_y,alpha=0.1,maxepochs=10000,epsilon=1e-4):
    
    xMat = np.mat(data_x)
    yMat = np.mat(data_y)
    m,n = xMat.shape
    weights = np.ones((n,1)) #初始化模型参数
    epochs_count = 0
    loss_list = []
    epochs_list = []
    while epochs_count < maxepochs:
        loss = cost(xMat,weights,yMat) #上一次损失值
        hypothesis = sigmoid(np.dot(xMat,weights)) #预测值
        error = hypothesis -yMat #预测值与实际值误差
        grad = (1.0/m)*np.dot(xMat.T,error) #损失函数的梯度
        last_weights = weights #上一轮迭代的参数
        weights = weights - alpha*grad #参数更新
        loss_new = cost(xMat,weights,yMat)#当前损失值
        if abs(loss_new-loss)<epsilon:#终止条件
            break
        loss_list.append(loss_new)
        epochs_list.append(epochs_count)
        epochs_count += 1
    print(loss_new)
    print('迭代到第{}次'.format(epochs_count))
    plt.plot(epochs_list,loss_list)
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.show()
    return weights


# In[8]:


if __name__ == '__main__':
    data_x,data_y = loaddata('testSet.txt')
    weights_bgd = BGD_LR(data_x, data_y, alpha=0.1, maxepochs=10000, epsilon=1e-4)


# In[27]:


if __name__ == '__main__':
    
    data_x,data_y = loaddata('testSet.txt')
    xMat = np.mat(data_x)
    yMat = np.mat(data_y)
    alpha=0.1
    maxepochs=10000
    epsilon=1e-4
    m,n = xMat.shape
    weights = np.ones((n,1)) #初始化模型参数
    epochs_count = 0
    loss_list = []
    epochs_list = []
    while epochs_count < maxepochs:
        loss = cost(xMat,weights,yMat) #上一次损失值
        hypothesis = sigmoid(np.dot(xMat,weights)) #预测值
        error = hypothesis -yMat #预测值与实际值误差
        grad = (1.0/m)*np.dot(xMat.T,error) #损失函数的梯度
        last_weights = weights #上一轮迭代的参数
        weights = weights - alpha*grad #权重更新
        loss_new = cost(xMat,weights,yMat)#当前损失值
        if abs(loss_new-loss)<epsilon:#终止条件
            break
        loss_list.append(loss_new)
        epochs_list.append(epochs_count)
        epochs_count += 1
    print(loss_new)
    print('迭代到第{}次'.format(epochs_count))
#     print(epochs_list)
    print(loss_list)
    fig, ax = plt.subplots()
    anim = SimpleAnimation(epochs_list, loss_list, label='GD', ax = ax)
    ax.legend(loc='upper left')
    HTML=(anim.to_jshtml())
    ax.set_xlim([0,epochs_count])
    ax.set_ylim([0.0,3])
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.show()


# In[28]:


import numpy as np
import matplotlib.pyplot as plt
import sys,os
from matplotlib import animation, rc
from IPython.display import HTML
from matplotlib.animation import FuncAnimation

# system path append
sys.path.append('../')

# import local modules
# from neuralsrc.perceptrons import CostFunction
# from neuralsrc.animation import SimpleAnimation


if __name__ == "__main__":

    weight, bias = 0.6, 0.9
    eta = 0.15
    epoch = 300
    x_input, y_output = 1.0, 0.0

    obj1 = CostFunction(weight, bias)
    obj1.GD(300, x_input, y_output, eta)
    loss_h = obj1.loss_h
    print(loss_h)

    # Animation
    epoch_num = np.arange(epoch)
    fig, ax = plt.subplots(figsize=(16,9))
    ax.set_xlim([0,epoch])
    ax.set_ylim([0.0,1])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Lost')
    ax.grid()

    anim = SimpleAnimation(epoch_num, loss_h, label='GD', ax = ax)
    ax.legend(loc='upper left')
    HTML=(anim.to_jshtml())
    plt.show()


