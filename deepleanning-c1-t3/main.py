import numpy as np
import h5py
import init_utils
import matplotlib.pyplot as plt
import sklearn
import scipy.io as sio
import reg_utils    #第二部分，正则化
import gc_utils     #第三部分，梯度校验
import testCases #参见资料包，或者在文章底部copy
from dnn_utils import sigmoid, sigmoid_backward, relu, relu_backward #参见资料包
import lr_utils #参见资料包，或者在文章底部copy
#%matplotlib inline #如果你使用的是Jupyter Notebook，请取消注释。
plt.rcParams['figure.figsize'] = (7.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

np.random.seed(1)#指定时间种子





#relu函数
def relu(X):
    return np.maximum(X,0)

def sigmoid(X):
    return 1/(1+np.exp(-X))
#参数初始化0：
def initialize_parameters_zeros(layers_dims):
    parameters={}
    L=len(layers_dims)
    for l in range(1,L):
        parameters["W"+str(l)]=np.zeros((layers_dims[l],layers_dims[l-1]))
        parameters["b"+str(l)]=np.zeros((layers_dims[l],1))

    #使用断言判断参数是否正确
        assert (parameters["W"+str(l)].shape==(layers_dims[l],layers_dims[l-1]))
        assert (parameters["b"+str(l)].shape==(layers_dims[l],1))
    return parameters

#参数随机初始化（高斯分布）
def initialize_parameters_random(layers_dims):
    np.random.seed(3)#指定随机种子
    parameters={}
    L=len(layers_dims)
    for l in range(1,L):
        parameters["W"+str(l)]=np.random.randn(layers_dims[l],layers_dims[l-1])* 10 #使用10倍缩放
        parameters["b"+str(l)]=np.random.randn(layers_dims[l],1)  #randn满足高斯分布，rand随机分布

    #使用断言判断参数是否正确
        assert (parameters["W"+str(l)].shape==(layers_dims[l],layers_dims[l-1]))
        assert (parameters["b"+str(l)].shape==(layers_dims[l],1))
    return parameters


def initialize_parameters_he(layers_dims):
    np.random.seed(3)  # 指定随机种子
    parameters = {}
    L = len(layers_dims)  # 层数

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l - 1]) * np.sqrt(2 / layers_dims[l - 1])
        parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))

        # 使用断言确保我的数据格式是正确的
        assert (parameters["W" + str(l)].shape == (layers_dims[l], layers_dims[l - 1]))
        assert (parameters["b" + str(l)].shape == (layers_dims[l], 1))

    return parameters



def forward_propagation(X,parameters):
    #print(type(parameters))
    W1= parameters["W1"]
    b1=parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    W3=parameters["W3"]
    b3=parameters["b3"]

    # LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID
    z1=np.dot(W1,X)+b1
    a1=relu(z1)
    z2=np.dot(W2,a1)+b2
    a2=relu(z2)
    z3=np.dot(W3,a2)+b3
    a3=sigmoid(z3)

    #将前向传播数据保存下来
    cache=(z1, a1, W1, b1, z2, a2, W2, b2, z3, a3, W3, b3)

    return a3,cache


def compute_loss(a3,Y):
    m=len(Y)
    loss=np.multiply(-Y,np.log(a3))+np.multiply(Y-1,np.log(1-a3))
    return 1/m*np.sum(loss)


def backward_propagation(X,Y,cache):
    (z1, a1, W1, b1, z2, a2, W2, b2, z3, a3, W3, b3) = cache
    m = X.shape[1]
    dz3=(a3-Y)   #为什么？
    dW3=np.dot(dz3,a2.T)
    db3=np.sum(dz3,axis=1,keepdims=True)

    da2=np.dot(W3.T,dz3)
    dz2=np.multiply(da2,np.int64(a2>0))
    dW2=np.dot(dz2,a1.T)
    db2=np.sum(dz2,axis=1,keepdims=True)

    da1=np.dot(W2.T,dz2)
    dz1 = np.multiply(da1, np.int64(a1 > 0))
    dW1=np.dot(dz1,X.T)
    db1 = np.sum(dz1, axis=1, keepdims=True)

    gradients = {"dz3": dz3, "dW3": dW3, "db3": db3,
                 "da2": da2, "dz2": dz2, "dW2": dW2, "db2": db2,
                 "da1": da1, "dz1": dz1, "dW1": dW1, "db1": db1}

    return gradients


def update_parameters(parameters,grads,learning_rate):
    L=len(parameters)//2    #层数
    for l in range(1,L+1):
        parameters["W"+str(l)]=parameters["W"+str(l)]-learning_rate*grads["dW"+str(l)]
        parameters["b" + str(l)] = parameters["b" + str(l)] - learning_rate * grads["db" + str(l)]
    return parameters


def model(X,Y,learning_rate=0.0001,iteration=15000,print_cost=True,initialization="he",is_polt=True):
    
    """
    实现一个三层的神经网络：LINEAR ->RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID

    参数：
        X - 输入的数据，维度为(2, 要训练/测试的数量)
        Y - 标签，【0 | 1】，维度为(1，对应的是输入的数据的标签)
        learning_rate - 学习速率
        num_iterations - 迭代的次数
        print_cost - 是否打印成本值，每迭代1000次打印一次
        initialization - 字符串类型，初始化的类型【"zeros" | "random" | "he"】
        is_polt - 是否绘制梯度下降的曲线图
    返回
        parameters - 学习后的参数
    """
    grads = {}
    costs = []
    m = X.shape[1]
    layers_dims = [X.shape[0],10,5,1]#神经网络层数

    #模型参数
    num_iterations=30000

    #初始化选择
    if initialization == 'zeros':
        parameters=initialize_parameters_zeros(layers_dims)
    elif initialization == 'random':
        parameters=initialize_parameters_random(layers_dims)
    elif initialization == 'he':
        parameters=initialize_parameters_he(layers_dims)
    else:
        print('参数输入错误')
        exit()

    #开始学习
    for i in range(0,num_iterations):
        #前向传播
        a3 , cache = forward_propagation(X,parameters)

        #计算成本
        cost = compute_loss(a3,Y)

        #反向传播
        grads = backward_propagation(X,Y,cache)

        #更新参数
        parameters = update_parameters(parameters,grads,learning_rate)

        #记录成本
        if i % 1000 == 0:
            costs.append(cost)
            #打印成本
            if print_cost:
                print("第" + str(i) + "次迭代，成本值为：" + str(cost))


    #学习完毕，绘制成本曲线
    if is_polt:
        plt.plot(costs)
        plt.ylabel('cost')
        plt.xlabel('iterations (per hundreds)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

    #返回学习完毕后的参数
    return parameters

#初始化部分
"""train_X, train_Y, test_X, test_Y = init_utils.load_dataset(is_plot=True)
print(train_X.shape, train_Y.shape, test_X.shape, test_Y.shape)      #(2, 300) (1, 300) (2, 100) (1, 100)
parameters = model(train_X, train_Y, initialization = "he",is_polt=True)
print ("训练集:")
predictions_train = init_utils.predict(train_X, train_Y, parameters)
print ("测试集:")
predictions_test = init_utils.predict(test_X, test_Y, parameters)
print("predictions_train = " + str(predictions_train))
print("predictions_test = " + str(predictions_test))

plt.title("Model with Zeros initialization")
axes = plt.gca()
axes.set_xlim([-1.5, 1.5])
axes.set_ylim([-1.5, 1.5])
init_utils.plot_decision_boundary(lambda x: init_utils.predict_dec(parameters, x.T), train_X, train_Y)"""


def compute_cost_with_regularization(A3, Y, parameters, lambd):
    """
    实现公式2的L2正则化计算成本

    参数：
        A3 - 正向传播的输出结果，维度为（输出节点数量，训练/测试的数量）
        Y - 标签向量，与数据一一对应，维度为(输出节点数量，训练/测试的数量)
        parameters - 包含模型学习后的参数的字典
    返回：
        cost - 使用公式2计算出来的正则化损失的值

    """
    m = Y.shape[1]
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    W3 = parameters["W3"]

    cross_entropy_cost = reg_utils.compute_cost(A3, Y)

    L2_regularization_cost = lambd * (np.sum(np.square(W1)) + np.sum(np.square(W2)) + np.sum(np.square(W3))) / (2 * m)

    cost = cross_entropy_cost + L2_regularization_cost

    return cost


# 当然，因为改变了成本函数，我们也必须改变向后传播的函数， 所有的梯度都必须根据这个新的成本值来计算。

def backward_propagation_with_regularization(X, Y, cache, lambd):
    """
    实现我们添加了L2正则化的模型的后向传播。

    参数：
        X - 输入数据集，维度为（输入节点数量，数据集里面的数量）
        Y - 标签，维度为（输出节点数量，数据集里面的数量）
        cache - 来自forward_propagation（）的cache输出
        lambda - regularization超参数，实数

    返回：
        gradients - 一个包含了每个参数、激活值和预激活值变量的梯度的字典
    """

    m = X.shape[1]

    (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3) = cache

    dZ3 = A3 - Y

    dW3 = (1 / m) * np.dot(dZ3, A2.T) + ((lambd * W3) / m)
    db3 = (1 / m) * np.sum(dZ3, axis=1, keepdims=True)

    dA2 = np.dot(W3.T, dZ3)
    dZ2 = np.multiply(dA2, np.int64(A2 > 0))
    dW2 = (1 / m) * np.dot(dZ2, A1.T) + ((lambd * W2) / m)
    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)

    dA1 = np.dot(W2.T, dZ2)
    dZ1 = np.multiply(dA1, np.int64(A1 > 0))
    dW1 = (1 / m) * np.dot(dZ1, X.T) + ((lambd * W1) / m)
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)

    gradients = {"dZ3": dZ3, "dW3": dW3, "db3": db3, "dA2": dA2,
                 "dZ2": dZ2, "dW2": dW2, "db2": db2, "dA1": dA1,
                 "dZ1": dZ1, "dW1": dW1, "db1": db1}

    return gradients




train_X, train_Y, test_X, test_Y = reg_utils.load_2D_dataset(is_plot=True)


def model(X, Y, learning_rate=0.3, num_iterations=30000, print_cost=True, is_plot=True, lambd=0, keep_prob=1):
    """
    实现一个三层的神经网络：LINEAR ->RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID

    参数：
        X - 输入的数据，维度为(2, 要训练/测试的数量)
        Y - 标签，【0(蓝色) | 1(红色)】，维度为(1，对应的是输入的数据的标签)
        learning_rate - 学习速率
        num_iterations - 迭代的次数
        print_cost - 是否打印成本值，每迭代10000次打印一次，但是每1000次记录一个成本值
        is_polt - 是否绘制梯度下降的曲线图
        lambd - 正则化的超参数，实数
        keep_prob - 随机删除节点的概率
    返回
        parameters - 学习后的参数
    """
    grads = {}
    costs = []
    m = X.shape[1]
    layers_dims = [X.shape[0], 20, 3, 1]

    # 初始化参数
    parameters = reg_utils.initialize_parameters(layers_dims)

    # 开始学习
    for i in range(0, num_iterations):
        # 前向传播
        ##是否随机删除节点
        if keep_prob == 1:
            ###不随机删除节点
            a3, cache = reg_utils.forward_propagation(X, parameters)
        elif keep_prob < 1:
            ###随机删除节点
            a3, cache = reg_utils.forward_propagation_with_dropout(X, parameters, keep_prob)
        else:
            print("keep_prob参数错误！程序退出。")
            exit

        # 计算成本
        ## 是否使用二范数
        if lambd == 0:
            ###不使用L2正则化
            cost = reg_utils.compute_cost(a3, Y)
        else:
            ###使用L2正则化
            cost = reg_utils.compute_cost_with_regularization(a3, Y, parameters, lambd)

        # 反向传播
        ##可以同时使用L2正则化和随机删除节点，但是本次实验不同时使用。
        assert (lambd == 0 or keep_prob == 1)

        ##两个参数的使用情况
        if (lambd == 0 and keep_prob == 1):
            ### 不使用L2正则化和不使用随机删除节点
            grads = reg_utils.backward_propagation(X, Y, cache)
        elif lambd != 0:
            ### 使用L2正则化，不使用随机删除节点
            grads = reg_utils.backward_propagation_with_regularization(X, Y, cache, lambd)
        elif keep_prob < 1:
            ### 使用随机删除节点，不使用L2正则化
            grads =reg_utils. backward_propagation_with_dropout(X, Y, cache, keep_prob)

        # 更新参数
        parameters = reg_utils.update_parameters(parameters, grads, learning_rate)

        # 记录并打印成本
        if i % 1000 == 0:
            ## 记录成本
            costs.append(cost)
            if (print_cost and i % 10000 == 0):
                # 打印成本
                print("第" + str(i) + "次迭代，成本值为：" + str(cost))

    # 是否绘制成本曲线图
    if is_plot:
        plt.plot(costs)
        plt.ylabel('cost')
        plt.xlabel('iterations (x1,000)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

    # 返回学习后的参数
    return parameters

parameters = model(train_X, train_Y,is_plot=True)
print("训练集:")
predictions_train = reg_utils.predict(train_X, train_Y, parameters)
print("测试集:")
predictions_test = reg_utils.predict(test_X, test_Y, parameters)
plt.title("Model without regularization")
axes = plt.gca()
axes.set_xlim([-0.75,0.40])
axes.set_ylim([-0.75,0.65])
reg_utils.plot_decision_boundary(lambda x: reg_utils.predict_dec(parameters, x.T), train_X, train_Y)
