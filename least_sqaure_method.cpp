import numpy as np
import matplotlib.pyplot as plt
# 目标函数 y=sin(2*pi*x)+N(0,0.001^2)
def objective_func(x):
    return np.sin(2*np.pi*x)+np.random.normal(0, 0.001)

# 参数为w的M阶多项式
def poly(w, x):
    func = np.poly1d(w)
    return func(x)

# 生成x的0～M次方
def generator(x, M):
    res = np.zeros((x.shape[0], M))
    for m in range(1, M+1):
        res[:, M-m] = pow(x, m)
    res = np.concatenate((res, np.ones((x.shape[0],1))), axis=1)
    return res

if __name__ == "__main__":
    # 采样十个样本点
    x = np.linspace(0, 1, 10)
    y = objective_func(x).reshape(10, 1)
    for M in [0, 1, 3, 9]:
        x_ = generator(x, M)
        # 解析解(x^T x)^-1 x^T y
        tmp_1 = np.linalg.inv(np.matmul(x_.T, x_))
        tmp_2 = np.matmul(x_.T, y)
        w = np.matmul(tmp_1, tmp_2)
        weight = [i[0] for i in w]
        # 画图
        plt.clf()
        x_points = np.linspace(0, 1, 1000)
        y_points = poly(weight, x_points)
        plt.plot(x_points, y_points, label='fitted curve')
        plt.plot(x, y, 'ro', label='sample')
        plt.legend()
        plt.savefig("{}".format(M))
        print("{}:{}".format(M,w))

        
