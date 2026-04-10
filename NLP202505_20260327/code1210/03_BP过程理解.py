# -*- coding: utf-8 -*-
"""
Create Date Time : 2025/12/10 21:17
Create User : 19410
Desc : BP过程理解
"""

import numpy as np
import matplotlib.pyplot as plt

# 学习率 -> 参数更新时候的超参数
alpha = 0.5
# 需要学习的模型参数
_w = np.asarray([
    0.1, 0.15, 0.2, 0.25, 0.3, 0.35,
    0.4, 0.45, 0.5, 0.55, 0.6, 0.65
])
# 固定不变的模型参数
_b = np.asarray([0.35, 0.65])
# 样本 - 输入特征x
_x = np.asarray([
    [5., 10.0]
])
# 样本 - 目标特征y
_y = np.asarray([
    [0.01, 0.99]
])


def w(i):
    return _w[i - 1]


def b(i):
    return _b[i - 1]


def x(i):
    return _x[0][i - 1]


def y(i):
    return _y[0][i - 1]


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def update_w(i, gd):
    # 基于梯度的参数更新公式
    _w[i - 1] = _w[i - 1] - alpha * gd


def run():
    ##### 前向过程 -> 计算损失
    # 第一层的第一个神经元
    net_h1 = w(1) * x(1) + w(2) * x(2) + b(1)
    out_h1 = sigmoid(net_h1)

    # 第一层第二个神经元
    net_h2 = w(3) * x(1) + w(4) * x(2) + b(1)
    out_h2 = sigmoid(net_h2)

    # 第一层的第三个神经元
    net_h3 = w(5) * x(1) + w(6) * x(2) + b(1)
    out_h3 = sigmoid(net_h3)

    # 第二层的第一个神经元(同时也是第一个输出节点)
    net_o1 = out_h1 * w(7) + out_h2 * w(9) + out_h3 * w(11) + b(2)
    out_o1 = sigmoid(net_o1)

    # 第二层的第二个神经元(同时也是第二个输出节点)
    net_o2 = out_h1 * w(8) + out_h2 * w(10) + out_h3 * w(12) + b(2)
    out_o2 = sigmoid(net_o2)

    # 第一个输出节点的损失
    loss1 = 0.5 * (out_o1 - y(1)) ** 2
    # 第二个输出节点的损失
    loss2 = 0.5 * (out_o2 - y(2)) ** 2
    # 当前样本的损失
    loss = loss1 + loss2

    # print(net_h1, out_h1)
    # print(net_h2, out_h2)
    # print(net_h3, out_h3)
    # print(net_o1, out_o1)
    # print(net_o2, out_o2)
    # print(loss1, loss2, loss)

    # 反向过程 -> 基于损失求解关于每个参数的梯度值，并基于梯度更新参数
    # 求解各个参数对应的梯度值
    gd_net_o1 = (out_o1 - y(1)) * out_o1 * (1 - out_o1)
    gd_net_o2 = (out_o2 - y(2)) * out_o2 * (1 - out_o2)
    gd_list = [
        (gd_net_o1 * w(7) + gd_net_o2 * w(8)) * out_h1 * (1 - out_h1) * x(1),
        (gd_net_o1 * w(7) + gd_net_o2 * w(8)) * out_h1 * (1 - out_h1) * x(2),
        (gd_net_o1 * w(9) + gd_net_o2 * w(10)) * out_h2 * (1 - out_h2) * x(1),
        (gd_net_o1 * w(9) + gd_net_o2 * w(10)) * out_h2 * (1 - out_h2) * x(2),
        (gd_net_o1 * w(11) + gd_net_o2 * w(12)) * out_h3 * (1 - out_h3) * x(1),
        (gd_net_o1 * w(11) + gd_net_o2 * w(12)) * out_h3 * (1 - out_h3) * x(2),
        gd_net_o1 * out_h1,  # w7的梯度值
        gd_net_o2 * out_h1,
        gd_net_o1 * out_h2,
        gd_net_o2 * out_h2,
        gd_net_o1 * out_h3,
        gd_net_o2 * out_h3
    ]
    # 基于梯度值更新参数
    for i in range(1, 13):
        update_w(i, gd_list[i - 1])

    # print(f"参数梯度:{gd_list}")
    # 返回 (当前样本的当前损失，(当前样本的第一个输出节点值，当前样本的第二个输出节点值))
    return loss, (out_o1, out_o2)


if __name__ == '__main__':
    # run()
    _losses = []
    _r = run()
    _losses.append(_r[0])  # 将当前这次计算的样本损失保存到列表中
    print(f"当前样本损失: {_r[0]}")
    print(f"当前样本的预测节点/输出节点值: {_r[1]}")
    print(_w)

    for i in range(10000):
        _r = run()
        _losses.append(_r[0])  # 将当前这次计算的样本损失保存到列表中

    print("=" * 100)
    print("总的迭代更新完成后的结果:")
    print(f"当前样本损失: {_r[0]}")
    print(f"当前样本的预测节点/输出节点值: {_r[1]}")
    print(_w)

    # 损失的可视化
    plt.plot(_losses)
    plt.show()
