"""
Najprostsza wersja regresji
liniowej metodą najmniejszych kwadratów
ALE NIE ANALITYCZNIE
czyli metodą numeryczną GRADIENTU PROSTEGO
y = a * x1 + b * x2 + c * x1 * x2 + d * x1 * x1 + e * x2 * x2 + f
a, b, c, d, e, f = w[] 

symbole:
theta = waga = w = Beta (z RSM)
liczba wag = liczba zmiennych xi bo zawsze X0 = 1
x_train (MACIERZ którego wiersze to kolejne zmienne xi a kolumny to wartości tych zmiennych trenujących model)
y_tran = Z (wektor pomierzonych wyników rzeczywistych)
z = y_pred = h0
Autor: Piotr Owerko
"""

import math
import random
#import matplotlib.pyplot as plt

def weighted_sum_gen(x_train, w):
    """
    finds predictions / values of hipoteses h0 / weighted sum   as list
    return list witch len is = len of x_train[0] = len of y_train
    """
    z = []
    for j in range(len(x_train[0])):
        z_local = 0
        for i in range(len(w)):
            z_local += (x_train[i][j] * w[i])
        z.append(z_local)
    return z

def regr_gradient_gen(x_train, y_train, w):
    """
    finds vector of gradient components
    returns list witch len = len of weights = len of features xi
    """
    d_cost_d_w = []
    for j in range(len(w)):
        grad_local = 0
        for i in range(len(x_train[0])):
            grad_local += (weighted_sum_gen(x_train, w)[i] - y_train[i]) * x_train[j][i]
        d_cost_d_w.append( grad_local)
    return [i * (1 / len(y_train)) for i in d_cost_d_w]

def regr_grad_desc_gen(x_train, y_train, l_rate, n_epochs):
    """finds optimal values of wieghts"""
    w_opt = len(x_train) * [random.uniform(0, 2)]
    for i in range(n_epochs):
        for j in range(len(x_train)):
            w_opt[j] -= l_rate * regr_gradient_gen(x_train, y_train, w_opt)[j]
    return w_opt
        
def main():
    x_train = [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [0, 1, 2, 3, 4, 0, 0, 0, 0, 2, 3], 
               [0, 0, 0, 0, 0, 1, 2, 3, 4, 3, 2], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 6],
               [0, 1, 4, 9, 16, 0, 0, 0, 0, 4, 9], [0, 0, 0, 0, 0, 1, 4, 9, 16, 9, 4]]
    y_train = [(x_train[0][i] * 3 + x_train[1][i] * 2.1 + x_train[2][i] * 0.4 
            + x_train[3][i] * 0.2 + x_train[4][i] * 0.5 + x_train[5][i] * 0.5
            + random.uniform(-0.01, 0.01)) for i in range(11)]
    # print(y_train)
    # fig = plt.figure()
    # ax = plt.gca(projection='3d')
    # ax.scatter(x_train[1], x_train[2], y_train)
    # plt.show()
    print(regr_grad_desc_gen(x_train, y_train, 0.01, 5000))


if __name__ == "__main__":
    main()