"""
Najprostsza wersja regresji
liniowej metodą najmniejszych kwadratów
ALE NIE ANALITYCZNIE
czyli metodą numeryczną GRADIENTU PROSTEGO
y = a * x + b

Autor: Piotr Owerko
"""

import math
import random


def weighted_sum(x, w1, b1):
    z = w1 * x + b1
    return z

def regr_gradient(x_train, y_train, w1, b1):
    d_cost_d_b1 = 0
    d_cost_d_w1 = 0
    for j in range(len(x_train)):
        d_cost_d_b1 += (weighted_sum(x_train[j], w1, b1) - y_train[j])
        d_cost_d_w1 += (weighted_sum(x_train[j], w1, b1) - y_train[j]) * x_train[j]
    return [(1 / len(x_train)) * d_cost_d_b1, (1 / len(x_train)) * d_cost_d_w1]

def regr_grad_desc(x_train, y_train, l_rate, n_loops):
    """finds optimal values of b1 and w1"""
    b1_opt = random.uniform(0, 5)
    w1_opt = random.uniform(0, 5)
    for i in range(n_loops):
        b1_opt -= l_rate * regr_gradient(x_train, y_train, w1_opt, b1_opt)[0]
        w1_opt -= l_rate * regr_gradient(x_train, y_train, w1_opt, b1_opt)[1]
    return [b1_opt, w1_opt]
        
def main():
    x_train = (1, 2, 3, 4)
    y_train = [(0.5 * x + 1 + random.uniform(-0.1, 0.1)) for x in x_train]
    print(regr_grad_desc(x_train, y_train, 0.01, 10000))


if __name__ == "__main__":
    main()
