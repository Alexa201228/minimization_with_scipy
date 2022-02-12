from scipy.optimize import minimize
import numpy as np
from sympy import *


# Целевая функция:
def func(sigma_coef):
    fun = lambda x: sigma_coef[0] * sigma_coef[0] * x[0] * x[0] + \
                    sigma_coef[1] * sigma_coef[1] * x[1] * x[1] + \
                    sigma_coef[2] * sigma_coef[2] * x[2] * x[2] + \
                    sigma_coef[3] * sigma_coef[3] * x[3] * x[3] + \
                    sigma_coef[4] * sigma_coef[4] * x[4] * x[4] + \
                    sigma_coef[5] * sigma_coef[5] * x[5] * x[5] + \
                    sigma_coef[6] * sigma_coef[6] * x[6] * x[6]
    return fun


# Ограничения, включая ограничения равенства и ограничения неравенства
def con(d, mr_coef):
    e = 1e-15 # Для обозначения нуля
    cons = ({'type': 'eq', 'fun': lambda x: mr_coef[0] * x[0] +
                                            mr_coef[1] * x[1] +
                                            mr_coef[2] * x[2] +
                                            mr_coef[3] * x[3] +
                                            mr_coef[4] * x[4] +
                                            mr_coef[5] * x[5] +
                                            mr_coef[6] * x[6] - d},
            {'type': 'eq', 'fun': lambda x: x[0] + x[1] + x[2] + x[3] + x[4] + x[5] + x[6] - 1},
            {'type': 'ineq', 'fun': lambda x: x[0] - e},
            {'type': 'ineq', 'fun': lambda x: x[1] - e},
            {'type': 'ineq', 'fun': lambda x: x[2] - e},
            {'type': 'ineq', 'fun': lambda x: x[3] - e},
            {'type': 'ineq', 'fun': lambda x: x[4] - e},
            {'type': 'ineq', 'fun': lambda x: x[5] - e},
            {'type': 'ineq', 'fun': lambda x: x[6] - e},)

    return cons


if __name__ == "__main__":
    print('Введите 7 значений среднего квадратичного чарез пробел')
    sigma_coef = list(map(float, input().strip().split(' ')))
    print('Введите 7 значений матожидания чарез пробел')
    mr_coef = list(map(float, input().strip().split(' ')))
    print('Введите значение общего матожидания')
    m = float(input())
    cons = con(m, mr_coef)

    x0 = np.array((0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1))  # Установить начальное значение

    res = minimize(func(sigma_coef), x0, method='SLSQP', constraints=cons)

    print(f'Оптимальное решение: {res.fun}')
    print('Проверка:')
    x, y, z, a, b, c, d = symbols('x y z a b c d', real=True)
    f = sigma_coef[0] * sigma_coef[0] * x * x + \
        sigma_coef[1] * sigma_coef[1] * y * y + \
        sigma_coef[2] * sigma_coef[2] * z * z + \
        sigma_coef[3] * sigma_coef[3] * a * a + \
        sigma_coef[4] * sigma_coef[4] * b * b + \
        sigma_coef[5] * sigma_coef[5] * c * c + \
        sigma_coef[6] * sigma_coef[6] * d * d
    H = np.array(hessian(f, (x, y, z, a, b, c, d)))
    H = H.astype('float64')
    print(f'Матрица Гессе:\n{H}')
    print(f'Детерминант матрицы:\n{round(np.linalg.det(H))}')

