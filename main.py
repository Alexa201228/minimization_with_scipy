from scipy.optimize import minimize
import numpy as np
from sympy import *


# Целевая функция:
def func(a_coef, b_coef):
    fun = lambda x: a_coef[0] + a_coef[1] * x[0] + a_coef[2] * x[0] ** 2 + b_coef[0] + b_coef[1] * x[1] + b_coef[2] * x[1] ** 2
    return fun


def con(d):
    cons = ({'type': 'eq', 'fun': lambda x: x[0] + x[1] - d},
            {'type': 'ineq', 'fun': lambda x: x[0] - 0},
            {'type': 'ineq', 'fun': lambda x: x[1] - 0})

    return cons


if __name__ == "__main__":
    print('Введите 3 коеффициента первого способа чарез пробел')
    a_coef = list(map(float, input().strip().split(' ')))
    print('Введите 3 коеффициента второго способа чарез пробел')
    b_coef = list(map(float, input().strip().split(' ')))
    print('Введите значение ограничения')
    d = float(input())
    cons = con(d)
    x0 = np.array((0.5, 0.5))  # Установить начальное значение, установка начального значения очень важна, легко сходиться к другой экстремальной точке, рекомендуется попробовать еще несколько значений

    # Решать#
    res = minimize(func(a_coef, b_coef), x0, method='SLSQP', constraints=cons)

    print(f'x1={round(res.x[0], 3)};  x2={round(res.x[1], 3)}')
    print(f'Оптимальное решение: {res.fun}')
    print('Проверка:')
    x, y = symbols('x y', real=True)
    f = a_coef[0] + a_coef[1] * x + a_coef[2] * x ** 2 + b_coef[0] + b_coef[1] * y + b_coef[2] * y ** 2
    H = np.array(hessian(f, (x, y)))
    H = H.astype('float64')
    print(f'Матрица Гессе:\n{H}')
    print(f'Детерминант матрицы:\n{round(np.linalg.det(H))}')