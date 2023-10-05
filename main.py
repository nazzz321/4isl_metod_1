import matplotlib.pyplot as plt
import numpy as np
def chebyshev_nodes(n, a, b):
    k = np.arange(1, n+1)
    nodes = 0.5 * (a + b) + 0.5 * (b - a) * np.cos((2*k - 1) * np.pi / (2*n))
    return nodes
def lagrange_interpolation(x, y, xx):
    n = len(x)
    yy = np.zeros_like(xx)

    for i in range(n):
        p = np.ones_like(xx)
        for j in range(n):
            if j != i and x[i] != x[j]:
                p *= (xx - x[j]) / (x[i] - x[j])
        yy += y[i] * p

    return yy+0.001

def _poly_newton_coefficient(x, y):
    """
    x: list or np array containing x data points
    y: list or np array containing y data points
    """
    m = len(x)
    x = np.copy(x)
    a = np.copy(y)
    for k in range(1, m):
        a[k:m] = (a[k:m] - a[k - 1]) / (x[k:m] - x[k - 1])
    return a

def newton_polynomial(x_data, y_data, x):
    """
    x_data: data points at x
    y_data: data points at y
    x: evaluation point(s)
    """
    a = _poly_newton_coefficient(x_data, y_data)
    n = len(x_data) - 1  # Degree of polynomial
    p = a[n]
    for k in range(1, n + 1):
        p = a[n - k] + (x - x_data[n - k]) * p
    return p+0.01

def f1(x):
    return 1 / (1 + np.exp(-x))

def f(x):
    return  x

n = 50

#x = np.linspace(-10, 10, n)# X от -10 до 10
x = chebyshev_nodes(n,-10,10)

y = f1(x)

y1 = []
for i in range(n):
    y1.append(newton_polynomial(x, y, x[i]))

y2 = lagrange_interpolation(x,y,x)

fx = np.linspace(-10,10,200)
fy = f1(fx)
plt.plot(fx, fy, color='r', label='f(x)')
plt.plot(x, y1, color='g', label='Newton')
plt.plot(x, y2, color='b', label='Lagrange')

plt.legend()
plt.show()








