import matplotlib.pyplot as plt
import numpy as np

def _poly_newton_coefficient(x, y):
    """
    x: list or np array contanining x data points
    y: list or np array contanining y data points
    """
    m = len(x)
    x = np.copy(x)
    a = np.copy(y)
    for k in range(1, m):
        a[k:m] = (a[k:m] - a[k - 1])/(x[k:m] - x[k - 1])
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
        p = a[n - k] + (x - x_data[n - k])*p
    return p

def f1(x):
    return 1/(1+np.exp(-x))


x = np.linspace(-10, 10, 50) # X от -5 до 5
y = []
for i in range(50):
    y.append(f1(x[i]))

y1 = []
for i in range(50):
    y1.append(newton_polynomial(x,y,x[i]))
    print("{:8.10f} {:8.10f}".format(y[i], y1[i]))




fig, ax = plt.subplots()

ax.plot(x, y)
ax.plot(x, y1)

plt.show()

