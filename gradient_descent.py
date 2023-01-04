'''numpy만을 이용하여 back prop구현
계산그래프를 이용한 방식
'''

import numpy as np
import matplotlib.pyplot as plt


def function_2(x):
    return x[0] ** 2 + x[1] ** 2


def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)

    for idx in range(x.size):
        tmp_val = x[idx]
        x[idx] = tmp_val + h
        f_up = f(x)
        x[idx] = tmp_val - h
        f_down = f(x)
        grad[idx] = (f_up - f_down) / (2 * h)
        x[idx] = tmp_val

    return grad


'''unpacking list of numpy array which are [x,y] pairs to plot'''


# import numpy as np
# list_x = []
# x = np.array(np.array([4, 5]))
# y = np.array(np.array([1, 2]))
# list_x.append(x)
# list_x.append(y)
# x, y = zip(*list_x)
# import matplotlib.pyplot as plt
# plt.plot(x, y, 'o')
# plt.show()


#
def gradient_descent(f, init_x, lr, step_num):
    x = init_x
    xs = []
    xs.append(x)

    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x = x - lr * grad
        xs.append(x)
        print(f"step {i} : {x}")

    aa, bb = zip(*xs)
    plt.scatter(aa, bb)

    # for i in xs:
    #     plt.scatter(*i)
    plt.xlim([-3, 3])
    plt.ylim([-3, 3])

    for r in np.linspace(0, 5, 10):
        radius = r
        theta = np.linspace(0, 2 * np.pi, 100)
        circle = radius * np.exp(1j * theta)
        x = circle.real
        y = circle.imag
        plt.plot(x, y, '--')
    plt.axis('equal')
    plt.show()
    return x


'''draw circle with numpy complex number'''
# import matplotlib.pyplot as plt
# import numpy as np
#
# radius = 1.0
# theta = np.linspace(0, 2 * np.pi, 100)
# circle = radius * np.exp(1j * theta)
# x = circle.real
# y = circle.imag
# plt.plot(x, y, '--')
# plt.axis('equal')

# import numpy as np
# import matplotlib.pyplot as plt
#
# a = list()
# a.append(np.array([1, 1]))
# a.append(np.array([2, 2]))
# for i in a:
#     plt.scatter(*i)

#

init_x = np.array([-3.0, 4.0])

x_final = gradient_descent(function_2, init_x, 0.1, 100)

'''maker numpy array using user-defined lambda function '''
# import numpy as np
# squarer = lambda t: t ** 2
# x = np.array([1, 2, 3, 4, 5])
# squarer(x)