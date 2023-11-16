import numpy as np
import matplotlib.pyplot as plt


# Define the kernel
def kernel_gaussian(a, b, l=1.0, s=1.0):
    '''
         k = s2*exp(-(a-b')**2/(2*l**2))
    :param a: shape of (N, 1)
    :param b: shape of (N, 1)
    :param l:
    :param s2:
    :return:
    '''
    sqdist = (a ** 2).reshape(-1, 1) + (b ** 2).reshape(1, -1) - 2 * (a @ b.T)
    # np.sum( ,axis=1) means adding all elements columnly; .reshap(-1, 1) add one dimension to make (n,) become (n,1)
    k = s**2 * np.exp(-sqdist/(2*l**2))
    return k


def plot_kernel(k, s=None):
    plt.imshow(k, cmap='rainbow')
    plt.colorbar()
    if s:
        plt.title(s)
    plt.tight_layout()
    plt.show()

