import numpy as np
from matplotlib import pylab as plt
from monai.transforms import LoadImage

loader = LoadImage()

def plot(path):
    data: np.ndarray
    data, meta = loader(path)
    print(data.size)
    print(data.min(), data.max())
    data = data.reshape((-1, data.shape[2]))
    plt.hist(data)
    plt.title(path)
    plt.show()

if __name__ == '__main__':
    pass
