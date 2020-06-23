import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from PIL import Image

def plot(data, title):
    plot.i += 1
    plt.subplot(2,2,plot.i)
    plt.imshow(data, cmap='gray', vmin=0, vmax=255)
    plt.gray()
    plt.title(title)
plot.i = 0

# Load the data...
im = Image.open('lena.png').convert('L')
# img = Image.open('lena.png').convert('LA')
# img.save('greyscale.png')
data = np.array(im, dtype=int)
# plt.imshow(data)
plot(data, 'Original')
print(data)

kernelLow = np.array([[1, 4, 6, 4, 1],
                      [4, 16, 24, 16, 4],
                      [6, 24, 36, 24, 6],
                      [4, 16, 24, 16, 4],
                      [1, 4, 6, 4, 1]])
lowpass_3x3Low = ndimage.convolve(data, kernelLow)/256

# oriLow = data + highpass_3x3Low
plot(lowpass_3x3Low, 'Simple 5x5 Highpass')

# A very simple and very narrow highpass filter
kernel = np.array([[0, -1, 0],
                   [-1, 5, -1],
                   [0, -1, 0]])
highpass_3x3 = ndimage.convolve(data, kernel)
plot(highpass_3x3, 'Simple 3x3 Highpass')

# Another way of making a highpass filter is to simply subtract a lowpass
# filtered image from the original. Here, we'll use a simple gaussian filter
# to "blur" (i.e. a lowpass filter) the original.
# lowpass = ndimage.gaussian_filter(data, 3)
gauss_highpass = data - lowpass_3x3Low
gauss_highpass = data + gauss_highpass
plot(gauss_highpass, r'Gaussian Highpass, $\sigma = 3 pixels$')

plt.show()
