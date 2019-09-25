from matplotlib import pyplot
from matplotlib.image import imread

folder = 'test/'

for i in range(9):
    pyplot.subplot(330 + 1 + i)
    filename = folder + 'img' + str(i) + '.jpg'
    image = imread(filename)
    pyplot.imshow(image)

pyplot.show()
