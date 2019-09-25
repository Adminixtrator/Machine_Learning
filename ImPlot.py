import os
import cv2
import matplotlib.pyplot as plt
from matplotlib.image import imread

dir = '210503/'

for i in os.listdir(dir):
    image = cv2.imread(dir+i)
    plt.imshow(image)
    plt.show()

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()
knn.fit(i)
