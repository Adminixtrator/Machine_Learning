import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import os
import cv2
#from matplotlib.image import  imread

plate_number_dir =  'Plate_numbers/' #'./plate_numbers/'
negative_images_dir = 'negative_images/' #'./negative_images/'
ROWS = 64
COLS = 64
CHANNELS = 3

knn = KNeighborsClassifier()

def read_image(file_path):
    img = cv2.imread(file_path, cv2.IMREAD_COLOR)
    resized_img = cv2.resize(img, (ROWS, COLS), interpolation = cv2.INTER_AREA)
    return resized_img

def prep_data(images):
    m = len(images)
    n_x = ROWS*COLS*CHANNELS
    X = np.ndarray((n_x,m), dtype = np.uint8)
    y = np.zeros((1,m))
    print("X.shape is {}".format(X.shape))
    for i,image_file in enumerate(images):

        image = read_image(image_file)
#        print(i, 'done')
        X[:,i] = np.squeeze(image.reshape((n_x,1)))

        if '-' in image_file.lower():
            y[0,i] = 1

        elif 'glass' in image_file.lower():
          y[0,i] = 0

          
        if i%100 == 0 :
            print("Proceed {} of {}".format(i, m))

    return X,y

plate_numbers_img = [plate_number_dir+i for i in os.listdir(plate_number_dir)]
negative_images_img = [negative_images_dir+i for i in os.listdir(negative_images_dir)]

plate_img, negative_img = prep_data(plate_numbers_img + negative_images_img)

#clf = LogisticRegressionCV()

plate_img_lr, neg_img_lr = plate_img.T, negative_img.T

#knn.fit(plate_img_lr, neg_img_lr)
print("Model accuracy: {:.2f}%".format(knn.score(plate_img_lr, neg_img_lr)*100))

for i in np.random.randint(0, len(plate_img_lr), 5) :
    show_image_prediction(plate_img_lr, i, knn)










