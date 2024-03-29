import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

DATADIR = os.getcwd() + '/assets/Cats and Dogs data/PetImages'
CATEGORIES = ['Dog', 'Cat']
IMG_SIZE = 50

training_data = []


def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)  # 0 dog / 1 cat
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resizing all to be the same
                # plt.imshow(new_array, cmap='gray')
                # plt.show()
                training_data.append([new_array, class_num])
            except Exception as e:
                pass


create_training_data()
print(len(training_data))

import random

random.shuffle(training_data)

for sample in training_data[:10]:
    print(sample[1])

