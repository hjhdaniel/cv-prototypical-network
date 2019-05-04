import os
import pandas as pd
import random

def split():
    image_class_labels = pd.read_csv(os.path.join('../dataset/CUB_200_2011/', 'CUB_200_2011', 'image_class_labels.txt'),
                                            sep=' ', names=['img_id', 'target'])
    class_list = list(range(1,201))
    random.shuffle(class_list)
    train_class = class_list[:100]
    val_class = class_list[100:150]
    test_class = class_list[150:]

    f = open(os.path.join('../dataset/CUB_200_2011/', 'CUB_200_2011', 'split.txt'), "w")
    for i in range(1,11789):
        img_class = image_class_labels.target[i-1]
        # print(img_class)
        if img_class in train_class:
            f.write(str(i) + " 0\n")
        elif img_class in val_class:
            f.write(str(i) + " 1\n")
        elif img_class in test_class:
            f.write(str(i) + " 2\n")
        else:
            raise Exception('class {} is not found'.format(img_class))
    f.close()

if __name__ == "__main__":
    split()