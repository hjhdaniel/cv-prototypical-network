import os
import random
import pickle
import shutil
import glob

def moveAllFilesinDir(srcDir, dstDir):
    # Check if both the are directories
    if os.path.isdir(srcDir) and os.path.isdir(dstDir) :
        # Iterate over all the files in source directory
        for filePath in glob.glob(srcDir + '\*'):
            # Move each file to destination Directory
            shutil.move(filePath, dstDir)
    else:
        print("srcDir & dstDir should be Directories")

def shuffle():
    # Shuffle classes into Train, Val, Test
    file_path = os.path.join('../dataset','cifar100/','all/')
    file_list = []
    for (dirpath, dirnames, filenames) in os.walk(file_path):
        file_list.extend(dirnames)
        break

    assert len(file_list) == 100

    random.shuffle(file_list)
    train_class = file_list[:60]
    val_class = file_list[60:80]
    test_class = file_list[80:]

    assert len(train_class) == 60
    assert len(val_class) == 20
    assert len(test_class) == 20

    return train_class, val_class, test_class, file_list

def split(train_class, val_class, test_class, file_list):
    # Split into Train, Val, Test by movin from 'all' folder
    for batch in ('test', 'val' ,'train'):
        file_path = os.path.join('../dataset','cifar100/')
        for i, filename in enumerate(file_list):
            if filename in train_class:
                current_batch = 'train'
            elif filename in val_class:
                current_batch = 'val'
            elif filename in test_class:
                current_batch = 'test'
            else:
                raise Exception('filename {} is not found'.format(filename))

            src = os.path.join(file_path,'all',filename)
            dest = os.path.join(file_path,current_batch)
            os.makedirs(dest, exist_ok=True)
            shutil.move(src,dest)

if __name__ == "__main__":
    train_class, val_class, test_class, file_list = shuffle()
    split(train_class, val_class, test_class, file_list)