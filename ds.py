import pickle
import numpy as np
from os import listdir
from os.path import isfile, join
import os
import matplotlib.pyplot as plt


def save_image(image, label, idx, root_directory):
    """Saves an image in a subdirectory based on its label."""
    directory = os.path.join(root_directory, f"label_{label}")
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = os.path.join(directory, f"img_{idx}.png")
    plt.imsave(filename, image)

# Function to unpickle the dataset
def unpickle_all_data(directory):
    
    # Initialize the variables
    train = dict()
    test = dict()
    train_x = []
    train_y = []
    test_x = []
    test_y = []
    
    # Iterate through all files that we want, train and test
    # Train is separated into batches
    for filename in listdir(directory):
        if isfile(join(directory, filename)):
            
            # The train data
            if 'data_batch' in filename:
                print('Handing file: %s' % filename)
                
                # Opent the file
                with open(directory + '/' + filename, 'rb') as fo:
                    data = pickle.load(fo, encoding='bytes')

                if 'data' not in train:
                    train['data'] = data[b'data']
                    train['labels'] = np.array(data[b'labels'])
                else:
                    train['data'] = np.concatenate((train['data'], data[b'data']))
                    train['labels'] = np.concatenate((train['labels'], data[b'labels']))
            # The test data
            elif 'test_batch' in filename:
                print('Handing file: %s' % filename)
                
                # Open the file
                with open(directory + '/' + filename, 'rb') as fo:
                    data = pickle.load(fo, encoding='bytes')
                
                test['data'] = data[b'data']
                test['labels'] = data[b'labels']
    
    # Manipulate the data to the propper format
    for image in train['data']:
        train_x.append(np.transpose(np.reshape(image,(3, 32,32)), (1,2,0)))
    train_y = [label for label in train['labels']]
    
    for image in test['data']:
        test_x.append(np.transpose(np.reshape(image,(3, 32,32)), (1,2,0)))
    test_y = [label for label in test['labels']]
    
    # Transform the data to np array format
    train_x = np.array(train_x)
    train_y = np.array(train_y)
    test_x = np.array(test_x)
    test_y = np.array(test_y)
    
    # Directory to save images
    save_dir_train = 'E:/PhD/Datasets/cifar10 - Copy/train'
    save_dir_test = 'E:/PhD/Datasets/cifar10 - Copy/test'

    # Save train images
    for idx, (image, label) in enumerate(zip(train_x, train_y)):
        save_image(image, label, idx, save_dir_train)

    # Save test images
    for idx, (image, label) in enumerate(zip(test_x, test_y)):
        save_image(image, label, idx, save_dir_test)

    return (train_x, train_y), (test_x, test_y)

# Run the function with and include the folder where the data are
(x_train, y_train), (x_test, y_test) = unpickle_all_data('E:\PhD\Datasets' + '/cifar-10-batches-py/')
print(x_train.shape)





# with open('data/validation/test-x', 'wb') as handle:
#     pickle.dump(x_test, handle, protocol=pickle.HIGHEST_PROTOCOL)
# with open('data/validation/test-y', 'wb') as handle:
#     pickle.dump(y_test, handle, protocol=pickle.HIGHEST_PROTOCOL)
# with open('data/train/train-x', 'wb') as handle:
#     pickle.dump(x_train, handle, protocol=pickle.HIGHEST_PROTOCOL)
# with open('data/train/train-y', 'wb') as handle:
#     pickle.dump(y_train, handle, protocol=pickle.HIGHEST_PROTOCOL)