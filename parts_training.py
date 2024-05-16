import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os
from tqdm import tqdm
import os
from os import makedirs
from os import listdir
from shutil import copyfile
from random import seed
from random import random
from tensorflow.keras.utils import to_categorical 
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical # convert to one-hot-encoding
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from tensorflow.keras.optimizers import RMSprop,Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from keras.regularizers import l2
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


ArtBank = 'images'

# Art_Categories = ['abstract','animal-painting','cityscape','flower-painting','landscape', 'portrait', 'still-life']
Art_Categories = ['abstract','genre-painting','landscape', 'portrait']

#! image size - important parameter
Image_size=128


l2_reg = 0.001
###

model = Sequential()
### modified 14:51 - changed filters to 4 from 8
model.add(Conv2D(filters = 4, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu', input_shape = (Image_size,Image_size,3)))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))
### modified 14:51 - changed filters to 8 from 16
model.add(Conv2D(filters = 8, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))
# fully connected
model.add(Flatten())
### modified 14:58
model.add(Dense(256, activation = "relu", kernel_regularizer=l2(l2_reg)))
###
model.add(Dropout(0.5))
model.add(Dense(len(Art_Categories), activation = "softmax"))

optimizer = Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999)

model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])

epochs = 30  # optimal choice
batch_size = 64

parts = 20

for iteration in range(0,parts):
    training_data=[]

    train_start = iteration * (15000 / parts)
    train_end = train_start + (15000 / parts) - 1

    def creating_training_data():
        for category in Art_Categories:
            path=os.path.join(ArtBank, category)
            class_num=Art_Categories.index(category)
            i=0
            ld = sorted(os.listdir(path))
            print(path)
            for img in ld:
                if i<train_start:
                    i+=1
                    continue
                try:
                    img_array=cv2.imread(os.path.join(path,img))
                    new_array=cv2.resize(img_array,(Image_size,Image_size))
                    
                    ### added 14:46
                    # flipped_img_array = cv2.flip(img_array, 1)
                    # training_data.append([flipped_img_array / 255, class_num])
                    ###
                    
                    training_data.append([new_array/255,class_num])#divided by 255
                except Exception as e:
                    pass
                if i==train_end: # data size for category
                    break
                # if i%(len(ld) / 10)==0:
                    # print(f"{(i/2999) * 100}%",end="\r")
                i+=1

    creating_training_data()
    
    Xr_images = []
    yr_labels = []
    for categories, label in training_data:
        Xr_images.append(categories)
        yr_labels.append(label)

    #Test train split
    X_Images = np.array(Xr_images, dtype = 'float32')
    from tensorflow.keras.utils import to_categorical 
    y_Labels = to_categorical(yr_labels, num_classes = len(Art_Categories))
    from sklearn.model_selection import train_test_split
    X_train, X_test, Y_train, Y_test = train_test_split(X_Images, y_Labels, test_size=0.1, random_state=1)
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=(0.1), random_state=1) # 0.111 x 0.9 = 0.1
    print("x_train shape",X_train.shape)
    print("x_test shape",X_val.shape)
    print("y_train shape",Y_train.shape)
    print("y_test shape",Y_val.shape)
    
    # Define the L2 regularization parameter

    checkpoint_filepath = f'ckpt/v2/checkpoint{iteration}.model.keras'
    model_checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_filepath,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True)

    history = model.fit(X_train,Y_train,
                    epochs = epochs, validation_data = (X_val,Y_val), steps_per_epoch=((X_train.shape[0] // batch_size) - 1), callbacks=[model_checkpoint_callback])
    
    
    
    model.save(f'models/v2/Art_Classification_v2_{iteration}.keras')
    
    
    plt.plot(history.history['val_loss'], color='b', label="validation loss")
    plt.title("Validation Loss")
    plt.xlabel("Number of Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
    # plot validation accuracy
    plt.plot(history.history['val_accuracy'], color='b', label="validation loss")
    plt.title("Validation Accuracy")
    plt.xlabel("Number of Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()
    
    Y_pred = model.predict(X_test)
 
    Y_pred_classes = np.argmax(Y_pred,axis = 1) 

    Y_true = np.argmax(Y_test,axis = 1) 
    
    confusion_matrix_1 = confusion_matrix(Y_true, Y_pred_classes) 
    
    confusion_matrix_2 = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_1, display_labels=Art_Categories)
    confusion_matrix_2.plot()
    
    print("Accuracy on training data is",accuracy_score(Y_true,Y_pred_classes))
