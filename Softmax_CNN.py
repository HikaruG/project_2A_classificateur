#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 14:59:28 2018

@author: Valentin
"""

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing import image
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import glob
import os.path

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution

#1st Layer
classifier.add(Conv2D(64, (3, 3), input_shape = (64, 64, 3), strides=(1,1) ,activation = 'relu'))


# 2nd Layer 
classifier.add(Conv2D(64, (3, 3), activation = 'relu'))

# 3rd Layer 
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))


# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(units = 256, activation = 'relu'))
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 3, activation = 'softmax'))

# Compiling the CNN
classifier.compile(optimizer = 'sgd', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Part 2 - Fitting the CNN to the images





train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('/Users/Valentin/Desktop/Artificial_Intelligence/ImageClassification/data2/train',
                                                 target_size = (64, 64),
                                                 batch_size = 64,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory('/Users/Valentin/Desktop/Artificial_Intelligence/ImageClassification/data2/validation',
                                            target_size = (64, 64),
                                            batch_size = 64,
                                            class_mode = 'categorical')



#classifier.fit_generator(training_set,
#                         steps_per_epoch = 2000,
#                         epochs = 10,
#                         validation_data = test_set,
#                         validation_steps = 1500
                        
                    

#classifier.save_weights('/Users/Valentin/Desktop/Artificial_Intelligence/ImageClassification/testSoftmax.h5')

#Chargement des poids W du modèle entrainé
classifier.load_weights('/Users/Valentin/Desktop/Artificial_Intelligence/ImageClassification/testSoftmax.h5')

#Affichage de la structure du modèle
print("\n Voici la structure de votre réseau de neurones :")
classifier.summary()


#Evaluation du modèle sur un échantillon donné
s=next(training_set)
score=classifier.evaluate(s[0],s[1])
score=score[1]*100
print("Le taux de reussite de ce modele est de : 0,%d " % score )
#Demande une entrée à l'utilisateur
addr=raw_input("Entrez le nom de l'image à analyser \n")
addr="/Users/Valentin/Desktop/Artificial_Intelligence/ImageClassification/data2/test/"+addr

#new_img=image.load_img('/Users/Valentin/Desktop/Artificial_Intelligence/ImageClassification/data/test/test.jpg')
#new_img_resized=image.load_img('/Users/Valentin/Desktop/Artificial_Intelligence/ImageClassification/data/test/test.jpg',target_size=(64,64))



new_img=image.load_img(addr)
new_img_resized=image.load_img(addr,target_size=(64,64))


#Affichage du Résultat sur une nouvelle image présente dans un dossier

print("\n Voici l'image que vous souhaitez analyser: ")
plt.imshow(new_img)
plt.show()
new_data = np.array(new_img_resized)
new_data=np.expand_dims(new_data, axis=0)

prediction=classifier.predict(new_data,batch_size=None, verbose=0)
prediction=prediction[0].tolist()
s=prediction.index(max(prediction))

if s==0:
    print"Il s'agit d'une voiture"
elif s==1:  
    print"Il s'agit d'un chat"
else:
    print"Il s'agit d'un chien"
    

#print(prediction)
#
#print("C'est un chat avec une probablité de : %s " %(prediction[0][1]))
#print("C'est un chien avec une probablité de : %s " %(prediction[0][2]))
#print("C'est une voiture avec une probablité de : %s " %(prediction[0][0]))


