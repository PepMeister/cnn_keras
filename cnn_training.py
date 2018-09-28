from keras.models import Sequential   #initialising neural network model as a sequential network
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator


cnn = Sequential()
cnn.add(Conv2D(32,(3,3), input_shape = (64, 64, 3), activation = 'relu'))
cnn.add(MaxPooling2D(pool_size = (2,2)))
cnn.add(Flatten())
cnn.add(Dense(units = 128, activation = 'relu'))
cnn.add(Dense(units = 1, activation = 'sigmoid'))
cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

train_datagen = ImageDataGenerator(rescale = 1./255, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)
training_set = train_datagen.flow_from_directory('./CNN_Data/training_set', target_size = (64, 64), batch_size = 32, class_mode = 'binary')
test_set = test_datagen.flow_from_directory('./CNN_Data/test_set', target_size = (64, 64), batch_size = 32, class_mode = 'binary')

cnn.fit_generator(training_set, steps_per_epoch = 8000, epochs = 25, validation_data = test_set, validation_steps = 2000)

import numpy as np
from keras.preprocessing import image

test_image = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg', target_size = (64,64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)

result = cnn.predict(test_image)
if result[0][0] == 1:
	prediction = 'dog'
else:
	prediction == 'cat'
print(prediction)