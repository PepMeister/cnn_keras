from keras.models import model_from_json

json_file = open('cnn_cat_or_dog.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
cnn = model_from_json(loaded_model_json)
cnn.load_weights("model.h5")
print("Loaded model from disk..")

# evaluate loaded model on test data
cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

import numpy as np
from keras.preprocessing import image

test_image = image.load_img('./CNN_Data/test_set/cats/cat.4999.jpg', target_size = (64,64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)

result = cnn.predict(test_image)
#print(result)
if result[0][0] == 1:
	prediction = 'dog'
else:
	prediction = 'cat'
print(prediction)