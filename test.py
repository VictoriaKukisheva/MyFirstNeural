import cv2
from tensorflow.keras.models import Model, load_model
from fastai.vision import *
from sklearn.metrics import accuracy_score

image_size = 299
batch_size = 32

model = load_model("inceptionv3_fine_tuned.h5")

img_path = 'photoes/val/COVID19/COVID19(511).jpg'

labels = {0: 'COVID19', 1: 'NORMAL', 2: 'PNEUMONIA', 3: 'TURBERCULOSIS'}

img_width, img_height = 224, 224
image = cv2.imread(img_path)
image = cv2.resize(image, (img_width, img_height))
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = image[np.newaxis, ...]
image = image / 255.

prediction = model.predict(image)
prediction = np.squeeze(prediction)

for i in labels:
    print(labels[i], ':', prediction[i])


prediction = np.argmax(prediction)
output = labels[prediction]

print()
print('The image which you entered, prediction is:',output)


