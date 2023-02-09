import cv2
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.layers import BatchNormalization, Activation
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from fastai.vision import *

path = Path('photoes/val')

image_size = 299
batch_size = 32

# Каталог с данными для обучения
train_dir = 'photoes/train'
test_dir = 'photoes/val'

# Генератор с дополнением данных (data augmentation)
train_datagen = ImageDataGenerator(rescale=1. / 255,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    fill_mode='nearest')

# Генератор для обучения
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(image_size, image_size),
    batch_size=batch_size,
    class_mode='categorical')

# Генератор для тестирования без дополнения данных
test_datagen = ImageDataGenerator(rescale=1. / 255)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(image_size, image_size),
    batch_size=batch_size,
    class_mode='categorical')

num_classes = len(train_generator.class_indices)
print(num_classes)
print(train_generator.class_indices)

# Загружаем предварительно обученную нейронную сеть Inception
pretrained_model = InceptionV3(weights='imagenet', include_top=False)
pretrained_model.summary()
pretrained_model.trainable = False

# Создаем составную нейронную сеть
x = pretrained_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(num_classes, activation='sigmoid')(x)
model = Model(inputs=pretrained_model.input, outputs=predictions)

# Обучаем составную нейронную сеть на основе InceptionV3
model.compile(loss='binary_crossentropy',
              optimizer=RMSprop(lr=1e-4),
              metrics=['accuracy'])

filepath='C:/Users/ВВК/PycharmProjects/pythonProject/inceptionv3_best.h5'
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1,
                             save_best_only=True, mode='max')
learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy',
                                            patience=3,
                                            verbose=1,
                                            factor=0.5,
                                            min_lr=0.00001)
callbacks_list = [checkpoint, learning_rate_reduction]

steps_per_epoch = 40
history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    validation_data=test_generator,
    validation_steps=5,
    callbacks=callbacks_list,
    epochs=20,
    verbose=2)

# Дообучение сверточных слоев Inception V3
model.load_weights("inceptionv3_best.h5")

model.compile(loss='binary_crossentropy',
              optimizer=RMSprop(lr=1e-4),
              metrics=['accuracy'])

filepath="inceptionv3_fine_tuned.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint, learning_rate_reduction]

steps_per_epoch = 40
history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    validation_data=test_generator,
    validation_steps=5,
    callbacks=callbacks_list,
    epochs=5,
    verbose=2)

# Тестируем модель
model = load_model("inceptionv3_fine_tuned.h5")

img_path = 'photoes/val/NORMAL/IM-0001-0001.jpeg'

img = image.load_img(img_path, target_size=(image_size, image_size))
plt.imshow(img)
plt.show()

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