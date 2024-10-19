import tensorflow as tf
import logging
import random

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from keras.src.callbacks import EarlyStopping
from keras.utils import image_dataset_from_directory
from keras import Sequential
from keras import layers
from keras.optimizers.legacy import Adam
from keras.losses import SparseCategoricalCrossentropy
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
tf.get_logger().setLevel(logging.ERROR)

main_path = './dataset/'

img_size = (64,64)
batch_size = 64
import os
import matplotlib.pyplot as plt

class_names = os.listdir(main_path)
geometry = {
    0: "circle",
    1: "kite",
    2: "parallelogram",
    3: "rectangle",
    4: "rhombus",
    5: "square",
    6: "trapezoid",
    7: "triangle"
}

samples_per_class = []

for class_name in class_names:
    class_path = os.path.join(main_path, class_name)
    # Brojanje broja datoteka (uzoraka) u svakom podfolderu (klasi)
    num_samples = len(os.listdir(class_path))
    samples_per_class.append(num_samples)

plt.xlabel('Klase')
plt.ylabel('Broj odbiraka')
plt.title('Broj odbiraka po klasi')
bars = plt.bar(class_names, samples_per_class, color='salmon', width=0.3,
               edgecolor='black')

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, yval, round(yval), ha='center', va='bottom', color='black')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

Xtrain= image_dataset_from_directory(main_path,
                                     subset='training',validation_split=0.4,
                                    image_size=img_size, # skaliranje podataka
                                    batch_size=batch_size,
                                    seed=123, shuffle=True
                                     )

Xvaltest = image_dataset_from_directory(main_path,
                                    subset = 'validation',
                                    validation_split=0.4,
                                    image_size=img_size,
                                    batch_size=batch_size,
                                    seed=123, shuffle=True)

validation_size = int(0.5 * len(Xvaltest))
test_size = len(Xvaltest) - validation_size

Xval = Xvaltest.take(validation_size)
Xtest = Xvaltest.skip(validation_size)
classes = Xtrain.class_names
print(classes)
def count_images_per_class(dataset):
    class_counts = {}
    for _, labels in dataset:
        for label in labels.numpy():
            if label not in class_counts:
                class_counts[label] = 1
            else:
                class_counts[label] += 1
    return class_counts

# Brojanje slika za skupove
train_class_counts = count_images_per_class(Xtrain)
print(f"\nUkupan broj slika u trening skupu: {sum(train_class_counts.values())}")
print("Broj slika po klasama u trening skupu:")
print(train_class_counts)

val_class_counts = count_images_per_class(Xval)
print(f"\nUkupan broj slika u validacionom skupu: {sum(val_class_counts.values())}")
print("Broj slika po klasama u validacionom skupu:")
print(val_class_counts)

test_class_counts = count_images_per_class(Xtest)
print(f"\nUkupan broj slika u test skupu: {sum(test_class_counts.values())}")
print("Broj slika po klasama u test skupu:")
print(test_class_counts)

fig, axes = plt.subplots(2, 4, figsize=(15, 6))

for i, class_name in enumerate(class_names):
    class_path = os.path.join(main_path, class_name)
    img_file = os.path.join(class_path, os.listdir(class_path)[0])  # Uzimamo prvi fajl iz klase
    img = Image.open(img_file)
    row = i // 4
    col = i % 4
    axes[row, col].imshow(img)
    axes[row, col].set_title(class_name)
    axes[row, col].axis('off')

plt.show()

import keras_tuner as kt
num_classes = len(classes)

def make_model(hp):
    data_augmentation = Sequential()
    data_augmentation.add(layers.RandomFlip("horizontal_and_vertical", input_shape=(64, 64, 3)))
    data_augmentation.add(layers.GaussianNoise(0.5, input_shape=(64, 64, 3)))
    data_augmentation.add(layers.RandomRotation(0.2))
    data_augmentation.add(layers.RandomZoom(0.1))


    model = Sequential([
        data_augmentation,
        layers.Rescaling(1. / 255, input_shape=(64, 64, 3)),  # normalizacija podataka, tacnije
        layers.Conv2D(32, 3, padding='same', activation='relu', input_shape=(64, 64, 3)),
        layers.MaxPooling2D(),

        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),

        layers.Conv2D(128, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),

        layers.Conv2D(256, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),

        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dense(256, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.summary()
    lr = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    opt = Adam(learning_rate=lr)

    model.compile(optimizer = opt,
                  loss=SparseCategoricalCrossentropy(),
                  metrics='accuracy')

    return model

es = EarlyStopping(monitor='val_loss', mode='min', patience=20, verbose=1, restore_best_weights=True)

tuner = kt.RandomSearch(make_model,
                        objective='val_loss',
                        overwrite = True,
                        max_trials=10)

tuner.search(Xtrain,
            epochs=10,
            validation_data=Xval,
            callbacks = [es],
            verbose=1)

best_model = tuner.get_best_models()
best_hyperparam = tuner.get_best_hyperparameters(num_trials=1)[0]

print('Optimalna konstanta obučavanja: ', best_hyperparam['learning_rate'])

model = tuner.hypermodel.build(best_hyperparam)

N = 8
history = model.fit(Xtrain,
                    epochs=50,
                    validation_data=Xval,
                    callbacks=[es],
                    verbose=1)
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure()
plt.subplot(121)
plt.plot(acc)
plt.plot(val_acc)
plt.title('Accuracy')
plt.subplot(122)
plt.plot(loss)
plt.plot(val_loss)
plt.title('Loss')
plt.show()

labels = np.array([])
pred = np.array([])

flagBad=True
flagGood=True

for img, lab in Xtest:
    labels = np.append(labels, lab)
    p= np.argmax(model.predict(img, verbose=0), axis=1)
    pred = np.append(pred, np.argmax(model.predict(img, verbose=0), axis=1))
    if flagBad or flagGood:
        if flagGood and p[0]==int(lab[0].numpy()):
            plt.imshow(img[0].numpy().astype('uint8'))
            plt.title(f'Good prediction:  Predicted: {geometry[pred[0]]}, Actual: {geometry[int(lab[0].numpy())]}')
            plt.show()
            flagGood=False
        if flagBad and p[0]!= int(lab[0].numpy()):
            plt.imshow(img[0].numpy().astype('uint8'))
            plt.title(f'Bad prediction: Predicted: {geometry[pred[0]]}, Actual: {geometry[int(lab[0].numpy())]}')
            plt.show()
            flagBad = False

print('Tačnost modela je: ' + str(100*accuracy_score(labels, pred)) + '%')

cm = confusion_matrix(labels, pred, normalize='true')
plt.figure(figsize=(16, 12))
cmDisplay = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
cmDisplay.plot(xticks_rotation='vertical')
plt.tight_layout(h_pad=2.0)
plt.show()

labels = np.array([])
pred = np.array([])
for img, lab in Xtrain:
    labels = np.append(labels, lab)
    pred = np.append(pred,np.argmax(model.predict(img, verbose=0), axis=1) )

cm = confusion_matrix(labels, pred, normalize='true')
plt.figure(figsize=(16, 12))
cmDisplay = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
cmDisplay.plot(xticks_rotation='vertical')
plt.tight_layout(h_pad=2.0)
plt.show()


