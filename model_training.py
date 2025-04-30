import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay, roc_curve, auc
from sklearn.preprocessing import label_binarize
from itertools import cycle
import os
import warnings
warnings.filterwarnings('ignore')

img_classes = ['Enfeksiyonel', 'Ekzama', 'Akne', 'Pigment', 'Benign', 'Malign']
train_dir = "C:/Users/Sai sri gowri/Downloads/train"
test_dir = "C:/Users/Sai sri gowri/Downloads/test"
val_dir = "C:/Users/Sai sri gowri/Downloads/val"

img_width, img_height = 400, 400
batch_size = 16

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=True,
    seed=42
)
val_data = val_datagen.flow_from_directory(
    val_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=True,
    seed=42
)
test_data = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=False
)

input_layer = Input(shape=(img_height, img_width, 3))
vgg19 = VGG19(input_shape=(img_height, img_width, 3), weights='imagenet', include_top=False)
vgg19.trainable = False

set_trainable = False
for layer in vgg19.layers:
    if layer.name == 'block5_conv1':
        set_trainable = True
    if set_trainable:
        layer.trainable = True

x = vgg19(input_layer)
x = GlobalAveragePooling2D()(x)
x = Dropout(0.4)(x)
x = Dense(100, activation='relu')(x)
output_layer = Dense(6, activation='softmax')(x)

model = Model(inputs=input_layer, outputs=output_layer)

adam = Adam(learning_rate=0.0001)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
model.summary()

chk_path = './best_model.keras'
checkpoint = ModelCheckpoint(filepath=chk_path, monitor='val_accuracy', mode='max', save_best_only=True, verbose=1)
early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, mode='max', verbose=1)

history = model.fit(
    train_data,
    epochs=100,
    validation_data=val_data,
    callbacks=[early_stopping, checkpoint]
)

best_model = load_model(chk_path)
loss, acc = best_model.evaluate(test_data)
print(f"\nTest Accuracy = {acc}\nTest Loss = {loss}")

plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'], label='Train Accuracy', color='blue')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='red')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='best')
plt.grid()
plt.savefig('accuracy_plot.png')
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Train Loss', color='blue')
plt.plot(history.history['val_loss'], label='Validation Loss', color='red')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(loc='best')
plt.grid()
plt.savefig('loss_plot.png')
plt.show()

test_data.reset()
y_pred = np.argmax(best_model.predict(test_data), axis=1)
y_true = test_data.classes
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=img_classes)
disp.plot(cmap='viridis', xticks_rotation=45)
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
plt.show()

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=img_classes))

class_counts = train_data.classes
sns.histplot(class_counts, bins=len(img_classes), kde=False)
plt.title('Training Data Class Distribution')
plt.xlabel('Class')
plt.ylabel('Count')
plt.xticks(ticks=np.arange(len(img_classes)), labels=img_classes, rotation=45)
plt.grid()
plt.savefig('class_distribution.png')
plt.show()

augmented_images, _ = next(train_data)
plt.figure(figsize=(12, 12))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(augmented_images[i])
    plt.axis('off')
    plt.title('Augmented Image')
plt.savefig('augmented_images.png')
plt.show()

y_true_binarized = label_binarize(y_true, classes=np.arange(len(img_classes)))
y_pred_proba = best_model.predict(test_data)

fpr = {}
tpr = {}
roc_auc = {}

for i in range(len(img_classes)):
    fpr[i], tpr[i], _ = roc_curve(y_true_binarized[:, i], y_pred_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

fpr["micro"], tpr["micro"], _ = roc_curve(y_true_binarized.ravel(), y_pred_proba.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

plt.figure(figsize=(12, 8))
colors = cycle(['blue', 'green', 'red', 'cyan', 'magenta', 'orange'])

for i, color in zip(range(len(img_classes)), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label=f"Class {img_classes[i]} (AUC = {roc_auc[i]:.2f})")

plt.plot(fpr["micro"], tpr["micro"], color='deeppink', linestyle=':', linewidth=4,
         label=f"Micro-average (AUC = {roc_auc['micro']:.2f})")
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Each Class')
plt.legend(loc='lower right')
plt.grid()
plt.savefig('roc_curves.png')
plt.show()
