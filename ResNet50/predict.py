import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os
from tensorflow.keras.models import load_model

def unnormalize_resnet(img):
    img = img + np.array([103.939, 116.779, 123.68])  # tambahkan kembali mean
    img = img[..., ::-1]  # BGR → RGB
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img


# Ambil ulang data gambar dan label dari test_data
images_all = []
labels_all = []
class_names = sorted(os.listdir("../datasets/test"))  # misal test_dir = 'path/to/test'
print(class_names)

# Ambil ulang data gambar dan label dari test_data
images_all = []
labels_all = []
# Parameters
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20

model = load_model('resnet_mangrove_best.h5')

test_dir = "../datasets/test"

test_data = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=False
)
for batch in test_data:
    images, labels = batch
    images_all.extend(images.numpy())
    labels_all.extend(np.argmax(labels.numpy(), axis=1))

# Konversi ke numpy
images_all = np.array(images_all)
labels_all = np.array(labels_all)

# Dapatkan prediksi dari model
preds = model.predict(images_all)
y_pred = np.argmax(preds, axis=1)

# Visualisasi: 10 gambar per kelas
num_classes = len(class_names)
images_per_class = 5

plt.figure(figsize=(20, 12))
idx = 1

for class_idx in range(num_classes):
    class_indices = np.where(labels_all == class_idx)[0][:images_per_class]

    for i in class_indices:
        plt.subplot(num_classes, images_per_class, idx)
        img = images_all[i]

        # UNNORMALIZE untuk ResNet50
        img = unnormalize_resnet(img)

        plt.imshow(img)
        plt.axis('off')
        true_label = class_names[labels_all[i]]
        pred_label = class_names[y_pred[i]]
        color = 'green' if true_label == pred_label else 'red'
        plt.title(f"T: {true_label}\nP:{pred_label}", fontsize=8, color=color)
        idx += 1

plt.tight_layout()
plt.suptitle("Hasil Prediksi Model - 10 Gambar per Kelas", fontsize=16, y=1.02)
plt.show()