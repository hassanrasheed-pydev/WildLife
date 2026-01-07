import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from train import get_history
from tensorflow.keras.utils import image_dataset_from_directory
import numpy as np

# dataset used for training is over 3Gbs
# you can download from kaggle like that
# kagglehub.dataset_download("waheed9002/hiking-wildlife")
path_to_dir = 'datasets/dataset'

img_size = (224, 224)
btch_size = 50

# val ds for evaluation
val_ds = image_dataset_from_directory(
    path_to_dir,
    label_mode='categorical',
    image_size=img_size,
    batch_size=btch_size,
   
)

# you will get the last trained models history
history = get_history()
acc = history['accuracy']
val_acc = history['val_accuracy']
loss = history['loss']
val_loss = history['val_loss']
epochs = range(1, len(acc) + 1)

y_true = []
y_pred = []

class_names = [
    "Horse",
    "bald_eagle",
    "black_bear",
    "bobcat",
    "cheetah",
    "cougar",
    "deer",
    "elk",
    "gray_fox",
    "hyena",
    "lion",
    "raccoon",
    "red_fox",
    "rhino",
    "tiger",
    "wolf",
    "zebra",
]

model = load_model("models/best_resnet_model.keras")

for images, labels in val_ds:
    preds = model.predict(images, verbose=0)
    y_true.extend(np.argmax(labels.numpy(), axis=1))
    y_pred.extend(np.argmax(preds, axis=1))

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(17, 8))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix with Class Names")
plt.tight_layout()
plt.show()

# We cant use history.history here so
# thats why we used history of last trained model
# you can save the history of each model in json or pickle file 
# if you wanted.

plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs, acc, label='Train Accuracy')
plt.plot(epochs, val_acc, label='Val Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training vs Validation Accuracy')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(epochs, loss, label='Train Loss')
plt.plot(epochs, val_loss, label='Val Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training vs Validation Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()