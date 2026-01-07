from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras import layers
import tensorflow as tf

# ------------------ config ------------------

# dataset used for training is over 3Gbs
# you can download from kaggle like that
# kagglehub.dataset_download("waheed9002/hiking-wildlife")
path_to_dir = 'datasets/dataset'

img_size = (224, 224)
btch_size = 50
seed = 62

# ------------------ datasets ------------------

train_ds = image_dataset_from_directory(
    path_to_dir,
    label_mode='categorical',
    image_size=img_size,
    batch_size=btch_size,
    validation_split=0.2,
    shuffle=True,
    seed=seed,
    subset='training',
)

val_ds = image_dataset_from_directory(
    path_to_dir,
    label_mode='categorical',
    image_size=img_size,
    batch_size=btch_size,
    validation_split=0.2,
    shuffle=False,
    seed=seed,
    subset='validation',
)

class_names = train_ds.class_names
print(class_names)

# ------------------ preprocessing ------------------

def resnet_preprocess(images, labels):
    images = tf.cast(images, tf.float32)
    images = preprocess_input(images)
    return images, labels

train_ds = (
    train_ds
    .map(resnet_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    .shuffle(1000, seed=seed)
    .prefetch(tf.data.AUTOTUNE)
)

val_ds = (
    val_ds
    .map(resnet_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    .prefetch(tf.data.AUTOTUNE)
)

# ------------------ augmentation ------------------

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

# ------------------ model ------------------

base_model = tf.keras.applications.ResNet50(
    include_top=False,
    weights="imagenet",
    input_shape=(224, 224, 3)
)

base_model.trainable = False

num_classes = train_ds.element_spec[1].shape[-1]

inputs = tf.keras.Input(shape=(224, 224, 3))
x = data_augmentation(inputs)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.BatchNormalization()(x)
x = layers.Dense(256, activation="relu")(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(num_classes, activation="softmax")(x)

model = tf.keras.Model(inputs, outputs)

# ------------------ training setup ------------------

optimizer = tf.keras.optimizers.AdamW(
    learning_rate=1e-3,
    weight_decay=1e-4
)

model.compile(
    optimizer=optimizer,
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ModelCheckpoint(
        "best_resnet_model.keras",
        monitor="val_loss",
        save_best_only=True
    )
]

# ------------------ train ------------------

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=30,
    callbacks=callbacks
)

# ------------------ save ------------------

model.save("models/best_resnet_model.keras")

# --------- history for evaluation----------
history = {'accuracy': [0.6916093826293945,
  0.8079367280006409,
  0.8450266122817993,
  0.8699260950088501,
  0.877058744430542,
  0.8849695324897766,
  0.8936583995819092,
  0.9027363657951355,
  0.9052003622055054,
  0.9088315367698669],
 'loss': [1.1031659841537476,
  0.6392372846603394,
  0.5213324427604675,
  0.4155902862548828,
  0.39497196674346924,
  0.3688940405845642,
  0.33191800117492676,
  0.3151596486568451,
  0.28891849517822266,
  0.2881230413913727],
 'val_accuracy': [0.9444732666015625,
  0.9356512427330017,
  0.950700581073761,
  0.9475868940353394,
  0.9621172547340393,
  0.9273481965065002,
  0.9481058716773987,
  0.9636741280555725,
  0.9615983366966248,
  0.936170220375061],
 'val_loss': [0.19035956263542175,
  0.2174569070339203,
  0.17872263491153717,
  0.1720951795578003,
  0.12275756895542145,
  0.2546690106391907,
  0.18067122995853424,
  0.14441268146038055,
  0.14522726833820343,
  0.2119835615158081]}
def get_history():
    return history