import tensorflow as tf
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet import preprocess_input


def predict_image(
    model,
    image_path,
    class_names,
    plot=False,
    target_size=(224, 224)
):
    # Load image
    img_orig = tf.keras.utils.load_img(image_path)
    img_array = tf.keras.utils.img_to_array(img_orig)

    # Resize and ResNet preprocessing
    preprocess_layer = tf.keras.Sequential([
        tf.keras.layers.Resizing(target_size[0], target_size[1]),
        tf.keras.layers.Lambda(preprocess_input)
    ])

    img_preprocessed = preprocess_layer(
        tf.expand_dims(img_array, axis=0)
    )

    # Model inference
    probs = model.predict(img_preprocessed, verbose=0)[0]
    pred_idx = np.argmax(probs)
    pred_class = class_names[pred_idx]

    # visualization
    if plot:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        axes[0].imshow(tf.keras.utils.array_to_img(img_array))
        axes[0].axis("off")
        axes[0].set_title("Input Image")

        sns.barplot(
            x=probs,
            y=class_names,
            ax=axes[1],
            palette=[
                "orange" if i == pred_idx else "skyblue"
                for i in range(len(class_names))
            ]
        )
        axes[1].set_xlim(0, 1)
        axes[1].set_xlabel("Probability")
        axes[1].set_title(f"Prediction: {pred_class}")

        plt.tight_layout()
        plt.show()

    return pred_class, probs


# ------------------ example ------------------
# print(tf.__version__)
# image_path = r"C:\Users\Hassan\Downloads\R (2).jpg"
# model = load_model("models/best_resnet_model (1).keras")
# class_names = [
#     "Horse",
#     "bald_eagle",
#     "black_bear",
#     "bobcat",
#     "cheetah",
#     "cougar",
#     "deer",
#     "elk",
#     "gray_fox",
#     "hyena",
#     "lion",
#     "raccoon",
#     "red_fox",
#     "rhino",
#     "tiger",
#     "wolf",
#     "zebra",
# ]

# print(tf.__version__)
# pred_class, probs = predict_image(
#     model,
#     image_path,
#     class_names=class_names,
#     plot=True
# )

# print("Predicted Class:", pred_class)
# print("Probabilities:", probs)