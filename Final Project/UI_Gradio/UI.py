import gradio as gr
import tensorflow as tf
import numpy as np


model = tf.keras.models.load_model("E:\\Uni\\Term 9 - Fall 1403\\NN & DL\\Project\\best_model.h5")


def predict_iris(image):
    image = image.convert("L")          # Convert to grayscale
    image = image.resize((240, 20))     # Resize to (width=240, height=20)
    image = np.array(image) / 255.0     # Normalize to [0,1]
    image = np.expand_dims(image, axis=(0, -1))  # Reshape to (1, 20, 240, 1)

    output = model(image)
    predicted_class = np.argmax(output.numpy())

    return predicted_class


demo = gr.Interface(
    fn=predict_iris,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="Iris Classifier by Amirreza Eftekhari - 99243014",
    description="Upload an image of an Iris to classify it (0-107 labels):"
)

demo.launch(share=True)
