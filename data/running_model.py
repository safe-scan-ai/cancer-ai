import tensorflow as tf
import pathlib
import numpy as np
# import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import load_img, img_to_array


model = tf.keras.models.load_model("melanoma.keras")

# load image 
img = load_img("melanoma.jpg", target_size=(180,180,3))
img_array = img_to_array(img)  # Convert to NumPy array
img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

pred = model.predict(img_array)
not_melanoma_probability, melanoma_probability = pred[0]

output = {
    "melanoma": melanoma_probability,
}

print(output)