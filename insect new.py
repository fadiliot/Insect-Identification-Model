import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

# Load the model
model = load_model("C:/Users/Fadil Anwar/Downloads/keras_model.h5", compile=False)

# Load the labels
class_names = open("C:/Users/Fadil Anwar/Downloads/labels.txt", "r").readlines()

# Create the array of the right shape to feed into the keras model
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

def classify_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        image = Image.open(file_path).convert("RGB")
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.LANCZOS)
        image_array = np.asarray(image)
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
        data[0] = normalized_image_array
        prediction = model.predict(data)
        index = np.argmax(prediction)
        class_name = class_names[index].strip()
        confidence_score = prediction[0][index]
        result_label.config(text=f"Class: {class_name}\nConfidence Score: {confidence_score:.2f}")
        print(f"Class: {class_name}, Confidence Score: {confidence_score:.2f}")


# Create the main window
root = tk.Tk()
root.title("Image Classifier")

# Create and configure a button to select an image
select_image_button = tk.Button(root, text="Select Image", command=classify_image)
select_image_button.pack(pady=10)

# Create a label to display the result
result_label = tk.Label(root, text="", wraplength=300)
result_label.pack()

root.mainloop()
