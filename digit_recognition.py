import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt

# Load and preprocess the MNIST dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
train_labels = to_categorical(train_labels)
test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255
test_labels = to_categorical(test_labels)

# Define a Convolutional Neural Network (CNN) for better accuracy
model = models.Sequential([
    # Input layer is 28x28 pixels
    # L1 Convolutional Layer with 32 filters, Kernel Size is 3x3
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    # L2 MaxPooling Layer (reduces dimensions by half)
    layers.MaxPooling2D((2, 2)),
    # L3 Convolution Layer with 64 filters and 3x3 kernel
    layers.Conv2D(64, (3, 3), activation='relu'),
    # L4 Pooling Layer (reduces dimensions by half)
    layers.MaxPooling2D((2, 2)),
    # L5 Flatten Layer (turns into 1D vector by multiplying out the shape (a,b,c) from previous layer)
    layers.Flatten(),
    # L6 Has 128 Nodes Fully Connected
    layers.Dense(128, activation='relu'), 
    # Output Layer Has 10 Nodes Fully Connected
    layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(train_images, train_labels, epochs=5, batch_size=128, validation_split=0.1)

# Evaluate the model on the test data
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f'Test accuracy: {test_acc:.4f}')

# Function to draw a digit and predict
from tkinter import Tk, Canvas, Button
from PIL import Image

class DrawDigitApp:
    def __init__(self, model):
        self.model = model
        self.root = Tk()
        self.root.title("Draw a Digit")

        self.canvas = Canvas(self.root, width=280, height=280, bg='white')
        self.canvas.grid(row=0, column=0, columnspan=2)
        self.canvas.bind("<B1-Motion>", self.paint)

        self.button_predict = Button(self.root, text="Predict", command=self.predict_digit)
        self.button_predict.grid(row=1, column=0)

        self.button_clear = Button(self.root, text="Clear", command=self.clear_canvas)
        self.button_clear.grid(row=1, column=1)

        self.drawing = np.zeros((280, 280))

    def paint(self, event):
        x, y = event.x, event.y
        self.drawing[y-10:y+10, x-10:x+10] = 1
        self.canvas.create_oval(x-10, y-10, x+10, y+10, fill='black')

    def clear_canvas(self):
        self.canvas.delete('all')
        self.drawing = np.zeros((280, 280))

    def predict_digit(self):
        img = Image.fromarray((self.drawing * 255).astype(np.uint8)).resize((28, 28)).convert('L')
        digit_img = np.array(img).reshape(1, 28, 28, 1).astype('float32') / 255
        predictions = self.model.predict(digit_img)[0]
        predicted_label = np.argmax(predictions)
        print(f"Predicted Digit: {predicted_label}")
        for i, prob in enumerate(predictions):
            print(f"{i}: {prob:.4f}")

    def run(self):
        self.root.mainloop()

# Launch the drawing app
draw_digit_app = DrawDigitApp(model)
draw_digit_app.run()
