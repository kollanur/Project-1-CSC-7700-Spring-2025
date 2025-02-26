from mlp import *

import torchvision.datasets as datasets
import matplotlib
import numpy as np  # linear algebra
import struct
from array import array
from os.path import join
import random
import matplotlib.pyplot as plt


# Download MNIST dataset in 'data/' folder
datasets.MNIST(root='./data', train=True, download=True)
datasets.MNIST(root='./data', train=False, download=True)

print("MNIST dataset downloaded in 'data/' folder")


class MnistDataloader:
    def __init__(self, training_images_filepath, training_labels_filepath,
                 test_images_filepath, test_labels_filepath):
        self.training_images_filepath = training_images_filepath
        self.training_labels_filepath = training_labels_filepath
        self.test_images_filepath = test_images_filepath
        self.test_labels_filepath = test_labels_filepath
    
    def read_images_labels(self, images_filepath, labels_filepath):
        # Read labels
        with open(labels_filepath, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError(f'Magic number mismatch for labels, expected 2049, got {magic}')
            labels = np.array(array("B", file.read()))  # Convert directly to NumPy array

        # Read images
        with open(images_filepath, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError(f'Magic number mismatch for images, expected 2051, got {magic}')
            image_data = np.frombuffer(file.read(), dtype=np.uint8)  # Read as NumPy array
        
        # Reshape images to (size, 28, 28)
        images = image_data.reshape(size, rows, cols)
        
        return images, labels
            
    def load_data(self):
        x_train, y_train = self.read_images_labels(self.training_images_filepath, self.training_labels_filepath)
        x_test, y_test = self.read_images_labels(self.test_images_filepath, self.test_labels_filepath)
        return (x_train, y_train), (x_test, y_test)

input_path = './data/MNIST/raw/'
training_images_filepath = join(input_path, 'train-images-idx3-ubyte')
training_labels_filepath = join(input_path, 'train-labels-idx1-ubyte')
test_images_filepath = join(input_path, 't10k-images-idx3-ubyte')
test_labels_filepath = join(input_path, 't10k-labels-idx1-ubyte')

mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)
(x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()

# Normalize images to range [0, 1]
x_train = x_train.reshape(-1, 28 * 28) / 255.0
x_test = x_test.reshape(-1, 28 * 28) / 255.0

# One-hot encode labels
def one_hot_encode(labels, num_classes=10):
    return np.eye(num_classes)[labels]

y_train = one_hot_encode(y_train)
y_test = one_hot_encode(y_test)


def create_deep_mlp(input_size=784, hidden_units=64, num_layers=8, output_size=10, dropout_rate=0.2):
    layers = []
    prev_units = input_size  # Input layer size
    
    for _ in range(num_layers - 1):
        layers.append(Layer(fan_in=prev_units, fan_out=hidden_units, activation_function=Relu(), dropout_rate=0.01))
        prev_units = hidden_units

    # Output layer with softmax activation
    layers.append(Layer(fan_in=prev_units, fan_out=output_size, activation_function=Softmax()))

    return MultilayerPerceptron(layers)


mlp = create_deep_mlp(hidden_units=64, num_layers=8)

# Define loss function
loss_func = CrossEntropy()

# Train the model
training_losses, validation_losses = mlp.train(
    train_x=x_train, train_y=y_train,
    val_x=x_test, val_y=y_test,
    loss_func=loss_func,
    learning_rate=0.0001, batch_size=64, epochs=25,
    rmsprop=True,  # Enable RMSProp
    beta=0.9,      # RMSProp decay factor
    epsilon=1e-8   # Small constant for numerical stability
)



from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Evaluate Model Performance
y_pred = mlp.forward(x_test, training = False)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

# Compute Accuracy
accuracy = accuracy_score(y_true_classes, y_pred_classes)
print(f"Test Accuracy: {accuracy:.4f}")


# Classification Report
print("Classification Report:")
print(classification_report(y_true_classes, y_pred_classes))

import matplotlib.pyplot as plt

plt.figure(figsize=(12, 5))

# Plot loss
plt.subplot(1, 2, 1)
plt.plot(training_losses, label="Training Loss", color="blue")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss Over Time")
plt.legend()

# Plot validation loss
plt.subplot(1, 2, 2)
plt.plot(validation_losses, label="Validation Loss", color="red")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Validation Loss Over Time")
plt.legend()

plt.show()




# Forward pass to get predictions
y_pred = mlp.forward(x_test, training=False)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

# Dictionary to store one sample per class (0-9)
selected_samples = {}

for i in range(len(y_test)):
    label = y_true_classes[i]
    if label not in selected_samples:
        selected_samples[label] = (x_test[i], y_pred_classes[i])
    if len(selected_samples) == 10:
        break

fig, axes = plt.subplots(2, 5, figsize=(8, 5))

for i, (label, (image, pred)) in enumerate(selected_samples.items()):
    row, col = divmod(i, 5)
    axes[row, col].imshow(image.reshape(28, 28), cmap="gray")
    axes[row, col].set_title(f"True: {label}\nPred: {pred}", color="green" if label == pred else "red")
    axes[row, col].axis("off")

plt.suptitle("MNIST Classification - One Sample per Class", fontsize=14)
plt.tight_layout()
plt.show()

