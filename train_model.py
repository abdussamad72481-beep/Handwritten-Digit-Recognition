"""Train a neural network to recognize handwritten digits (MNIST)."""
from tensorflow import keras
from tensorflow.keras import layers

print("Loading MNIST dataset. Internet is needed on the first run...")
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Normalize pixel values to the range 0 to 1.
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

model = keras.Sequential([
    layers.Input(shape=(28, 28)),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.2),
    layers.Dense(10, activation="softmax"),
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

print("Training model...")
model.fit(x_train, y_train, epochs=5, batch_size=128, validation_split=0.1)

loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"Test accuracy: {accuracy * 100:.2f}%")

model.save("digit_model.keras")
print("Saved model as digit_model.keras")
