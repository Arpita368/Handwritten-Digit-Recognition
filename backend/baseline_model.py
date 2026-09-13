from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten
from preprocessing import x_train, y_train_cat

model = Sequential([
    Flatten(input_shape=(28,28)),
    Dense(128, activation="relu"),
    Dense(64, activation="relu"),
    Dense(10, activation="softmax")
])

model.compile(
    optimizer = "adam",
    loss = "categorical_crossentropy",
    metrics = ["accuracy"]
)

history = model.fit(x_train, y_train_cat,
                    validation_split=0.1,
                    epochs=10,
                    batch_size=32)