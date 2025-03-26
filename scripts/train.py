import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
train_df = pd.read_csv("dataset/train.csv")
val_df = pd.read_csv("dataset/val.csv")

tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
tokenizer.fit_on_texts(train_df["text"])

X_train = pad_sequences(tokenizer.texts_to_sequences(train_df["text"]), maxlen=50, padding="post")
X_val = pad_sequences(tokenizer.texts_to_sequences(val_df["text"]), maxlen=50, padding="post")

label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(train_df["label"])
y_val = label_encoder.transform(val_df["label"])

y_train = tf.keras.utils.to_categorical(y_train)
y_val = tf.keras.utils.to_categorical(y_val)

model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(input_dim=5000, output_dim=128, input_length=50),
    tf.keras.layers.LSTM(64, return_sequences=True),
    tf.keras.layers.LSTM(32),
    tf.keras.layers.Dense(16, activation="relu"),
    tf.keras.layers.Dense(len(label_encoder.classes_), activation="softmax")
])

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32)
model.save("models/emotion_detector_model.h5")
print("✅ Model Trained and Saved!")
