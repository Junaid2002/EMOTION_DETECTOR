import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" 
import tensorflow as tf
import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder

try:
    model = tf.keras.models.load_model("models/emotion_detector_model.h5")
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()
try:
    train_df = pd.read_csv("dataset/train.csv")
    texts = train_df["text"].tolist()
    labels = train_df["label"].tolist()
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit()

tokenizer_path = "models/tokenizer.pkl"
try:
    with open(tokenizer_path, "rb") as handle:
        tokenizer = pickle.load(handle)
    print("Tokenizer loaded from file.")
except FileNotFoundError:
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=5000, oov_token="<OOV>")
    tokenizer.fit_on_texts(texts)
    with open(tokenizer_path, "wb") as handle:
        pickle.dump(tokenizer, handle)
    print("Tokenizer trained and saved.")

label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

label_mapping = {i: label for i, label in enumerate(label_encoder.classes_)}

def predict_emotion(text):
    try:
        seq = tokenizer.texts_to_sequences([text])
        padded_seq = pad_sequences(seq, maxlen=50, padding="post")
        prediction = model.predict(padded_seq)
        emotion = label_mapping[np.argmax(prediction)]
        return emotion
    except Exception as e:
        return f"Prediction error: {e}"

print(predict_emotion("I will not be able to work today!"))
print(predict_emotion("Its raining"))
