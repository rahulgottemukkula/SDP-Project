# CaptionGeneration.py
# Raw code reconstructed from the project report's Appendix A: SAMPLECODE.
# NOTE: This is a faithful extraction of the original notebook logic, adapted into a single script file.

import os
import pickle
import numpy as np
from tqdm import tqdm
import cv2

# Keras / TensorFlow imports
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.utils import to_categorical
from keras.layers import Input, Dense, LSTM, Embedding, Dropout
from keras.layers import add
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam

import pandas as pd
import matplotlib.pyplot as plt

# NLP + features
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import mahotas
from sklearn.preprocessing import MinMaxScaler

# ------------------------------
# Feature extraction functions
# ------------------------------

def fd_hu_moments(image):
    # feature-descriptor-1: Hu Moments
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feature = cv2.HuMoments(cv2.moments(image)).flatten()
    return feature

def fd_haralick(image):
    # feature-descriptor-2: Haralick Texture
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    haralick = mahotas.features.haralick(gray).mean(axis=0)
    return haralick

def fd_histogram(image, mask=None):
    # feature-descriptor-3: Color Histogram (HSV)
    bins = 8
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([image], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

# ------------------------------
# Video -> feature vector
# ------------------------------

def video_to_frames(video_path, max_frames=50):
    count = 0
    image_list = []
    cap = cv2.VideoCapture(video_path)
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if (not ret) or (count > max_frames):
                break
            img = cv2.resize(frame, (224, 224))
            fv_hu = fd_hu_moments(img.copy())
            fv_har = fd_haralick(img.copy())
            fv_hist = fd_histogram(img.copy())
            feat = np.hstack([fv_hist, fv_har, fv_hu])
            image_list.append(feat)
            count += 1
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return np.asarray(image_list).ravel()

# ------------------------------
# Caption helpers
# ------------------------------

def ensure_vader():
    try:
        _ = SentimentIntensityAnalyzer()
    except:
        nltk.download('vader_lexicon')
    return SentimentIntensityAnalyzer()

analyzer = ensure_vader()

def getCaption_for_path_exact(dataset_df, video_path):
    """
    ORIGINAL semantics from the notebook:
    Construct path = dataset[i,0] + "_" + str(dataset[i,1]) + "_" + str(dataset[i,2]) + ".avi"
    Match with given video_path and return cleaned caption from column index 7.
    """
    df = dataset_df.values
    for i in range(len(df)):
        path = f"{df[i,0]}_{df[i,1]}_{df[i,2]}.avi"
        if video_path == path:
            caption = str(df[i,7]).strip().lower()
            caption = caption.replace(".", "")
            # Mimic the notebook regex replacements with simple filters
            # Remove non-alpha and condense spaces
            filtered = []
            for w in caption.split():
                w2 = ''.join(ch for ch in w if ch.isalpha())
                if len(w2) > 1:
                    filtered.append(w2)
            caption_clean = 'startseq ' + " ".join(filtered) + ' endseq'
            return caption_clean
    return ""

# ------------------------------
# Data prep
# ------------------------------

def prepare_data(image_features, caption_features, tokenizer, max_length, vocab_size):
    x1, x2, y = [], [], []
    for m in range(len(image_features)):
        caption = caption_features[m]
        seq = tokenizer.texts_to_sequences([caption.split()])[0]
        length = len(seq)
        for i in range(1, length):
            x2_seq, y_seq = seq[:i] , seq[i]
            x2_seq = pad_sequences([x2_seq], maxlen=max_length, padding='post')[0]
            y_seq = to_categorical([y_seq], num_classes=vocab_size)[0]
            x1.append(image_features[m])
            x2.append(x2_seq)
            y.append(y_seq)
    return np.array(x1), np.array(x2), np.array(y)

def word_for_id(integer, tokenizr):
    for word, index in tokenizr.word_index.items():
        if index == integer:
            return word
    return None

def predictCaption(model, tokenizer, features, max_length):
    in_text = 'startseq'
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length, padding='post')
        yhat = model.predict([features, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = word_for_id(yhat, tokenizer)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'endseq':
            break
    in_text = in_text.replace('startseq', '').replace('endseq', '').strip()
    return in_text

def getEmotion(text):
    scores = analyzer.polarity_scores(text)
    if scores['compound'] >= 0.05:
        return "Positive"
    elif scores['compound'] <= -0.05:
        return "Negative"
    else:
        return "Neutral"

def playVideo(filename, caption, emotion):
    cap = cv2.VideoCapture(filename)
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (800, 400))
            cv2.putText(frame, caption, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.putText(frame, "Emotion " + emotion, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.imshow('Output Video', frame)
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

# ------------------------------
# Model build
# ------------------------------

def build_model(vocab_size, max_length):
    # encoder (image features)
    inputs1 = Input(shape=(27132,))  # as per original
    fe1 = Dropout(0.4)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)

    # sequence feature layers
    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = Dropout(0.4)(se1)
    se3 = LSTM(256)(se2)

    # decoder
    decoder1 = add([fe2, se3])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='sigmoid')(decoder2)

    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    optimizer = Adam(0.001)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

# ------------------------------
# Training + Evaluation (optional main)
# ------------------------------

def main():
    # Expect a Captions.csv in current working directory
    if not os.path.exists("Captions.csv"):
        print("Captions.csv not found. Place the dataset CSV in the working directory.")
        return

    dataset_df = pd.read_csv("Captions.csv")

    # Either load pre-saved numpy arrays or build features from raw videos per original logic
    images_path = "model/images.npy"
    captions_path = "model/captions.npy"
    features_pckl = "model/features.pckl"
    os.makedirs("model", exist_ok=True)

    if os.path.exists(images_path) and os.path.exists(captions_path):
        images = np.load(images_path, allow_pickle=True)
        captions = np.load(captions_path, allow_pickle=True)
    else:
        features = {}
        # ORIGINAL code iterates rows and constructs video path:
        df = dataset_df.values
        index = 0
        for i in range(len(df)):
            video_path = f"{df[i,0]}_{df[i,1]}_{df[i,2]}.avi"
            if video_path not in features:
                vid_full_path = os.path.join('YouTubeClips', video_path)
                if not os.path.exists(vid_full_path):
                    # skip missing
                    continue
                frames = video_to_frames(vid_full_path)
                # ORIGINAL literal shape check:
                if frames.shape[0] == 27132:
                    features[video_path] = frames
                    index += 1
                if len(features) > 300:
                    break
        with open(features_pckl, 'wb') as f:
            pickle.dump(features, f)

        images, captions = [], []
        for key, values in features.items():
            caption = getCaption_for_path_exact(dataset_df, key)
            images.append(values)
            captions.append(caption)
        images = np.asarray(images)
        captions = np.asarray(captions)
        np.save(images_path, images)
        np.save(captions_path, captions)
        print("Video Features & Caption Extraction Completed")
        print("Total Number of Loaded Videos = " + str(images.shape[0]))
        print("Total Frames Processed = " + str(images.shape[0] * 50))

    # Normalize features
    scaler = MinMaxScaler((0, 1))
    images = scaler.fit_transform(images)

    # Tokenizer + lengths
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(captions)
    vocab_size = len(tokenizer.word_index) + 1
    max_length = max(len(c.split()) for c in captions)

    # Prepare data (train / validation split as per original ranges)
    train_x1, train_x2, train_y = prepare_data(images[0:250], captions[0:250], tokenizer, max_length, vocab_size)
    validate_x1, validate_x2, validate_y = prepare_data(images[250:len(images)], captions[250:len(images)], tokenizer, max_length, vocab_size)

    # Build / train / load weights
    model = build_model(vocab_size, max_length)
    weights_path = "model/cnn_weights.h5"
    if not os.path.exists(weights_path):
        model_check_point = ModelCheckpoint(filepath=weights_path, verbose=1, save_best_only=True)
        hist = model.fit([train_x1, train_x2], train_y, verbose=1, epochs=20,
                         callbacks=[model_check_point],
                         validation_data=([validate_x1, validate_x2], validate_y))
        with open('model/cnn_history.pckl', 'wb') as f:
            pickle.dump(hist.history, f)
    else:
        model.load_weights(weights_path)

    # Evaluate
    score = model.evaluate([validate_x1, validate_x2], validate_y, verbose=1)
    print('Validation loss:', score[0])
    print('Validation accuracy:', score[1])

    # Plot training curves if history exists
    hist_path = 'model/cnn_history.pckl'
    if os.path.exists(hist_path):
        with open(hist_path, 'rb') as f:
            train_values = pickle.load(f)
        plt.figure(figsize=(6, 3))
        plt.xlabel('EPOCH')
        plt.ylabel('Accuracy')
        plt.plot(train_values.get('accuracy', []))
        plt.plot(train_values.get('val_accuracy', []))
        plt.legend(['Training Accuracy', 'Validation Accuracy'], loc='upper left')
        plt.title('CNN + LSTM Training Accuracy Graph')
        plt.show()

        plt.figure(figsize=(6, 3))
        plt.xlabel('EPOCH')
        plt.ylabel('Loss')
        plt.plot(train_values.get('loss', []))
        plt.plot(train_values.get('val_loss', []))
        plt.legend(['Training Loss', 'Validation Loss'], loc='upper left')
        plt.title('CNN + LSTM Training Loss Graph')
        plt.show()

    # Simple test helper
    def readTestVideo(video_test):
        photo = [video_to_frames(video_test)]
        photo = np.asarray(photo)
        photo = scaler.transform(photo)
        caption = predictCaption(model, tokenizer, photo, max_length)
        emotion = getEmotion(caption)
        print("Caption = ", caption)
        print("Emotion = ", emotion)
        playVideo(video_test, caption, emotion)

    # Example (paths must exist)
    # readTestVideo("testVideos/1.avi")

if __name__ == "__main__":
    main()
