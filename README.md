# 🎥 Video Captioning and Emotion Detection using CNN + LSTM

An end-to-end **Deep Learning project** that automatically understands the content of a video, generates a natural-language caption describing the video, and analyzes the sentiment/emotion of the generated caption.

The project combines **Computer Vision, CNN, LSTM, Natural Language Processing (NLP), and Sentiment Analysis** to create an automated video understanding system.

---

## 🚀 Project Overview

The system takes a video as input and performs the following tasks:

1. Extracts frames from the video.
2. Extracts visual features from each frame using:

   * Hu Moments
   * Haralick Texture Features
   * Color Histogram
3. Processes video features using a **CNN-based encoder**.
4. Uses an **LSTM-based decoder** to generate a caption.
5. Analyzes the generated caption using **VADER Sentiment Analysis**.
6. Displays the predicted caption and emotion directly on the video.

### Example

**Input:**
🎥 A video of a man playing guitar.

**Generated Caption:**

> man playing guitar outside his house

**Detected Emotion:**

> Positive

---

## 🧠 Architecture

```text
                Input Video
                     │
                     ▼
              Extract Video Frames
                     │
                     ▼
          Visual Feature Extraction
        ┌────────────┼────────────┐
        ▼            ▼            ▼
    Hu Moments   Haralick      Color
                 Texture      Histogram
        └────────────┼────────────┘
                     ▼
              Feature Vector
                     │
                     ▼
              CNN Encoder
                     │
                     ├──────────────┐
                     │              │
                     ▼              ▼
             Caption Tokens      LSTM
                     │              │
                     └──────┬───────┘
                            ▼
                   Generated Caption
                            │
                            ▼
                  VADER Sentiment
                            │
                            ▼
                 Positive / Negative /
                      Neutral
                            │
                            ▼
                  Output Video
```

---

## 📊 Dataset

The project uses the **YouTubeClips video dataset** along with its caption annotations.

The caption dataset contains information including:

* Video ID
* Start time
* End time
* Worker ID
* Source
* Annotation time
* Language
* Description

The project uses the English video descriptions as training captions.

> **Note:** The complete YouTubeClips dataset is not included in this repository because of its large size.

---

## 🔧 Technologies Used

### Programming Language

* Python

### Deep Learning

* TensorFlow
* Keras
* CNN
* LSTM

### Computer Vision

* OpenCV
* Hu Moments
* Haralick Texture Features
* Color Histograms
* Mahotas

### NLP

* Keras Tokenizer
* Sequence Padding
* VADER Sentiment Analyzer

### Data Processing

* NumPy
* Pandas
* Scikit-learn
* MinMaxScaler

### Visualization

* Matplotlib

---

## 📁 Project Structure

```text
Video-Captioning-Emotion-Detection/
│
├── CaptionGeneration.ipynb
├── Captions.csv
├── requirements.txt
├── README.md
│
├── model/
│   ├── cnn_weights.h5
│   ├── cnn_history.pckl
│   ├── features.pckl
│   ├── images.npy
│   └── captions.npy
│
├── testVideos/
│   ├── 1.avi
│   ├── 2.avi
│   ├── 3.avi
│   └── 4.avi
│
└── .gitignore
```

The large `YouTubeClips/` training-video directory is intentionally excluded from the repository.

---

## ⚙️ How It Works

### 1. Video Feature Extraction

Each input video is read using OpenCV. Frames are resized and visual features are extracted.

The project extracts three types of features:

* **Hu Moments** — shape information
* **Haralick Features** — texture information
* **Color Histogram** — color distribution

These features are combined into a single feature vector.

---

### 2. Feature Normalization

The extracted features are normalized using `MinMaxScaler` so that the feature values are brought into a common range.

---

### 3. Caption Tokenization

The English captions are converted into numerical sequences using the Keras `Tokenizer`.

Special sequence tokens are used:

```text
startseq
    ↓
caption words
    ↓
endseq
```

This allows the LSTM decoder to learn when to start and stop generating a caption.

---

### 4. CNN + LSTM Model

The project uses a **CNN + LSTM encoder-decoder architecture**.

The visual feature vector is passed through dense layers, while the caption sequence is processed through an embedding layer and LSTM.

The two representations are combined before predicting the next word.

The model is trained using categorical cross-entropy loss and the Adam optimizer.

---

### 5. Caption Generation

During prediction, the model starts with:

```text
startseq
```

It predicts one word at a time until it generates:

```text
endseq
```

The start and end tokens are then removed to produce the final caption.

---

### 6. Emotion Detection

The generated caption is passed to **VADER Sentiment Analysis**.

The sentiment is classified as:

```text
Positive
Negative
Neutral
```

For example:

```text
Caption: man playing guitar outside his house
Emotion: Positive
```

---

## 🧪 Testing

Test videos can be placed inside:

```text
testVideos/
```

The notebook can then be used to process videos such as:

```python
readTestVideo("testVideos/1.avi")
```

The system displays:

```text
Caption = bird in sink keeps getting under the running water from faucet
Emotion = Neutral
```

The generated caption and emotion are also displayed on the video output.

---

## 📈 Model Performance

The trained model was evaluated on the validation data.

**Validation Accuracy:**

```text
79.71%
```

**Validation Loss:**

```text
2.5014
```

These results are based on the current training configuration and dataset subset used in the project.

---

## 💾 Saved Model Files

The project saves the processed data and trained model information inside the `model/` directory.

```text
model/
├── images.npy
├── captions.npy
├── features.pckl
├── cnn_weights.h5
└── cnn_history.pckl
```

This allows the trained model and processed features to be reused without repeating the complete feature-extraction and training process.

---

## ▶️ Installation

Clone the repository:

```bash
git clone <YOUR-GITHUB-REPOSITORY-URL>
cd Video-Captioning-Emotion-Detection
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

For VADER:

```python
import nltk
nltk.download('vader_lexicon')
```

Then open:

```text
CaptionGeneration.ipynb
```

and run the notebook cells in order.

---

## 🔮 Future Improvements

Some possible improvements include:

* Use pretrained CNN architectures such as **ResNet, EfficientNet, or InceptionV3** for better visual representations.
* Use **attention mechanisms** in the LSTM decoder.
* Replace the traditional CNN-LSTM architecture with a **Transformer-based video captioning model**.
* Improve caption quality using beam search.
* Train on a larger video-caption dataset.
* Add multilingual caption generation.
* Develop a web application for real-time video captioning.
* Improve emotion detection using a dedicated emotion-classification model.

---

## 👨‍💻 Project Highlights

* 🎥 Automated video understanding
* 📝 Natural-language video caption generation
* 🧠 CNN + LSTM deep learning architecture
* 👁️ Computer vision feature extraction
* 💬 NLP-based sentiment analysis
* 📊 Model evaluation and visualization
* ⚡ Automated caption and emotion overlay on video

---

## ⭐ Conclusion

This project demonstrates how **Computer Vision, Deep Learning, and NLP** can be combined to build an automated video understanding system.

Given a video, the system extracts visual information, generates an English caption describing the activity, and determines the sentiment of the generated caption as **Positive, Negative, or Neutral**.

