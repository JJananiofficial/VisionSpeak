#  VisionSpeak

**VisionSpeak** is a real-time AI-based sign gesture recognition system that uses computer vision and deep learning to recognize hand gestures through a webcam and convert the recognized signs into spoken output.

The project combines **MediaPipe Hand Landmarks**, **TensorFlow/Keras**, **OpenCV**, and **pyttsx3** to create an accessible human-computer interaction system.

---

##  Features

-  Real-time hand gesture recognition
-  Webcam-based interaction
-  Deep-learning gesture classification
-  MediaPipe hand landmark detection
-  Support for up to two hands
-  Confidence-based prediction
-  Automatic sentence construction
-  Text-to-speech output
-  Real-time OpenCV interface
-  Pre-trained model included for direct testing

---

##  How It Works

VisionSpeak converts hand gestures into numerical landmark features and uses a trained neural network to classify them.

```text
Webcam / Gesture Images
          ↓
   MediaPipe Hands
          ↓
  Hand Landmark Points
          ↓
  84 Normalized Features
          ↓
 TensorFlow Neural Network
          ↓
   Recognized Gesture
          ↓
   Sentence Formation
          ↓
    Text-to-Speech
```

For each hand, MediaPipe provides 21 landmarks. Each landmark contributes an X and Y coordinate:

```text
21 landmarks × 2 coordinates = 42 features
```

The application supports up to two hands:

```text
42 × 2 = 84 features
```

The coordinates are normalized relative to the wrist to reduce the effect of the hand's position in the camera frame.

---

##  Project Structure

```text
VisionSpeak/
│
├── 1_collect_data.py
├── 2_train_model.py
├── 3_realtime_app.py
│
├── gesture_dataset.csv
├── label_encoder.pkl
├── vision_speak_model.h5
│
├── requirements.txt
├── .gitignore
└── README.md
```

### File Description

| File | Purpose |
|---|---|
| `1_collect_data.py` | Collects hand-landmark samples from a webcam |
| `2_train_model.py` | Trains the neural-network classifier |
| `3_realtime_app.py` | Runs real-time gesture recognition |
| `gesture_dataset.csv` | Processed landmark dataset |
| `label_encoder.pkl` | Stores the gesture-label encoder |
| `vision_speak_model.h5` | Trained TensorFlow/Keras model |
| `requirements.txt` | Python dependencies |
| `.gitignore` | Files excluded from Git |

---

#  Installation

## 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/VisionSpeak.git
cd VisionSpeak
```

Replace `YOUR_USERNAME` with your GitHub username.

## 2. Create a Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

#  Running VisionSpeak

The repository already contains the trained model and label encoder, so you can run the application without retraining.

Start the real-time application:

```bash
python 3_realtime_app.py
```

Make sure your webcam is connected and accessible.

The application will open a window and display the detected gesture and the sentence being constructed.

---

#  Controls

| Key | Action |
|---|---|
| `SPACE` | Speak the current sentence |
| `ENTER` | Speak the current sentence |
| `C` | Clear the sentence |
| `Q` | Quit the application |

> Click on the OpenCV video window before using the keyboard controls.

---

#  Collecting Your Own Dataset

You can collect additional gesture samples using:

```bash
python 1_collect_data.py
```

Inside the file, change:

```python
label = "Sorry"
```

to the gesture you want to collect.

For example:

```python
label = "Hello"
```

The script collects hand landmarks from the webcam and stores them in:

```text
gesture_dataset.csv
```

By default, it collects:

```text
200 samples per gesture
```

The collected data contains 84 normalized coordinate features for up to two hands.

---

#  Training the Model

After collecting or updating the dataset, run:

```bash
python 2_train_model.py
```

The training pipeline:

```text
gesture_dataset.csv
        ↓
Feature / Label Separation
        ↓
Label Encoding
        ↓
80% Training / 20% Testing
        ↓
Neural Network Training
        ↓
vision_speak_model.h5
        +
label_encoder.pkl
```

### Neural Network

The current model uses:

```text
Input: 84 features
        ↓
Dense: 128 neurons + ReLU
        ↓
Dropout: 20%
        ↓
Dense: 64 neurons + ReLU
        ↓
Dropout: 20%
        ↓
Output: Number of gesture classes + Softmax
```

The model is trained using:

- Adam optimizer
- Sparse categorical cross-entropy
- Accuracy metric
- 50 epochs
- 80/20 train-test split

---

#  Text-to-Speech

When a gesture is recognized consistently above the confidence threshold, it is added to the sentence.

The complete sentence can then be spoken using the `SPACE` or `ENTER` key.

VisionSpeak uses **pyttsx3** for local text-to-speech.

---

#  Technologies Used

- **Python**
- **OpenCV** – webcam and real-time image processing
- **MediaPipe** – hand landmark detection
- **TensorFlow / Keras** – neural-network training and inference
- **NumPy** – numerical processing
- **Pandas** – dataset processing
- **Scikit-learn** – label encoding and train/test splitting
- **pyttsx3** – text-to-speech
- **Pickle** – saving/loading the label encoder

---

#  Limitations

- Recognition is limited to gestures represented in the training dataset.
- Accuracy depends on the quality and diversity of training samples.
- Lighting, background, camera angle, hand orientation, and occlusion can affect recognition.
- The current implementation focuses on static hand-landmark patterns rather than full temporal/dynamic sign-language sequences.
- A larger and more diverse dataset can improve generalization.
- Text-to-speech behavior can vary depending on the operating system and installed voices.

---

#  Future Enhancements

- [ ] Expand the gesture vocabulary
- [ ] Add a larger ASL/ISL dataset
- [ ] Support dynamic sign recognition
- [ ] Use LSTM/GRU/Transformer models for gesture sequences
- [ ] Improve sentence-level language understanding
- [ ] Add automatic punctuation
- [ ] Add multilingual speech output
- [ ] Build a web or mobile version
- [ ] Improve robustness to different backgrounds and lighting
- [ ] Add model evaluation metrics and confusion matrix
- [ ] Deploy the model as an assistive-technology application

---

#  Privacy

The real-time recognition pipeline processes webcam frames locally on the user's computer. The project does not require uploading webcam footage to a remote server.

---

#  Contributing

Contributions are welcome.

1. Fork the repository.
2. Create a feature branch.
3. Make your changes.
4. Commit your changes.
5. Push the branch.
6. Open a Pull Request.

Example:

```bash
git checkout -b feature/new-gesture
git add .
git commit -m "Add new gesture recognition"
git push origin feature/new-gesture
```

---

#  License

This project is intended for educational, research, and demonstration purposes.

If you add third-party datasets, images, models, or other resources, make sure you comply with their respective licenses and attribution requirements.

---

#  Author

**Janani J**

[Janani J | LinkedIn](https://www.linkedin.com/in/janani-j-aa35532a4/)

B.Tech – Computer Science and Business Systems

Areas of interest:

- Artificial Intelligence
- Computer Vision
- Data Analytics
- Human-Centered Technology
- Assistive Technology
- Social Innovation

---

##  If you find VisionSpeak interesting

Consider giving the repository a ⭐ and contributing improvements to help make gesture-based communication technology more accessible.
