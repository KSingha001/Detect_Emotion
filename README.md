# Detect Emotion

Detect Emotion is a Python-based project for **real-time facial emotion recognition** using a webcam.  
It uses OpenCV for face detection and a trained machine learning / deep learning model to classify emotions from facial expressions.

The project includes:

- `train_model.py` – script to train the emotion detection model.
- `main.py` – script to run real-time emotion detection from a webcam.
- `haarcascade_frontalface_default.xml` – pretrained Haar Cascade classifier for face detection.
- `requirements.txt` – list of Python dependencies.

---

## Features

- Real-time video capture from your webcam.
- Face detection using Haar Cascade.
- Emotion prediction for detected faces.
- On-screen display of detected emotion labels.

> Note: The exact emotions detected (e.g. happy, sad, angry, surprised, neutral, etc.) depend on how the model is defined and trained in `train_model.py`.

---

## Project Structure

```text
Detect_Emotion/
├── .gitignore
├── haarcascade_frontalface_default.xml   # OpenCV Haar Cascade for face detection
├── main.py                               # Real-time emotion detection
├── requirements.txt                      # Python dependencies
└── train_model.py                        # Model training script
```

---

## Prerequisites

Make sure you have the following installed:

- **Python**: 3.8+ (recommended)
- A working **webcam** (for real-time detection)
- **pip** (Python package manager)

You may optionally want to use a virtual environment (recommended):

- `venv` (built-in with Python)
- or tools like `conda`

---

## Setup Instructions (Step by Step)

### 1. Clone the repository

```bash
git clone https://github.com/KSingha001/Detect_Emotion.git
cd Detect_Emotion
```

### 2. (Optional) Create and activate a virtual environment

**On Windows:**

```bash
python -m venv .venv
.venv\Scripts\activate
```

**On macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

You should now see `(.venv)` or similar in your terminal prompt.

### 3. Install dependencies

All required packages are listed in `requirements.txt`.

```bash
pip install -r requirements.txt
```

If you have both Python 2 and 3 installed, you may need to use:

```bash
python3 -m pip install -r requirements.txt
```

---

## Training the Model

> If the repository already contains a saved model file (e.g. `model.h5`, `emotion_model.pkl`, etc.), and `main.py` loads it directly, you may **not** need to retrain.  
> If not, or if you want to update the model, follow these steps.

1. Ensure that any required **dataset paths** and **output model paths** inside `train_model.py` are correctly configured.
   - Common patterns might include:
     - A folder with images sorted into subfolders per emotion.
     - A path variable like `DATASET_DIR`, `TRAIN_DIR`, or similar.
     - A model-save path variable like `MODEL_PATH` or similar.

2. Run the training script:

   ```bash
   python train_model.py
   ```

   or, depending on your system:

   ```bash
   python3 train_model.py
   ```

3. After training completes, verify that the **model file** (for example `model.h5`, `emotion_model.h5`, or similar) is saved in the expected location.  
   `main.py` will typically expect this file in a specific path; make sure they match.

---

## Running Real-Time Emotion Detection

Once dependencies are installed and the trained model file is available:

1. Make sure your webcam is connected and accessible.

2. Run the main script:

   ```bash
   python main.py
   ```

   or:

   ```bash
   python3 main.py
   ```

3. A window should open showing the webcam feed:
   - Faces will be detected using `haarcascade_frontalface_default.xml`.
   - The model will predict the emotion for each detected face.
   - The predicted emotion label will be overlaid on the video frame.

4. To stop the program:
   - Focus on the video window.
   - Press the key defined in `main.py` to exit (commonly `q` or `Esc`).

---

## How It Works (High-Level Explanation)

1. **Face Detection**
   - The script uses OpenCV’s `CascadeClassifier` with `haarcascade_frontalface_default.xml` to detect faces in each frame from the webcam.

2. **Preprocessing**
   - For each detected face, the corresponding region of interest (ROI) is extracted.
   - The ROI is resized and processed (e.g. grayscale conversion, normalization) to match the input format expected by the model.

3. **Emotion Classification**
   - The preprocessed face is passed through the trained model (defined and saved by `train_model.py`).
   - The model outputs probabilities for each emotion class.
   - The emotion with the highest probability is selected as the prediction.

4. **Display**
   - Bounding boxes and emotion labels are drawn on the video frame using OpenCV.
   - The annotated frame is shown in a window in real time.

---

## Customization

You can customize the project in several ways:

- **Change emotions / classes**  
  Modify the model and dataset in `train_model.py` to support different or additional emotion categories.

- **Tune model architecture**  
  If a deep learning framework (e.g. TensorFlow / Keras / PyTorch) is used in `train_model.py`, you can:
  - Change the network layers.
  - Adjust input size.
  - Try different optimizers, learning rates, etc.

- **Adjust detection parameters**  
  In `main.py`, parameters like:
  - Scale factor
  - Min neighbors
  - Minimum face size  
  for the Haar Cascade can be tuned to improve detection accuracy or performance.

- **Change input source**  
  Instead of a webcam, you can:
  - Use a video file.
  - Run emotion detection on static images.

---

## Troubleshooting

- **No webcam window appears**
  - Check that your webcam is not being used by another application.
  - Make sure `cv2.VideoCapture(0)` (or similar) is correctly opening the camera; try changing `0` to `1` or another index.

- **Import errors or missing packages**
  - Re-run:
    ```bash
    pip install -r requirements.txt
    ```
  - Ensure you are in the correct virtual environment.

- **Model file not found**
  - Confirm the path in `main.py` matches where `train_model.py` saves the model.
  - Update the path in the script if needed.

---

## Future Improvements (Ideas)

- Add a simple **GUI** with buttons and status indicators.
- Support **multi-face** emotion tracking with IDs.
- Log predictions over time for each session.
- Integrate with external services (e.g. saving statistics to a database or API).
- Improve accuracy using more advanced models or transfer learning.

---

## License

Specify your license here (e.g. MIT, Apache 2.0), for example:

```text
This project is currently unlicensed. Please contact the repository owner for usage permissions.
```

Or add a proper `LICENSE` file and reference it:

```text
This project is licensed under the MIT License – see the LICENSE file for details.
```