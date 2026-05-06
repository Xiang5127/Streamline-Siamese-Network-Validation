# Building Identity Verification with Siamese Network

A Streamlit web application for validating building identity from images using a Siamese neural network. The system compares reference and live building photos by extracting deep feature embeddings and measuring their Euclidean distance.

## Live Deployment

Access the deployed application here:

[https://streamline-siamese-network-validation.streamlit.app/](https://streamline-siamese-network-validation.streamlit.app/)

## Project Overview

This project uses a Siamese neural network approach to determine whether two building images represent the same building or different buildings. The model generates an embedding for each image using a MobileNetV2-based feature extractor, then compares the embeddings using Euclidean distance.

A lower distance means the two images are more similar. If the distance is below the selected threshold, the app classifies the pair as a match.

## Main Features

- **Single Comparison**: Upload one reference image and one live image to verify whether they show the same building.
- **Batch Testing**: Upload multiple predefined image pairs and evaluate them in one run.
- **Auto Pairing**: Upload multiple images and automatically compare every possible image pair.
- **Model Validation**: Evaluate model performance using a validation folder and CSV pair labels.
- **Threshold Control**: Adjust the match threshold directly from the sidebar.
- **Validation Metrics**: View accuracy, precision, recall, F1 score, FAR, FRR, EER, AUC-ROC, confusion matrix, and distance distribution.

## Tech Stack

- Python
- Streamlit
- TensorFlow Lite
- TensorFlow / Keras
- MobileNetV2
- NumPy
- Pandas
- Pillow
- Altair

## Repository Structure

```text
FYP_siamese/
├── app.py
├── building_dna_extractor.tflite
├── requirements.txt
├── pairs_train.csv
├── pairs_val.csv
├── create_validation_folder.py
├── generate_test_pairs.py
├── demo_pairs/
├── raw_datasets_terracepacels/
├── validation_folder/
└── phases/
    ├── phase1_pair_generator/
    │   └── generate_pairs.py
    ├── phase2_data_pipeline/
    │   └── data_pipeline.py
    ├── phase3_siamese_model/
    │   └── siamese_model.py
    └── phase4_train_export/
        └── train.py
```

## Installation

1. Clone the repository.

```bash
git clone <repository-url>
cd FYP_siamese
```

2. Create and activate a virtual environment.

```bash
python -m venv .venv
```

On Windows:

```bash
.venv\Scripts\activate
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

3. Install dependencies.

```bash
pip install -r requirements.txt
```

## Running the App Locally

Start the Streamlit application with:

```bash
streamlit run app.py
```

Or run with a specific port:

```bash
python -m streamlit run app.py --server.headless true --server.port 8502
```

The app requires `building_dna_extractor.tflite` to be available in the project root.

## How to Use

### Single Comparison

1. Select **Single Comparison** from the sidebar.
2. Upload a reference building image.
3. Upload a live building image.
4. Adjust the match threshold if needed.
5. Review the Euclidean distance and match decision.

### Batch Testing

1. Select **Batch Testing**.
2. Choose the number of image pairs.
3. Upload each reference/live pair.
4. Click **Run Batch Test**.
5. Review individual results and the summary report.

### Auto Pairing

1. Select **Auto Pairing**.
2. Upload two or more building images.
3. Click **Run Auto Pairing**.
4. The app compares all possible image combinations.

### Model Validation

1. Select **Model Validation**.
2. Provide the validation folder path.
3. Provide the pairs CSV path.
4. Click **Run Validation**.
5. Review model metrics, confusion matrix, distance distribution, FAR/FRR curve, and per-pair results.

## Dataset Pair Format

Training and validation CSV files use the following format:

```csv
path_a,path_b,label
raw_datasets_terracepacels/siamese_terrace_1/image1.jpg,raw_datasets_terracepacels/siamese_terrace_1/image2.jpg,1
raw_datasets_terracepacels/siamese_terrace_1/image1.jpg,raw_datasets_terracepacels/siamese_terrace_2/image1.jpg,0
```

Label meaning:

- `1`: same building
- `0`: different buildings

## Model Pipeline

The project is organized into phases:

1. **Pair Generation**
   - Script: `phases/phase1_pair_generator/generate_pairs.py`
   - Generates positive and negative image pairs.
   - Outputs `pairs_train.csv` and `pairs_val.csv`.

2. **Data Pipeline**
   - Script: `phases/phase2_data_pipeline/data_pipeline.py`
   - Loads image pairs from CSV.
   - Resizes images to `224x224`.
   - Normalizes pixel values to `[0, 1]`.

3. **Siamese Model**
   - Script: `phases/phase3_siamese_model/siamese_model.py`
   - Builds a twin-tower Siamese network using a shared MobileNetV2 embedding model.
   - Produces 128-dimensional L2-normalized embeddings.

4. **Training and Export**
   - Script: `phases/phase4_train_export/train.py`
   - Trains the Siamese model with contrastive loss.
   - Extracts the embedding network.
   - Exports the model to `building_dna_extractor.tflite`.

## Training the Model

To regenerate the training and validation pair CSV files:

```bash
python phases/phase1_pair_generator/generate_pairs.py
```

To train and export the TensorFlow Lite model:

```bash
python phases/phase4_train_export/train.py
```

The exported model will be saved as:

```text
building_dna_extractor.tflite
```

## Deployment

This project is deployed on Streamlit Community Cloud:

[https://streamline-siamese-network-validation.streamlit.app/](https://streamline-siamese-network-validation.streamlit.app/)

Streamlit deployment should use Python `3.11.x`. The project includes `runtime.txt` to specify the Python runtime version for Streamlit Community Cloud.

For deployment, ensure these files are included in the repository:

- `app.py`
- `requirements.txt`
- `runtime.txt`
- `building_dna_extractor.tflite`
- Any validation/demo data required by the app

## Notes

- The default match threshold in the app is `0.8`.
- Images are resized to `224x224` before inference.
- The app uses TensorFlow Lite for efficient embedding extraction during deployment.
- Model validation depends on the selected validation folder and CSV paths being available in the runtime environment.

## Project Purpose

This project was developed as a building identity verification prototype using Siamese neural networks. It demonstrates how deep image embeddings can support automated visual similarity validation for building photographs.
