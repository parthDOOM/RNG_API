# RNG Flaw Detection API

This API provides a service to predict the health and potential flaw type of a True Random Number Generator (TRNG) sequence. It uses a pre-trained machine learning model (a TensorFlow/Keras neural network) to classify sequences based on statistical features extracted from the input binary data.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Technology Stack](#technology-stack)
- [Prerequisites](#prerequisites)
- [Setup and Local Execution](#setup-and-local-execution)
- [API Endpoints](#api-endpoints)
  - [Endpoints Summary](#endpoints-summary)
  - [Health Check](#health-check)
  - [Predict Flaw from Binary File](#predict-flaw-from-binary-file)
- [Input Details](#input-details)
- [Output Details](#output-details)
  - [Prediction Response](#prediction-response)
- [RNG Classes](#rng-classes)
- [Deployment (Render)](#deployment-render)
- [Troubleshooting](#troubleshooting)

## Overview

The API takes a binary file (typically `.bin`) containing a sequence of bytes from an RNG. It then performs the following steps:
1.  Reads the binary content.
2.  Extracts a comprehensive set of statistical features from the byte sequence. These features include mean, standard deviation, entropy, chi-squared test p-value, Kolmogorov-Smirnov test p-value, autocorrelation, FFT coefficients, PSD coefficients, and more.
3.  Scales the extracted features using a pre-trained scaler.
4.  Feeds the scaled features into a pre-trained multi-class classification model.
5.  Returns the predicted class (e.g., "Healthy", "Biased", "Periodic") and the probabilities for each class.

## Features

* **FastAPI Framework:** Modern, fast (high-performance) web framework for building APIs with Python.
* **Machine Learning Integration:** Utilizes a TensorFlow/Keras model for RNG flaw classification.
* **Comprehensive Feature Extraction:** Implements a detailed feature extraction process to capture various statistical properties of the RNG sequence.
* **Clear API Definition:** Provides well-defined request and response models using Pydantic.
* **Easy Deployment:** Includes a `render.yaml` for straightforward deployment on Render.

## Technology Stack

* **Python 3.x**
* **FastAPI:** Web framework.
* **Uvicorn:** ASGI server.
* **TensorFlow:** For loading and running the Keras model.
* **Scipy:** For statistical computations in feature extraction.
* **NumPy:** For numerical operations.
* **Joblib:** For loading the feature scaler.
* **Pydantic:** For data validation and settings management.

## Prerequisites

Before running the application locally or deploying it, ensure you have the following:

1.  **Python** (version 3.7+ recommended).
2.  **Pip** (Python package installer).
3.  The **model file**: `rng_flaw_multiclass_model.h5`
4.  The **scaler file**: `rng_flaw_multiclass_scaler.pkl`

These model and scaler files must be present in the root directory of the project where `main.py` is located, or accessible via a pre-configured download mechanism (e.g., from cloud storage if modified for such a setup).

## Setup and Local Execution

1.  **Clone the Repository (if applicable):**
    ```bash
    git clone <your-repository-url>
    cd <your-repository-directory>
    ```

2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install Dependencies:**
    Make sure your `requirements.txt` includes all necessary packages:
    ```text
    fastapi
    uvicorn[standard]
    tensorflow
    numpy
    pydantic
    scipy
    joblib
    # requests # if you implement model download from URL
    ```
    Then install them:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Place Model and Scaler Files:**
    Ensure `rng_flaw_multiclass_model.h5` and `rng_flaw_multiclass_scaler.pkl` are in the project's root directory.

5.  **Run the API:**
    ```bash
    uvicorn main:app --reload --host 0.0.0.0 --port 8000
    ```
    The API will be accessible at `http://localhost:8000`. You can view the auto-generated documentation at `http://localhost:8000/docs`.

## API Endpoints

### Endpoints Summary

| Endpoint        | Method | Description                                                                 |
|-----------------|--------|-----------------------------------------------------------------------------|
| `/health`       | GET    | Checks API status and model/scaler load.                                    |
| `/predict_bin/` | POST   | Predicts RNG flaw type from an uploaded binary file.                        |

### Health Check

* **Endpoint:** `/health`
* **Method:** `GET`
* **Description:** Checks if the API is running and if the model and scaler have been loaded successfully.
* **Input:** None
* **Output (Success):**
    ```json
    {
      "status": "API is up and running. Model and scaler loaded."
    }
    ```
* **Output (Failure/Partial Load):**
    ```json
    {
      "status": "API is up, but model/scaler might not be loaded correctly."
    }
    ```

### Predict Flaw from Binary File

* **Endpoint:** `/predict_bin/`
* **Method:** `POST`
* **Description:** Accepts a binary file (`.bin`), extracts features, and predicts the RNG flaw type.
* **Input:**
    * `file`: An uploaded binary file (`UploadFile`). The content should be raw bytes from an RNG. The API is designed to work with sequences, and the feature extraction process (`extract_enhanced_features`) expects a sequence of bytes. The `SEQ_LENGTH` constant (currently 1024 bytes in `main.py`) might imply an optimal or expected length for some internal processing, but the feature extraction itself attempts to handle variable lengths. However, model performance is typically best when input data characteristics match the training data.
* **Output (Success):** `PredictionResponse` (see [Output Details](#prediction-response))
* **Output (Error):**
    * `400 Bad Request`: If the uploaded file is empty or cannot be read.
    * `500 Internal Server Error`: If an error occurs during feature extraction, scaling, or model prediction.
    * `503 Service Unavailable`: If the model or scaler is not loaded.
    * Example Error:
        ```json
        {
          "detail": "Error during feature extraction: <specific error message>"
        }
        ```

## Input Details

For the `/predict_bin/` endpoint:

* **Content-Type:** `multipart/form-data`
* **File Parameter Name:** `file`
* **File Content:** The file should contain raw binary data from the RNG. For example, a file with 1024 bytes of random data.

**Example using `curl`:**
```bash
curl -X POST "http://localhost:8000/predict_bin/" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@/path/to/your/rng_sequence.bin"
```
Replace `/path/to/your/rng_sequence.bin` with the actual path to your binary file.

## Output Details

### Prediction Response

The `/predict_bin/` endpoint returns a JSON object with the following structure:

```json
{
  "predicted_class_id": 0,
  "predicted_class_name": "Healthy",
  "probabilities": {
    "Healthy": 0.95,
    "Biased": 0.01,
    "Stuck_00": 0.005,
    "Stuck_FF": 0.005,
    "ReducedEntropy": 0.01,
    "Periodic": 0.005,
    "Correlated": 0.01,
    "LCG_Flawed": 0.005
  },
  "detail": "Successfully processed your_file.bin.",
  "input_file_size_bytes": 1024,
  "processing_time_seconds": 0.1234
}
```

**Fields:**

* `predicted_class_id` (integer): The numerical ID of the predicted RNG class.
* `predicted_class_name` (string): The human-readable name of the predicted RNG class (see [RNG Classes](#rng-classes)).
* `probabilities` (object): A dictionary where keys are class names and values are their corresponding predicted probabilities (float, summing to approximately 1.0).
* `detail` (string): A message indicating the status of the processing, usually including the input filename.
* `input_file_size_bytes` (integer): The size of the input binary file in bytes.
* `processing_time_seconds` (float): The total time taken by the API to process the request, in seconds.

## RNG Classes

The model predicts one of the following classes for the input RNG sequence:

| ID | Class Name       | Description                                     |
|----|------------------|-------------------------------------------------|
| 0  | Healthy          | The RNG sequence appears to be random and healthy. |
| 1  | Biased           | The RNG sequence shows statistical bias.        |
| 2  | Stuck_00         | The RNG sequence is predominantly stuck at 0x00.  |
| 3  | Stuck_FF         | The RNG sequence is predominantly stuck at 0xFF.  |
| 4  | ReducedEntropy   | The RNG sequence has lower entropy than expected. |
| 5  | Periodic         | The RNG sequence exhibits periodic patterns.    |
| 6  | Correlated       | The RNG sequence shows correlations.            |
| 7  | LCG_Flawed       | The RNG sequence resembles a flawed LCG output. |

*Note: The `EXPECTED_FEATURE_LENGTH` is currently `131` and `SEQ_LENGTH` is `1024` in `main.py`. The feature extraction is designed to produce this many features. While the API can process files of varying sizes, the model was likely trained on sequences of a particular length or characteristics, and performance might vary for significantly different inputs.*

## Deployment (Render)

This API is configured for deployment on Render using the `render.yaml` file.

1.  Ensure your `requirements.txt` is up-to-date.
2.  Commit `main.py`, `requirements.txt`, `render.yaml`, `rng_flaw_multiclass_model.h5`, and `rng_flaw_multiclass_scaler.pkl` to your Git repository.
    * *Alternatively, for large model/scaler files, consider hosting them on a cloud storage service and modifying `main.py` to download them on startup. Update `render.yaml` or the Render dashboard with environment variables for the download URLs.*
3.  Connect your Git repository to a new Web Service on Render.
4.  Render should automatically detect the settings from `render.yaml`:
    * **Build Command:** `pip install -r requirements.txt`
    * **Start Command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`
5.  Deploy the service. Render will provide a public URL.

## Troubleshooting

* **Model/Scaler Not Found:** Ensure `rng_flaw_multiclass_model.h5` and `rng_flaw_multiclass_scaler.pkl` are in the correct location (root of the project for local/Render deployment, or downloadable if configured). Check logs for `FileNotFoundError`.
* **Feature Extraction Errors:** The `extract_enhanced_features` function can be sensitive to the input data (e.g., very short sequences, all zero sequences). The API includes some error handling, but unusual inputs might cause issues. Check API logs for details. The `main.py` includes many `print` statements which will appear in the logs and can help diagnose issues with feature values.
* **Dependency Issues:** Double-check that all packages in `requirements.txt` are correctly installed and compatible.
* **Render Deployment Issues:** Consult the build and runtime logs on the Render dashboard for specific error messages.

