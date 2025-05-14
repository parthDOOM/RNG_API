from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
import numpy as np
import os
import joblib # For loading the scaler
from tensorflow.keras.models import load_model
import math
from scipy.stats import chisquare, entropy, kstest, skew, kurtosis, norm
from scipy.fftpack import fft
from scipy.signal import periodogram
import time
import traceback # For detailed error logging

# --- Constants from your notebook (adjust if necessary) ---
EXPECTED_FEATURE_LENGTH = 131
SEQ_LENGTH = 1024

RNG_CLASSES = {
    0: "Healthy",
    1: "Biased",
    2: "Stuck_00",
    3: "Stuck_FF",
    4: "ReducedEntropy",
    5: "Periodic",
    6: "Correlated",
    7: "LCG_Flawed"
}
NUM_CLASSES = len(RNG_CLASSES)

# --- Feature Extraction Functions (Adapted from Cell 4 of your notebook) ---
def runs_test(sequence):
    """
    Performs the Wald-Wolfowitz runs test for randomness on a sequence of bytes.
    Args:
        sequence (bytes or np.ndarray): The input byte sequence.
    Returns:
        float: The p-value of the runs test.
    """
    n = len(sequence)
    if n < 2: return 0.5

    if isinstance(sequence, bytes):
        seq_int = np.frombuffer(sequence, dtype=np.uint8)
    elif not isinstance(sequence, np.ndarray) or sequence.dtype != np.uint8:
        seq_int = np.array(sequence, dtype=np.uint8)
    else:
        seq_int = sequence

    if len(seq_int) < 2: return 0.5

    median_val = np.median(seq_int)
    signs = np.sign(seq_int - median_val)

    for i in range(n):
        if signs[i] == 0:
            if i > 0:
                signs[i] = signs[i-1] if signs[i-1] != 0 else 1
            else:
                signs[i] = 1

    if np.all(signs == signs[0]) or np.all(signs == 0):
        return 0.0

    runs = 1
    for i in range(1, n):
        if signs[i] != signs[i-1] and signs[i] != 0 and signs[i-1] != 0:
            runs += 1

    n1 = np.sum(signs > 0)
    n2 = np.sum(signs < 0)

    if n1 == 0 or n2 == 0:
        return 0.0

    if (n1 + n2) <= 1: return 0.5

    runs_exp = ((2.0 * n1 * n2) / (n1 + n2)) + 1
    std_dev_sq_numerator = (2.0 * n1 * n2 * (2.0 * n1 * n2 - n1 - n2))
    std_dev_sq_denominator = (((n1 + n2)**2) * (n1 + n2 - 1.0))

    if std_dev_sq_denominator == 0:
         return 0.0 if abs(runs - runs_exp) > 1e-6 else 0.5

    std_dev_sq = std_dev_sq_numerator / std_dev_sq_denominator

    if std_dev_sq <= 0:
        return 0.0 if abs(runs - runs_exp) > 1e-6 else 0.5

    std_dev_runs = np.sqrt(std_dev_sq)
    if std_dev_runs < 1e-9:
        return 0.0 if abs(runs - runs_exp) > 1e-6 else 0.5

    z = (runs - runs_exp) / std_dev_runs
    p_value = 2.0 * (1.0 - norm.cdf(abs(z)))
    return p_value if not math.isnan(p_value) else 0.5


def extract_enhanced_features(sequence: bytes) -> np.ndarray:
    """
    Extracts a comprehensive set of statistical features from a byte sequence.
    Args:
        sequence (bytes): The input byte sequence.
    Returns:
        np.ndarray: A NumPy array of extracted features.
    """
    seq_int = np.frombuffer(sequence, dtype=np.uint8)
    n = len(seq_int)
    print(f"Inside extract_enhanced_features: n = {n}")
    if n > 0:
        print(f"First 10 bytes (int): {seq_int[:10]}")

    num_fft_coeffs = 64
    num_psd_coeffs = 32
    num_bins = 16
    num_lags = 8
    num_basic_stats = 11

    current_expected_len = num_basic_stats + num_lags + num_fft_coeffs + num_psd_coeffs + num_bins
    if current_expected_len != EXPECTED_FEATURE_LENGTH:
        print(f"Warning: Calculated feature length {current_expected_len} does not match global EXPECTED_FEATURE_LENGTH {EXPECTED_FEATURE_LENGTH}.")

    if n == 0:
        print("Warning: Empty sequence provided to extract_enhanced_features. Returning zeros.")
        return np.zeros(EXPECTED_FEATURE_LENGTH)

    mean_val = np.mean(seq_int) if n > 0 else 0.0
    std_dev = np.std(seq_int) if n > 0 else 0.0
    print(f"Mean: {mean_val}, StdDev: {std_dev}")

    if n > 3 and std_dev > 1e-9:
        skew_val = skew(seq_int)
        kurtosis_val = kurtosis(seq_int)
    else:
        skew_val = 0.0
        kurtosis_val = -3.0 if n > 0 else 0.0

    counts = np.bincount(seq_int, minlength=256)
    ent_val = 0.0
    if n > 0:
        prob_dist = counts / n
        ent_val = entropy(prob_dist, base=2)
    print(f"Entropy: {ent_val}")

    # --- MODIFIED CHI-SQUARED LOGIC ---
    chi_p = 1.0 # Default p-value (high, suggests randomness or test not applicable)
    if n > 0:
        # Test for uniformity across all 256 possible byte values.
        # scipy.stats.chisquare, when f_exp is not provided, assumes expected frequencies
        # are uniform (i.e., n/256 for each of the 256 categories).
        # This is consistent with the original notebook's direct approach.
        # The function itself will issue a warning if expected frequencies are too low (e.g. < 5).
        # For n=1024, expected is 4 per bin. This might give a warning but should run without sum mismatch.
        if n >= 20: # A general threshold for chi-squared to be somewhat meaningful
            try:
                chi_stat, chi_p = chisquare(f_obs=counts)
            except ValueError as e:
                # This might occur if, for example, n is so small that all bins have counts < 1
                # and scipy's internal checks fail, though it usually just warns.
                print(f"Chi-squared test (on all 256 bins) failed with ValueError: {e}. Defaulting chi_p to 1.0")
                chi_p = 1.0 # Default to pass if test fails unexpectedly
        # else: for n < 20, chi_p remains 1.0 (test not reliable or not performed)
    print(f"Chi-P: {chi_p}")
    # --- END OF MODIFIED CHI-SQUARED LOGIC ---

    ks_p = 1.0
    if n > 0:
        ks_stat, ks_p = kstest(seq_int / 255.0, 'uniform')
    print(f"KS-P: {ks_p}")

    autocorrs = []
    lags_to_use = [1, 2, 3, 5, 8, 13, 21, 34][:num_lags]
    for lag in lags_to_use:
        if lag < n:
            valid_data = seq_int[:-lag]
            valid_data_lagged = seq_int[lag:]
            if len(valid_data) > 1 and np.std(valid_data) > 1e-9 and len(valid_data_lagged) > 1 and np.std(valid_data_lagged) > 1e-9:
                corr = np.corrcoef(valid_data, valid_data_lagged)[0, 1]
                autocorrs.append(corr if not math.isnan(corr) else 0.0)
            else:
                autocorrs.append(0.0)
        else:
            autocorrs.append(0.0)
    autocorrs.extend([0.0] * (num_lags - len(autocorrs)))
    print(f"Autocorrs (first 3): {autocorrs[:3]}")

    runs_pval = runs_test(seq_int) if n > 1 else 0.5
    print(f"Runs-PVal: {runs_pval}")

    fft_features = np.zeros(num_fft_coeffs)
    if n > 1:
        fft_coeffs_complex = fft(seq_int - (mean_val if n > 0 else 0.0) )
        fft_coeffs_abs = np.abs(fft_coeffs_complex)
        num_fft_points_to_consider = min(n // 2, num_fft_coeffs)
        if num_fft_points_to_consider > 0:
            valid_fft_coeffs = fft_coeffs_abs[1 : num_fft_points_to_consider + 1]
            log_fft = np.log(valid_fft_coeffs + 1e-10)
            fft_features[:len(log_fft)] = log_fft
            if len(log_fft) > num_fft_coeffs:
                 fft_features = log_fft[:num_fft_coeffs]
    print(f"FFT features (sum): {np.sum(fft_features)}")

    psd_features = np.zeros(num_psd_coeffs)
    if n > 1:
        freqs, psd = periodogram(seq_int, fs=1.0)
        num_psd_points_to_consider = min(len(psd) - 1, num_psd_coeffs)
        if num_psd_points_to_consider > 0 :
            valid_psd = psd[1 : num_psd_points_to_consider + 1]
            log_psd = np.log(valid_psd + 1e-10)
            psd_features[:len(log_psd)] = log_psd
            if len(log_psd) > num_psd_coeffs:
                psd_features = log_psd[:num_psd_coeffs]
    print(f"PSD features (sum): {np.sum(psd_features)}")

    high_bytes = np.sum(seq_int >= 128) / n if n > 0 else 0.0
    byte_transitions = np.mean(np.abs(np.diff(seq_int.astype(float)))) if n > 1 else 0.0
    
    monotonic_runs_count = 0
    if n > 2:
        diffs = np.sign(np.diff(seq_int.astype(float)))
        diffs_no_zeros = diffs[diffs != 0]
        if len(diffs_no_zeros) > 1:
            monotonic_runs_count = np.sum(np.diff(diffs_no_zeros) != 0) + 1
        elif len(diffs_no_zeros) == 1:
            monotonic_runs_count = 1
    elif n == 2 and seq_int[0] != seq_int[1]:
        monotonic_runs_count = 1
    print(f"Monotonic runs: {monotonic_runs_count}")

    bins_hist, _ = np.histogram(seq_int, bins=num_bins, range=(0, 256)) if n > 0 else (np.zeros(num_bins), None)
    bins_normalized = bins_hist / n if n > 0 else np.zeros(num_bins)

    feature_vector_list = [
        mean_val / 255.0, std_dev / 128.0 if std_dev > 1e-9 else 0.0,
        ent_val / 8.0, chi_p, ks_p, skew_val, kurtosis_val, runs_pval,
        high_bytes, byte_transitions / 255.0,
        monotonic_runs_count / n if n > 0 else 0.0
    ]
    feature_vector_list.extend(autocorrs)
    feature_vector_list.extend(list(fft_features))
    feature_vector_list.extend(list(psd_features))
    feature_vector_list.extend(list(bins_normalized))

    feature_vector = np.array(feature_vector_list)
    feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=1e6, neginf=-1e6)

    if len(feature_vector) != EXPECTED_FEATURE_LENGTH:
        print(f"Critical Warning: Final feature vector length mismatch! Expected {EXPECTED_FEATURE_LENGTH}, Got {len(feature_vector)}. Adjusting...")
        if len(feature_vector) < EXPECTED_FEATURE_LENGTH:
            feature_vector = np.pad(feature_vector, (0, EXPECTED_FEATURE_LENGTH - len(feature_vector)), 'constant', constant_values=0)
        else:
            feature_vector = feature_vector[:EXPECTED_FEATURE_LENGTH]
    
    print(f"Feature vector (first 5): {feature_vector[:5]}")
    return feature_vector
# --- End of Feature Extraction Functions ---


app = FastAPI(
    title="RNG Flaw Detection API",
    description="Predicts the health/flaw type of a TRNG sequence from a .bin file using a multi-class model."
)

MODEL_PATH = "rng_flaw_multiclass_model.h5"
SCALER_PATH = "rng_flaw_multiclass_scaler.pkl"
model = None
scaler = None

@app.on_event("startup")
async def load_resources():
    global model, scaler
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}.")
    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(f"Scaler file not found at {SCALER_PATH}.")

    print(f"Loading model from {MODEL_PATH}...")
    model = load_model(MODEL_PATH)
    print("Model loaded successfully.")

    print(f"Loading scaler from {SCALER_PATH}...")
    scaler = joblib.load(SCALER_PATH)
    print("Scaler loaded successfully.")
    
    if hasattr(scaler, 'n_features_in_') and scaler.n_features_in_ != EXPECTED_FEATURE_LENGTH:
        print(f"Warning: Scaler expects {scaler.n_features_in_} features, but EXPECTED_FEATURE_LENGTH is {EXPECTED_FEATURE_LENGTH}.")
    elif not hasattr(scaler, 'n_features_in_'):
         print("Warning: Cannot verify scaler's expected input feature length.")


class PredictionResponse(BaseModel):
    predicted_class_id: int
    predicted_class_name: str
    probabilities: dict[str, float]
    detail: str
    input_file_size_bytes: int
    processing_time_seconds: float


@app.get("/health")
async def health_check():
    if model is not None and scaler is not None:
        return {"status": "API is up and running. Model and scaler loaded."}
    else:
        return {"status": "API is up, but model/scaler might not be loaded correctly."}

@app.post("/predict_bin/", response_model=PredictionResponse)
async def predict_from_bin(file: UploadFile = File(...)):
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model or scaler not loaded.")

    request_start_time = time.time()
    binary_content = b''
    file_size = 0

    try:
        print(f"\n--- New Request for file: {file.filename} ---")
        binary_content = await file.read()
        file_size = len(binary_content)
        print(f"Received file: {file.filename}, size: {file_size} bytes")
    except Exception as e:
        print(f"Error reading uploaded file: {file.filename}, Error: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=f"Error reading uploaded file: {str(e)}")
    finally:
        await file.close()

    if not binary_content:
        print(f"File {file.filename} is empty.")
        raise HTTPException(status_code=400, detail="Uploaded .bin file is empty.")

    features = None
    try:
        print(f"Extracting features from binary content of {file.filename} (size: {file_size} bytes)...")
        extraction_start_time = time.time()
        features = extract_enhanced_features(binary_content)
        extraction_time = time.time() - extraction_start_time
        print(f"Features extracted successfully for {file.filename}, shape: {features.shape}, time: {extraction_time:.4f}s")
        if np.isnan(features).any() or np.isinf(features).any():
            print(f"WARNING: NaN or Inf values detected in features for {file.filename} AFTER extraction. Features (first 10): {features[:10]}")
    except Exception as e:
        print(f"Error during feature extraction for {file.filename}: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error during feature extraction: {str(e)}")

    scaled_features = None
    try:
        print(f"Scaling features for {file.filename}...")
        scaling_start_time = time.time()
        scaled_features = scaler.transform(features.reshape(1, -1))
        scaling_time = time.time() - scaling_start_time
        print(f"Features scaled successfully for {file.filename}, shape: {scaled_features.shape}, time: {scaling_time:.4f}s")
        if np.isnan(scaled_features).any() or np.isinf(scaled_features).any():
            print(f"WARNING: NaN or Inf values detected in scaled_features for {file.filename}. Scaled features (first 10): {scaled_features[0,:10]}")
            scaled_features = np.nan_to_num(scaled_features, nan=0.0, posinf=1e6, neginf=-1e6)
            print(f"Attempted to clean NaN/Inf in scaled_features for {file.filename}. New (first 10): {scaled_features[0,:10]}")
    except Exception as e:
        print(f"Error during feature scaling for {file.filename}: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error during feature scaling: {str(e)}")

    predicted_class_id = -1
    predicted_class_name = "Error"
    class_probabilities = {}
    try:
        print(f"Making prediction for {file.filename}...")
        prediction_start_time = time.time()
        probabilities_all = model.predict(scaled_features)[0]
        prediction_time = time.time() - prediction_start_time
        
        predicted_class_id = int(np.argmax(probabilities_all))
        predicted_class_name = RNG_CLASSES.get(predicted_class_id, f"Unknown Class ID: {predicted_class_id}")
        print(f"Prediction made for {file.filename}: {predicted_class_name} (ID: {predicted_class_id}), time: {prediction_time:.4f}s")

        class_probabilities = {RNG_CLASSES.get(i, f"Class_{i}"): float(prob) for i, prob in enumerate(probabilities_all)}
    except Exception as e:
        print(f"Error during model prediction for {file.filename}: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error during model prediction: {str(e)}")

    total_processing_time = time.time() - request_start_time
    print(f"Total processing time for {file.filename}: {total_processing_time:.4f} seconds")

    return PredictionResponse(
        predicted_class_id=predicted_class_id,
        predicted_class_name=predicted_class_name,
        probabilities=class_probabilities,
        detail=f"Successfully processed {file.filename}.",
        input_file_size_bytes=file_size,
        processing_time_seconds=round(total_processing_time, 4)
    )

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    print(f"Starting Uvicorn server on host 0.0.0.0 and port {port}")
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        print(f"CRITICAL ERROR: Model ('{MODEL_PATH}') or Scaler ('{SCALER_PATH}') file not found.")
        print("Please ensure these files are in the current working directory or update paths.")
    else:
        uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
