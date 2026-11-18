Here is the task description detailing what is accomplished across all the scripts in the project:
Project Task Description: Automated MLOps Pipeline for Iris Classification
This project implements an end-to-end MLOps automation pipeline that continuously monitors a machine learning service for input data degradation, triggers conditional retraining when necessary, and validates the system through automated CI/CD checks.
Component Breakdown:
 * app.py (Model Serving): Hosts the machine learning model using FastAPI. It exposes a /predict endpoint to accept feature inputs and return classifications, along with a /health endpoint to monitor API availability, including a startup fallback to train a default model if none exists.
 * retrain.py (Automated Retraining): Fetches the raw dataset, retrains a fresh Random Forest Classifier instance, and saves the updated model weights (model.pkl) to refresh the production artifact.
 * drift_monitor.py (Data Drift Detection): Runs a statistical Kolmogorov-Smirnov (KS) test (scipy.stats.ks_2samp) comparing baseline training data against incoming production data. If statistical drift is found beyond the threshold, it flags the event for the CI/CD runner.
 * requirements.txt (Dependency Management): Specifies and locks down all required Python packages (fastapi, uvicorn, scikit-learn, numpy, scipy, joblib) to ensure consistent execution environments.
 * deploy.yml (CI/CD Orchestration): Automates the entire lifecycle via GitHub Actions:
   * Checks for data drift using drift_monitor.py.
   * Conditionally triggers retrain.py only if drift is detected.
   * Spins up the FastAPI server locally to execute automated smoke and integration tests (curl health and prediction checks).
   * Verifies the final model artifact for deployment readiness.
     
:: 1. Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate.bat

:: 2. Install dependencies
pip install -r requirements.txt

:: 3. (Optional) Run drift monitoring
python drift_monitor.py

:: 4. (Optional) Train/retrain the model
python retrain.py

:: 5. Start the FastAPI server
uvicorn app:app --reload --host 127.0.0.1 --port 8000
