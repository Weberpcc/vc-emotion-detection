from pathlib import Path
import pandas as pd
import pickle
import json
import logging

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score
)


# ---------------- Paths ---------------- #

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = PROJECT_ROOT / "models" / "model.pkl"
FEATURES_DIR = PROJECT_ROOT / "data" / "features"
REPORTS_DIR = PROJECT_ROOT / "reports"


# ---------------- Logging ---------------- #

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


# ---------------- Load Model ---------------- #

def load_model():
    try:
        with open(MODEL_PATH, "rb") as file:
            return pickle.load(file)

    except FileNotFoundError:
        logger.error("model.pkl not found")
        raise

    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise


# ---------------- Load Test Data ---------------- #

def load_test_data():
    try:
        test_data = pd.read_csv(
            FEATURES_DIR / "test_bow.csv"
        )

        X_test = test_data.iloc[:, :-1].values
        y_test = test_data.iloc[:, -1].values

        return X_test, y_test

    except FileNotFoundError:
        logger.error("Test data not found")
        raise

    except Exception as e:
        logger.error(
            f"Error loading test data: {e}"
        )
        raise


# ---------------- Evaluate Model ---------------- #

def evaluate_model(model, X_test, y_test):
    try:
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(
            y_test,
            y_pred
        )

        precision = precision_score(
            y_test,
            y_pred,
            pos_label=1
        )

        recall = recall_score(
            y_test,
            y_pred,
            pos_label=1
        )

        auc = roc_auc_score(
            y_test,
            y_pred_proba
        )

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "auc": auc
        }

    except Exception as e:
        logger.error(
            f"Error evaluating model: {e}"
        )
        raise


# ---------------- Save Metrics ---------------- #

def save_metrics(metrics):
    try:
        REPORTS_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        metrics_path = REPORTS_DIR / "metrics.json"

        with open(metrics_path, "w") as file:
            json.dump(
                metrics,
                file,
                indent=4
            )

        logger.info(
            f"Metrics saved to {metrics_path}"
        )

    except Exception as e:
        logger.error(
            f"Error saving metrics: {e}"
        )
        raise


# ---------------- Main Pipeline ---------------- #

def main():
    try:
        model = load_model()

        X_test, y_test = load_test_data()

        metrics = evaluate_model(
            model,
            X_test,
            y_test
        )

        save_metrics(metrics)

        logger.info(
            "Model evaluation completed successfully."
        )

    except Exception as e:
        logger.error(
            f"Model evaluation failed: {e}"
        )
        raise


if __name__ == "__main__":
    main()