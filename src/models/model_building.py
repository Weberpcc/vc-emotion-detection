from pathlib import Path
import pandas as pd
import pickle
import yaml
import logging

from sklearn.ensemble import GradientBoostingClassifier


# ---------------- Paths ---------------- #

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PARAMS_FILE = PROJECT_ROOT / "params.yaml"
FEATURES_DIR = PROJECT_ROOT / "data" / "features"
MODELS_DIR = PROJECT_ROOT / "models"


# ---------------- Logging ---------------- #

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


# ---------------- Load Parameters ---------------- #

def load_params():
    try:
        with open(PARAMS_FILE, "r") as file:
            params = yaml.safe_load(file)

        return params["model_building"]

    except FileNotFoundError:
        logger.error("params.yaml not found")
        raise

    except KeyError:
        logger.error(
            "model_building parameters not found"
        )
        raise


# ---------------- Load Data ---------------- #

def load_data():
    try:
        train_data = pd.read_csv(
            FEATURES_DIR / "train_tfidf.csv"
        )

        X_train = train_data.iloc[:, :-1].values
        y_train = train_data.iloc[:, -1].values

        return X_train, y_train

    except FileNotFoundError:
        logger.error("Training data not found")
        raise

    except Exception as e:
        logger.error(
            f"Error loading training data: {e}"
        )
        raise


# ---------------- Train Model ---------------- #

def train_model(X_train, y_train, params):
    try:
        clf = GradientBoostingClassifier(
            n_estimators=params["n_estimators"],
            learning_rate=params["learning_rate"],
            max_depth=1,
            random_state=0
        )

        clf.fit(X_train, y_train)

        return clf

    except Exception as e:
        logger.error(
            f"Error training model: {e}"
        )
        raise


# ---------------- Save Model ---------------- #

def save_model(model):
    try:
        MODELS_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        model_path = MODELS_DIR / "model.pkl"

        with open(model_path, "wb") as file:
            pickle.dump(model, file)

        logger.info(
            f"Model saved to {model_path}"
        )

    except Exception as e:
        logger.error(
            f"Error saving model: {e}"
        )
        raise


# ---------------- Main Pipeline ---------------- #

def main():
    try:
        params = load_params()

        X_train, y_train = load_data()

        model = train_model(
            X_train,
            y_train,
            params
        )

        save_model(model)

        logger.info(
            "Model training completed successfully."
        )

    except Exception as e:
        logger.error(
            f"Model building failed: {e}"
        )
        raise


if __name__ == "__main__":
    main()