from pathlib import Path
import logging
import pandas as pd
import yaml

from sklearn.feature_extraction.text import TfidfVectorizer


# ---------------- Paths ---------------- #

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PARAMS_FILE = PROJECT_ROOT / "params.yaml"
INTERIM_DATA_DIR = PROJECT_ROOT / "data" / "interim"
FEATURES_DIR = PROJECT_ROOT / "data" / "features"


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

        return params["feature_engineering"]["max_features"]

    except FileNotFoundError:
        logger.error("params.yaml not found")
        raise

    except KeyError:
        logger.error(
            "max_features not found in params.yaml"
        )
        raise


# ---------------- Load Data ---------------- #

def load_data():
    try:
        train_data = pd.read_csv(
            INTERIM_DATA_DIR / "train_processed.csv"
        )

        test_data = pd.read_csv(
            INTERIM_DATA_DIR / "test_processed.csv"
        )

        return train_data.fillna(""), test_data.fillna("")

    except FileNotFoundError:
        logger.error(
            "Preprocessed data files not found"
        )
        raise

    except Exception as e:
        logger.error(
            f"Error loading data: {e}"
        )
        raise


# ---------------- Create BOW Features ---------------- #

def create_tfidf_features(
    train_data,
    test_data,
    max_features
):
    try:
        X_train = train_data["content"]
        y_train = train_data["sentiment"]

        X_test = test_data["content"]
        y_test = test_data["sentiment"]

        vectorizer = TfidfVectorizer(
            max_features=max_features
        )

        X_train_bow = vectorizer.fit_transform(X_train)
        X_test_bow = vectorizer.transform(X_test)

        train_df = pd.DataFrame(
            X_train_bow.toarray()
        )
        train_df["label"] = y_train.values

        test_df = pd.DataFrame(
            X_test_bow.toarray()
        )
        test_df["label"] = y_test.values

        return train_df, test_df

    except KeyError as e:
        logger.error(
            f"Required column not found: {e}"
        )
        raise

    except Exception as e:
        logger.error(
            f"Error creating BOW features: {e}"
        )
        raise


# ---------------- Save Features ---------------- #

def save_features(train_df, test_df):
    try:
        FEATURES_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        train_df.to_csv(
            FEATURES_DIR / "train_tfidf.csv",
            index=False
        )

        test_df.to_csv(
            FEATURES_DIR / "test_tfidf.csv",
            index=False
        )

    except Exception as e:
        logger.error(
            f"Error saving features: {e}"
        )
        raise


# ---------------- Main Pipeline ---------------- #

def main():
    try:
        max_features = load_params()

        train_data, test_data = load_data()

        train_df, test_df = create_tfidf_features(
            train_data,
            test_data,
            max_features
        )

        save_features(
            train_df,
            test_df
        )

        logger.info(
            "Feature engineering completed successfully."
        )

    except Exception as e:
        logger.error(
            f"Feature engineering failed: {e}"
        )
        raise


if __name__ == "__main__":
    main()