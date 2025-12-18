"""Main script for California housing data analysis."""
from src.california_housing_data_prep import load_housing_data, add_high_value_flag
from src.split_train_test import split_train_test_data
from src.exceptions import DataMismatchException
from src.model_train import train_model, evaluate_model, get_model_coffecients, save_model
import logging
import traceback
import requests

# Configure basic logging to show INFO level messages and above
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s', force=True,
                    handlers=[
                        logging.FileHandler("ml_pipeline.log"),
                        logging.StreamHandler()
                    ])

logging.info("variable__name__: = %s", __name__)


def main() -> None:

    try:
        logging.info("Running the ML pipeline\n")

        """ 1. load california housing data"""
        housing_data = load_housing_data()
        logging.info("Describe loaded data:\n%s",
                     housing_data.describe().T)
        logging.info("Schema check :\n%s",
                     housing_data.dtypes)
        logging.info("Size check :\n%s",
                     housing_data.shape)

        """2. add high value flag"""
        logging.info("Adding High Value flag to data\n")
        housing_data_with_highvalue_flag = add_high_value_flag(housing_data)[0]

        logging.info("Decribe data with high value flag:\n%s",
                     housing_data_with_highvalue_flag.describe().T)
        logging.info("Schema check :\n%s",
                     housing_data_with_highvalue_flag.dtypes)
        logging.info("Size check: \n%s",
                     housing_data_with_highvalue_flag.shape)
        if len(housing_data_with_highvalue_flag) != len(housing_data):
            raise DataMismatchException(
                "Missing rows in housing_data_with_highvalue_flag")

        """3. Prepare features and target labels"""
        y = housing_data_with_highvalue_flag['HighValue']
        X = housing_data_with_highvalue_flag.drop(
            columns=['HighValue', 'MedHouseVal'])

        logging.info("Schema check X:\n%s",
                     X.dtypes)
        logging.info("Schema check y:\n%s",
                     y.dtypes)

        """4. Split Features and target labels into train and test"""
        logging.info("Splitting data into train and test subsets\n")
        X_train, X_test, y_train, y_test = split_train_test_data(X=X, y=y)
        logging.info("X_train size: %s", len(X_train))
        logging.info("y_train size: %s", len(y_train))
        logging.info("X_test size: %s", len(X_test))
        logging.info("y_test size: %s", len(y_test))

        logging.info("Verifying if Sratify worked?:\n")
        logging.info("y_train value counts: %s\n", y_train.value_counts())
        logging.info("y_test value counts: %s\n", y_test.value_counts())

        """ 5.Train the model"""
        model = train_model(X_train=X_train, y_train=y_train)

        """ 6.Evaluate a model"""
        accuracy_score = evaluate_model(
            model=model, X_test=X_test, y_test=y_test)

        logging.debug("Model accuracy score: %s", accuracy_score)

        """ 7.Inspect the model coefficients"""
        model_coef = get_model_coffecients(
            model=model, feature_names=X.columns)
        logging.debug("Feature coefficients shape: %s", model_coef.shape)
        logging.debug("Feature coefficients are: %s", model_coef)

        """ 8. Save model"""
        save_model(model, "models/logistic_regression.pkl")

    except DataMismatchException as e:
        logging.error("Data validation between steps failed %s", e)
        raise

    except Exception as err:
        logging.error("\nSomething went wrong" + str(err))
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
