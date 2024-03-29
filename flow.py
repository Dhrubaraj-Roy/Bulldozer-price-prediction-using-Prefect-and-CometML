from prefect import flow

from steps. ingest_data import ingest_df
from steps.clean_data import clean_df
from steps.train_model import train_model
from steps.evaluation import evaluate_model
## import comet_ml at the top of your file
from comet_ml import Experiment

## Create an experiment with your api key
@flow(retries=3, retry_delay_seconds=5, log_prints=True)
def my_flow():
    data_path="/home/dhrubaubuntu/gigs_projects/Bulldozer-price-prediction/data/TrainAndValid.csv"
    df = ingest_df(data_path)
    X_train, X_test, y_train, y_test = clean_df(df)
    model = train_model(X_train, X_test, y_train, y_test)
    r2_score, rmse = evaluate_model(model, X_test, y_test)

# Run the Prefect Flow
if __name__ == "__main__":
    my_flow()