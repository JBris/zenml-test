from zenml.client import Client
from zenml.logger import get_logger
from zenml import step, Model, pipeline, log_metadata
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
import numpy as np

logger = get_logger(__name__)
model_name = "IrisClassifier"
model = Model(
    name=model_name,
    description="Iris classifier",
    tags=["quickstart", "iris", "logistic"],
)


@step 
def get_data() -> tuple[np.ndarray, np.ndarray]:
    X, y = load_iris(return_X_y=True)
    return X, y

@step 
def get_model() -> LogisticRegression:
    clf = LogisticRegression(random_state=0)
    return clf

@step
def assess_model(clf, X, y) -> float:
    clf = clf.fit(X, y)
    clf.predict(X[:2, :])
    clf.predict_proba(X[:2, :])
    score = clf.score(X, y)

    log_metadata(metadata={"accuracy": score})
    return score

@pipeline(model=model)
def main():
    client = Client()
    run_args_train = {}

    orchf = client.active_stack.orchestrator.flavor

    sof = None
    if client.active_stack.step_operator:
        sof = client.active_stack.step_operator.flavor

    pipeline_args = {}
    pipeline_args["enable_cache"] = False

    X,y = get_data()
    clf = get_model()
    score = assess_model(clf, X, y)

if __name__ == "__main__":
    main()