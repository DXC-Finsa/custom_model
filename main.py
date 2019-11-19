import mlflow
import mlflow.pyfunc

from src.load_train_data import load_data
from src.model_class import KNN_Customized

# SET EXPERIMENT
mlflow.set_experiment("custom_model")

# Load training data
train_x, test_x, train_y, test_y = load_data()

# Construct and train the model
model_path = "custom_knn_conda"
custom_model = KNN_Customized(n=5)
custom_model.fit(train_x, train_y)

# Log the model
mlflow.pyfunc.log_model(artifact_path=model_path, python_model=custom_model, conda_env="conda.yaml")