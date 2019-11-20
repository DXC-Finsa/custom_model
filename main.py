import sys, time

import mlflow
import mlflow.pyfunc

from src.load_train_data import load_data
from src.model_class import KNN_Customized
from src.validate import eval_metrics

# SET EXPERIMENT
mlflow.set_experiment("custom_model")

# USER INPUT PARAMETERS
n = int(sys.argv[1]) if len(sys.argv) > 1 else 5

# Load training data
train_x, test_x, train_y, test_y = load_data()

# Construct and train the model
start_time = time.time()
model_path = "custom_knn_conda"
custom_model = KNN_Customized(n=n)
custom_model.fit(train_x, train_y)
print("Model trained. Train time: {} s".format(time.time() - start_time))

# Guardamos el modelo en local y cargamos el contexto para poder evaluarlo
#mlflow.pyfunc.save_model(path=model_path, python_model=custom_model)
#context = mlflow.pyfunc.PythonModelContext({'custom_model':model_path})

# Calculate CV metrics
pred = custom_model.predict(test_x)
acc, prec, f1 = eval_metrics(test_y, pred)

with mlflow.start_run():
    # Log training parameters
    mlflow.log_param("n", n)

    # Log metrics
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("f1_score", f1)

    print("  knn model no of neighbors:", n)
    print("  accuracy: %s" % acc)
    print("  precision score: %s" % prec)
    print("  f1 score: %s" % f1)

    # Set tags
    mlflow.set_tags({'pipeline':'norm + dummie',
                     'algorithm':'knn'})

    # Log the model
    mlflow.pyfunc.log_model(artifact_path=model_path, python_model=custom_model, conda_env="conda.yaml")

    print("Model saved in run: {}".format(mlflow.active_run().info.run_uuid))
    print("Serve model: {}".format("·"))