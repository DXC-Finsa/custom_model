from sklearn.neighbors import KNeighborsClassifier
import mlflow.pyfunc

from src.preprocess import preprocess
from src.postprocess import postprocess

# Define the model class
class KNN_Customized(mlflow.pyfunc.PythonModel):

    def __init__(self, n):
        self.n = n
        self.knn_clf = KNeighborsClassifier(n_neighbors=self.n)
        
    def fit(self, train_x, train_y):
        train_x = preprocess(train_x)
        self.knn_clf.fit(train_x, train_y)

    def predict(self, context, model_input):
        model_input = preprocess(model_input)
        predicted_target = self.knn_clf.predict(model_input)
        predicted_target = postprocess(predicted_target)
        return predicted_target