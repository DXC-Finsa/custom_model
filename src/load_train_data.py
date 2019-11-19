import pandas as pd
import numpy as np

from sklearn import datasets
from sklearn.model_selection import train_test_split

def load_data():
    iris = datasets.load_iris()

    # convert to pd dataframe
    data = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                        columns= iris['feature_names'] + ['target'])

    # Split the data into training and test sets. (0.75, 0.25) split.
    train, test = train_test_split(data)
    
    # The predicted column is "quality" which is a scalar from [3, 9]
    train_x = train.drop(["target"], axis=1)
    test_x = test.drop(["target"], axis=1)
    train_y = train[["target"]]
    test_y = test[["target"]]

    return train_x, test_x, train_y, test_y