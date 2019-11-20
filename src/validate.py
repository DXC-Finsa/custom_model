# Metrics: Classification
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score

from sklearn.model_selection import cross_val_score

def eval_metrics(actual, pred):
    acc = accuracy_score(actual, pred)
    prec = precision_score(actual, pred, average='micro')
    f1 = f1_score(actual, pred, average='micro')
    return acc, prec, f1

