import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score




def rebalance(unbalanced_data):


    # Separate majority and minority classes
    zeros = unbalanced_data[unbalanced_data.label==0]
    ones = unbalanced_data[unbalanced_data.label==1]

    if len(zeros) > len(ones):
    # Upsample minority class
        data_minority = ones
        data_majority = zeros
        n_samples = len(data_majority)
    elif len(zeros) < len(ones):
        data_minority = zeros
        data_majority = ones
        n_samples = len(data_majority)
    else:
        return unbalanced_data

    print len(data_minority)
    print len(data_majority)

    data_minority_upsampled = resample(data_minority, replace=True, n_samples=n_samples, random_state=5)

    # Combine majority class with upsampled minority class
    data_upsampled = pd.concat([data_majority, data_minority_upsampled])

    data_upsampled.sort_index(inplace=True)

    # Display new class counts
    # print data_upsampled.label.value_counts()

    return data_upsampled


def normalize(x):

    scaler = StandardScaler()
    x_norm = scaler.fit_transform(x.values)
    x_norm = pd.DataFrame(x_norm, index=x.index, columns=x.columns)
    return x_norm

def scores(models, X, y):

    for model in models:
        y_pred = model.predict(X)
        acc = accuracy_score(y, y_pred)
        f1 = f1_score(y, y_pred)
        auc = roc_auc_score(y, y_pred)
        print str(model).partition("(")[0]
        print("Accuracy Score: {0:0.2f} %".format(acc * 100))
        #print("F1 Score: {0:0.4f}".format(f1))
        #print("Area Under ROC Curve Score: {0:0.4f}".format(auc))