import json
import pickle
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from skopt import BayesSearchCV
from sklearn.metrics import f1_score, accuracy_score


def print_metrics(model, X, y, pre=""):
    y_pred = model.predict(X)

    scores = {}
    scores["f1_score"] = f1_score(y, y_pred, average="macro")
    scores["accuracy_score"] = accuracy_score(y, y_pred)

    for name, score in scores.items():
        print(f"{pre}{name:10s} - {score:2.4f}")


def get_best(results):
    return sorted(list(results.items()), key=lambda x: x[1][1], reverse=True)[0]


def report(results, X, y):
    for ds_size, (best_model, best_score, best_params) in results.items():
        print(f"For ds of size {ds_size}:")
        print(f"\tTraining score: {best_score:2.4f}")
        print(f"\tTesting score:")
        print_metrics(best_model, X, y, pre="\t\t")

    print()
    best_size, (best_model, _, best_params) = get_best(results)
    print(f"Best model for ds_size = {best_size}")
    print(f"Best params:")
    print(json.dumps(best_params, indent=4))


def save_model(model, filename, dir="models"):
    os.makedirs(dir, exist_ok=True)
    pickle.dump(model, open(os.path.join(dir, filename + ".pkl"), "wb"))


def search_model_2(datasets, estimator, params, **kwargs):
    results = {}
    for name, (X, y) in datasets.items():
        results[name] = search_model_1(X, y, estimator, params, **kwargs)
    return results


def search_model_1(X, y, estimator, params, random_state=200, verbose=0, n_iter=50):
    search = BayesSearchCV(
        estimator=estimator,
        search_spaces=params,
        n_jobs=-1,
        n_iter=n_iter,
        cv=3,
        random_state=random_state,
        verbose=verbose,
    )
    search.fit(X, y)
    return search.best_estimator_, search.best_score_, search.best_params_


# Based on:
# https://www.analyseup.com/learn-python-for-data-science/python-random-forest-feature-importance-plot.html
def plot_feature_importance(importance, model_type):
    # Create arrays from feature importance and feature names
    feature_importance = np.array(importance)

    # Create a DataFrame using a Dictionary
    data = {
        "feature_importance": feature_importance,
        "feature_names": [f"pixel_{i}" for i in range(importance.shape[0])],
    }
    fi_df = pd.DataFrame(data)

    # Sort the DataFrame in order decreasing feature importance
    fi_df.sort_values(by=["feature_importance"], ascending=False, inplace=True)

    plt.figure(figsize=(10, 6))
    sns.barplot(x=fi_df["feature_importance"], y=fi_df["feature_names"])
    plt.title(model_type + "FEATURE IMPORTANCE")
    plt.xlabel("FEATURE IMPORTANCE")
    plt.ylabel("")
    plt.yticks([])
