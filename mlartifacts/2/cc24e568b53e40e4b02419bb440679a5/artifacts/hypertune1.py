from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
import pandas as pd
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name = 'target')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestClassifier(random_state=42)

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}

grid_search_cv = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)

#grid_searh_cv.fit(X_train, y_train)
#print("Best Hyperparameters: ", grid_searh_cv.best_params_)

mlflow.set_experiment("Breast Cancer Hyperparameter Tuning")

with mlflow.start_run() as parent:
    grid_search_cv.fit(X_train, y_train)

    # log all the child runs
    for i in range(len(grid_search_cv.cv_results_['params'])):

        with mlflow.start_run(nested=True) as child:
            mlflow.log_params(grid_search_cv.cv_results_["params"][i])
            mlflow.log_metric("accuracy", grid_search_cv.cv_results_["mean_test_score"][i])

    # Displaying the best parameters and the best score
    best_params = grid_search_cv.best_params_
    best_score = grid_search_cv.best_score_

    # Log params
    mlflow.log_params(best_params)

    # Log metrics
    mlflow.log_metric("accuracy", best_score)

    # Log training data
    train_df = X_train.copy()
    train_df['target'] = y_train

    train_df = mlflow.data.from_pandas(train_df)
    mlflow.log_input(train_df, "training")

    # Log test data
    test_df = X_test.copy()
    test_df['target'] = y_test

    test_df = mlflow.data.from_pandas(test_df)
    mlflow.log_input(test_df, "testing")

    # Log source code
    mlflow.log_artifact(__file__)

    # Log the best model
    mlflow.sklearn.log_model(grid_search_cv.best_estimator_, "random_forest")

    # Set tags
    mlflow.set_tag("author", "Aryan Malik")
