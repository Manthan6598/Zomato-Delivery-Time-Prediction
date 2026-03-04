import pandas as pd
import joblib
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression , Ridge , Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error,root_mean_squared_error,mean_squared_error,r2_score
from src.features.preprocess import split_df,preprocessor_pipeline
import mlflow
import mlflow.sklearn
from src.utils.logger import logger


def evaluate_model(name, pipeline, X_train, y_train,X_test, y_test, params= None):

    with mlflow.start_run(run_name=name):
        if params:
            mlflow.log_params(params)

        cv_scores = cross_val_score(
            pipeline,
            X_train,
            y_train,
            cv = 5,
            scoring = "neg_root_mean_squared_error",
            n_jobs=-1
        )

        mean_cv_rmse = float(-np.mean(cv_scores))

        pipeline.fit(X_train, y_train)

        train_predictions = pipeline.predict(X_train)

        train_rmse = mean_squared_error(y_train, train_predictions, squared=False)

        

        predictions = pipeline.predict(X_test)

        mae = float(mean_absolute_error(y_test,predictions))
        mse = float(mean_squared_error(y_test,predictions))
        rmse = float(root_mean_squared_error(y_test,predictions))
        r2 = float(r2_score(y_test,predictions))

        mlflow.log_metric("train_rmse", train_rmse)
        mlflow.log_metric("cv_rmse", mean_cv_rmse)
        mlflow.log_metric("test_rmse", rmse)
        mlflow.log_metric("test_r2", r2)
        mlflow.log_metric("test_mae", mae)

        mlflow.sklearn.log_model(pipeline, "model")


        print(f"\n{name}")
        print("Train RMSE:", round(train_rmse, 4))
        print("CV RMSE: ", round(mean_cv_rmse,4))
        print("Test MAE: ", round(mae,4))
        print("Test MSE: ", round(mse,4))
        print("Test RMSE: ", round(rmse,4))
        print("Test R2 Score: ", round(r2,4))

        return mean_cv_rmse, pipeline



def train():

    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("Zomato_Delivery_Time_Experiment")

    logger.info("Training of Dataset has started....")

    df = pd.read_csv("data/processed/processed_data.csv")

    logger.info("Splitting of Training Dataset has started....")

    X_train,X_test,y_train,y_test = split_df(df)

    

    results = []
    logger.info("Linear Regression Model Started....")
    baseline_pipeline = Pipeline([
        ("preprocessor",preprocessor_pipeline()),
        ("regressor",LinearRegression())
    ])

    baseline_rmse, baseline_model = evaluate_model(
        "LinearRegression (Baseline)",
        baseline_pipeline,
        X_train,
        y_train,
        X_test,
        y_test
    )

    results.append(("Linear Regression",baseline_rmse,baseline_model))
    logger.info("Linear Regression Model Completed....")


    logger.info("L2 Regression Model Started....")

    ridge_alphas = [0.001,0.01,0.1,1,10,100]
    best_ridge_rmse = float("inf")
    best_ridge_model = None
    best_ridge_alpha = None

    for alpha in ridge_alphas:

        ridge_pipeline = Pipeline([
            ("preprocessor",preprocessor_pipeline()),
            ("regressor",Ridge(alpha = alpha))
        ])

        cv_rmse, model = evaluate_model(
            f"Ridge (alpha = {alpha})",
            ridge_pipeline,
            X_train,
            y_train,
            X_test,
            y_test
        )

        if cv_rmse < best_ridge_rmse:
            best_ridge_rmse = cv_rmse
            best_ridge_model = model
            best_ridge_alpha = alpha

    print(f"\n Best Ridge ALpha: {best_ridge_alpha}")
    results.append(("Ridge", best_ridge_rmse,best_ridge_model))
    logger.info("L2 Regression Model Completed....")

    logger.info("L1 Regression Model Started....")

    lasso_alphas = [0.0001, 0.001,0.01,0.1,1]
    best_lasso_rmse = float("inf")
    best_lasso_model = None
    best_lasso_alpha = None

    for alpha in lasso_alphas:
        lasso_pipeline = Pipeline([
            ("preprocessor",preprocessor_pipeline()),
            ("regressor", Lasso(alpha=alpha))
        ])

        cv_rmse, model = evaluate_model(
            f"Lasso (alpha={alpha})",
            lasso_pipeline,
            X_train,
            y_train,
            X_test,
            y_test
        )

        if cv_rmse < best_lasso_rmse:
            best_lasso_rmse = cv_rmse
            best_lasso_alpha = alpha
            best_lasso_model = model 

    print(f"\n Best Lasso Alpha : {best_lasso_alpha}")
    results.append(("Lasso", best_lasso_rmse,best_lasso_model))
    logger.info("L1 Regression Model Completed....")


    logger.info("Decision Tree Regression Model Started....")
    for depth in [5,10,15,20]:
        for min_split in [5,10,15,20]:
            dt_pipeline = Pipeline([
                ("preprocessor", preprocessor_pipeline()),
                ("regressor", DecisionTreeRegressor(
                    max_depth = depth,
                    min_samples_split = min_split,
                    random_state = 58 
                ))
            ])

            name = f"DecisionTree depth={depth} split={min_split}"

            cv_rmse,model = evaluate_model(
                name,
                dt_pipeline,
                X_train,
                y_train,
                X_test,
                y_test
            )

            results.append((name,cv_rmse,model))


    logger.info("Decision Tree Regression Model Completed....")


    logger.info("Random Forest Regression Model Started....")


    for n_est in [100,200]:
        for depth in [10,20]:
            rf_pipeline = Pipeline([
                ("preprocessor",preprocessor_pipeline()),
                ("regressor", RandomForestRegressor(
                    n_estimators=n_est,
                    max_depth= depth,
                    min_samples_split=10,
                    n_jobs=-1,
                    random_state=59
                ))
            ])

            name = f"RandomForest n_estimator={n_est} depth={depth}"

            cv_rmse,model = evaluate_model(
                name,
                rf_pipeline,
                X_train,
                y_train,
                X_test,
                y_test
            )

            results.append((name,cv_rmse,model))


    logger.info("Random Forest Regression Model Completed....")

    logger.info("XGBoost Regression Model Started....")

    for lr in [0.05, 0.1]:
        for depth in [4,6]:
            xgb_pipeline = Pipeline([
                ("preprocessor", preprocessor_pipeline()),
                ("regressor",XGBRegressor(
                    n_estimators = 300,
                    learning_rate = lr,
                    max_depth = depth,
                    subsample = 0.8,
                    colsample_bytree = 0.8,
                    random_state = 42,
                    verbosity = 0
                ))
            ])

            name = f"XGBosst lr= {lr} depth = {depth}"

            cv_rmse, model = evaluate_model(
                name,
                xgb_pipeline,
                X_train,
                y_train,
                X_test,
                y_test
            )

            results.append((name,cv_rmse,model))

    logger.info("XGBoost Regression Model Completed....")


    logger.info("Selection of Best Regression Model In Progress....")

    best_model_name , best_rmse, best_model = min(results, key=lambda x: x[1])

    print("\n==============================")
    print("FINAL BEST MODEL:", best_model_name)
    print("BEST CV RMSE:", best_rmse)
    print("==============================")

    logger.info("Selection of Best Regression Model Completed....")

    # ---------------- Log Best Model Separately ----------------
    with mlflow.start_run(run_name="Best_Model"):

        mlflow.log_param("best_model_name", best_model_name)
        mlflow.log_metric("best_cv_rmse", float(best_rmse))

        # Log model artifact
        mlflow.sklearn.log_model(best_model, "best_model")

        run_id = mlflow.active_run().info.run_id

    # ---------------- Register Model ----------------
    model_uri = f"runs:/{run_id}/best_model"

    registered_model = mlflow.register_model(
        model_uri=model_uri,
        name="ZomatoDeliveryModel"
    )

    print("\nModel Registered Successfully.")

    # ---------------- Save Locally As Backup ----------------
    joblib.dump(
        best_model,
        "models/delivery_time_model.pkl"
    )

    logger.info("Model Pkl File Exported Successfully!")
    logger.info("Training Pipeline Completed....")

if __name__ == "__main__":
    train()
