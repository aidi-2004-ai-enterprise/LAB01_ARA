import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import xgboost as xgb

def load_penguins():


    df = sns.load_dataset("penguins")
    return df.dropna()

def split_dataset(df, test_size=0.2, random_state=45):

    X = df.drop(["species", "island", "sex"], axis=1)
    y = df["species"]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def build_model():

    return xgb.XGBClassifier(use_label_encoder=False, eval_metric="mlogloss")

def fit_and_evaluate(model, X_train, X_test, y_train, y_test):

    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc  = le.transform(y_test)

    model.fit(X_train, y_train_enc)
    preds_enc = model.predict(X_test)

    acc = accuracy_score(y_test_enc, preds_enc)
    print(f"Test accuracy: {acc:.4f}")

def tune_hyperparameters(model, X_train, y_train):

    param_grid = {
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.1, 0.2]
    }
    grid = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=3,
        scoring="accuracy",
        n_jobs=-1
    )
    grid.fit(X_train, y_train)
    print("Best params:", grid.best_params_)
    print(f"Best CV score: {grid.best_score_:.4f}")

def main():
    #  Load & clean data
    df = load_penguins()

    # Split into train/test
    X_train, X_test, y_train, y_test = split_dataset(df)

    #  Build & evaluate
    model = build_model()
    fit_and_evaluate(model, X_train, X_test, y_train, y_test)

    # Hyperparameter tuning
    print("\nStarting hyperparameter tuning...")
    tune_hyperparameters(model, X_train, y_train)

if __name__ == "__main__":
    main()
