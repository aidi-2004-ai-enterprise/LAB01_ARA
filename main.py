import seaborn as sns
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

def load_penguins():

    df = sns.load_dataset("penguins")
    return df.dropna()

def split_data(df, test_size=0.2, random_state=45):
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
    print("Test accuracy:", acc)

def main():
    df = load_penguins()
    X_train, X_test, y_train, y_test = split_data(df)
    model = build_model()
    fit_and_evaluate(model, X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    main()

    main()
