import seaborn as sns
from sklearn.model_selection import train_test_split
import xgboost as xgb

def load_penguins():

    df = sns.load_dataset("penguins")
    return df.dropna()

def split_dataset(df, test_size=0.2, random_state=45):

    X = df.drop("species", axis=1)
    y = df["species"]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def build_model():
    """
    Instantiate an XGBoost classifier.
   
    """
    return xgb.XGBClassifier(use_label_encoder=False, eval_metric="mlogloss")

def main():
    df = load_penguins()
    X_train, X_test, y_train, y_test = split_dataset(df)
    model = build_model()
    print("Model created:", model)

if __name__ == "__main__":
    main()
