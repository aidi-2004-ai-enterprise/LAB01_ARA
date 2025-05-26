import seaborn as sns
from sklearn.model_selection import train_test_split

def load_penguins():

    df = sns.load_dataset("penguins")
    return df.dropna()

def split_dataset(df, test_size=0.2, random_state=42):
    """
    Split the DataFrame into feature matrix X and target y,
    """
    X = df.drop("species", axis=1)
    y = df["species"]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def main():

    df = load_penguins()

    X_train, X_test, y_train, y_test = split_dataset(df)

    print("X_train shape:", X_train.shape)
    print("X_test  shape:", X_test.shape)
    print("y_train shape:", y_train.shape)
    print("y_test  shape:", y_test.shape)

if __name__ == "__main__":
    main()
