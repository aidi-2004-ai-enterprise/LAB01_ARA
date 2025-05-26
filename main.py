import seaborn as sns

def load_penguins():

    df = sns.load_dataset("penguins")
    print(df.head())

if __name__ == "__main__":
    main()
