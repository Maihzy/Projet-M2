import pandas as pd
from sklearn.model_selection import train_test_split

class DataSplit:
    
    def __init__(self, df, target_col="loan_status", random_state=42):
        """
        Initialisation avec le dataset complet et les paramètres.
        """
        self.df = df.copy()
        self.target_col = target_col
        self.random_state = random_state
        self.X_train = None
        self.X_valid = None
        self.X_test = None
        self.y_train = None
        self.y_valid = None
        self.y_test = None

    def split(self, test_size=0.3):
        """
        Split stratifié du dataset en train, validation et test.
        """
        X = self.df.drop(columns=[self.target_col])
        y = self.df[self.target_col]

        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=self.random_state
        )

        X_valid, X_test, y_valid, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=self.random_state
        )

        self.X_train, self.X_valid, self.X_test = X_train, X_valid, X_test
        self.y_train, self.y_valid, self.y_test = y_train, y_valid, y_test

        print("\nRépartition des défauts par ensemble :")
        for name, target in zip(["Train", "Validation", "Test"], [y_train, y_valid, y_test]):
            pct_defaut = round(100 * sum(target == "Charged Off") / len(target), 2)
            print(f"{name} - Total : {len(target)} - % défauts : {pct_defaut}%")

    def save(self, save_path="DATA"):
        """
        Sauvegarde des ensembles dans des fichiers parquet / CSV.
        """
        if self.X_train is None:
            raise ValueError("Les ensembles n'ont pas encore été créés. Exécutez la méthode split() d'abord.")

        self.X_train.to_parquet(f"{save_path}/X_train.parquet", index=False)
        self.X_valid.to_parquet(f"{save_path}/X_valid.parquet", index=False)
        self.X_test.to_parquet(f"{save_path}/X_test.parquet", index=False)
        self.y_train.to_csv(f"{save_path}/y_train.csv", index=False)
        self.y_valid.to_csv(f"{save_path}/y_valid.csv", index=False)
        self.y_test.to_csv(f"{save_path}/y_test.csv", index=False)

        print(f"\nJeux de données sauvegardés dans le dossier '{save_path}'.")
