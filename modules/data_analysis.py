import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from scipy.stats import chi2_contingency

warnings.filterwarnings('ignore')

class DataAnalysis:

    def __init__(self, dataframe):
        """
        Initialise l'analyse avec un DataFrame déjà chargé.
        """
        self.data = dataframe

    def overview(self):
        print("\nAperçu des premières lignes :")
        print(self.data.head())

        print("\nInfos générales :")
        print(self.data.info())

        print("\nStatistiques descriptives :")
        print(self.data.describe(include='all'))

        duplicates = self.data.duplicated().sum()
        print(f"\nNombre de doublons dans le dataset : {duplicates}")

        missings = self.data.isnull().sum()
        missing_cols = missings[missings > 0].index

        if not missing_cols.empty:
            total = len(self.data)
            pct_missing = (missings / total * 100).round(2)

            print("\n=== Colonnes avec valeurs manquantes ===")
            print(missings[missing_cols])
            print("\n=== % de valeurs manquantes ===")
            print(pct_missing[missing_cols])
            print("\n=== Types des colonnes avec valeurs manquantes ===")
            print(self.data.dtypes[missing_cols])
        else:
            print("\nAucune colonne avec valeurs manquantes détectée.")

    def analyze_numerical(self):
        print("\nAnalyse des variables numériques :")
        numerical_features = self.data.select_dtypes(include=np.number).columns.tolist()
        print("\nVariables numériques :", numerical_features)

        if not numerical_features:
            print("\nAucune variable numérique détectée.")
            return

        print("\n=== Moyennes ===")
        print(self.data[numerical_features].mean())

        print("\n=== Médianes ===")
        print(self.data[numerical_features].median())

        print("\n=== Quartiles ===")
        print(self.data[numerical_features].quantile([0.25, 0.5, 0.75]))

        print("\n=== Minimum et Maximum ===")
        print(self.data[numerical_features].min())
        print(self.data[numerical_features].max())

        print("\n=== Variance ===")
        print(self.data[numerical_features].var())

        print("\n=== Écart-type ===")
        print(self.data[numerical_features].std())

    def correlation_matrix(self):
        plt.figure(figsize=(12, 10))
        sns.heatmap(self.data.corr(numeric_only=True), annot=True, cmap='coolwarm')
        plt.title('Matrice de corrélation complète')
        plt.show()

    def filtered_correlation_matrix(self, threshold=0.3):
        corr_matrix = self.data.corr(numeric_only=True)
        strong_corr = corr_matrix[(corr_matrix.abs() > threshold) & (corr_matrix.abs() < 1)]

        plt.figure(figsize=(12, 10))
        sns.heatmap(strong_corr, annot=True, cmap='coolwarm', mask=strong_corr.isnull())
        plt.title(f'Matrice de corrélation (|corr| > {threshold})')
        plt.show()

    def loan_status_distribution(self):
        print("\nRépartition de loan_status :")
        print(self.data['loan_status'].value_counts())

        plt.figure()
        sns.countplot(x='loan_status', data=self.data)
        plt.title('Répartition des statuts de prêt')
        plt.xticks(rotation=45)
        plt.show()

    def relation_with_loan_status(self):
        categorical_features = self.data.select_dtypes(include='object').columns.tolist()
        numerical_features = self.data.select_dtypes(include=np.number).columns.tolist()

        for col in categorical_features:
            if col != 'loan_status':
                plt.figure()
                sns.countplot(x=col, hue='loan_status', data=self.data, order=self.data[col].value_counts().index)
                plt.title(f'{col} en fonction de loan_status')
                plt.xticks(rotation=45)
                plt.show()

        for col in numerical_features:
            plt.figure()
            sns.boxplot(x='loan_status', y=col, data=self.data)
            plt.title(f'{col} en fonction de loan_status')
            plt.show()

    def relation_with_loan_status_selected(self, columns):
        for col in columns:
            if col not in self.data.columns:
                print(f"\nLa colonne '{col}' n'existe pas.")
                continue

            if self.data[col].dtype == 'object':
                plt.figure()
                sns.countplot(x=col, hue='loan_status', data=self.data, order=self.data[col].value_counts().index)
                plt.title(f'{col} en fonction de loan_status')
                plt.xticks(rotation=45)
                plt.show()
            else:
                plt.figure()
                sns.boxplot(x='loan_status', y=col, data=self.data)
                plt.title(f'{col} en fonction de loan_status')
                plt.show()

    def chi2_categorical_vs_target(self, target_col='loan_status'):
        cat_cols = self.data.select_dtypes(include='object').columns

        print("\n=== Test du Chi2 entre les variables catégoriques et la cible ===")

        for col in cat_cols:
            if col != target_col:
                contingency_table = pd.crosstab(self.data[col], self.data[target_col])
                chi2, p, _, _ = chi2_contingency(contingency_table)
                print(f"{col} : p-value = {p:.4f}")

                if p < 0.05:
                    print("➡ Association significative détectée.\n")
                else:
                    print("➡ Pas d'association significative.\n")

    def qualitative_summary(self):
        print("--- Sélection et affichage des variables qualitatives ---")
        qualitative_vars = self.data.select_dtypes(include='object').columns

        if not qualitative_vars.empty:
            print(f"\nVariables qualitatives : {list(qualitative_vars)}\n")
            print("-" * 50)

            for col in qualitative_vars:
                print(f"\n**Variable : '{col}'**")
                print(f"Nombre de modalités : {self.data[col].nunique()}")
                print("Répartition :")
                if self.data[col].nunique() > 10:
                    print(self.data[col].value_counts().head(10))
                    print(f"... et {self.data[col].nunique() - 10} autres valeurs.")
                else:
                    print(self.data[col].value_counts())
                print("-" * 50)
        else:
            print("\nAucune variable qualitative détectée.")

    def unique_values_report(self, max_display=20):
        cat_cols = self.data.select_dtypes(include=["object", "category"]).columns

        if len(cat_cols) == 0:
            print("\nAucune colonne catégorielle détectée.")
            return

        print("\n=== Valeurs uniques par colonne catégorielle ===")
        for col in cat_cols:
            uniques = self.data[col].dropna().unique()
            n_uniques = len(uniques)
            print(f"\n• {col} ({n_uniques} modalités) ")

            if n_uniques <= max_display:
                print(list(uniques))
            else:
                print(list(uniques[:max_display]) + [f"... (+{n_uniques - max_display} autres)"])

    def target_balance_check(self, target_col='loan_status'):
        if target_col not in self.data.columns:
            print(f"\nLa colonne '{target_col}' n'existe pas.")
            return

        counts = self.data[target_col].value_counts()
        percentages = (counts / len(self.data) * 100).round(2)

        print(f"\n=== Répartition des classes pour '{target_col}' ===")
        print(counts)
        print("\nPourcentages :")
        print(percentages)

        if percentages.max() > 70:
            print("\n⚠️ Déséquilibre des classes détecté.")
        else:
            print("\nAucun déséquilibre majeur détecté.")

    def detect_outliers(self, columns):
        print("\n=== Détection des valeurs aberrantes ===")
        for col in columns:
            if col in self.data.columns:
                Q1 = self.data[col].quantile(0.25)
                Q3 = self.data[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = self.data[(self.data[col] < (Q1 - 1.5 * IQR)) | (self.data[col] > (Q3 + 1.5 * IQR))]
                print(f"{col} : {len(outliers)} valeurs aberrantes détectées")
            else:
                print(f"Colonne {col} introuvable dans le dataset.")

    def missing_values_by_class(self, target_col='loan_status'):
        print(f"\n=== Valeurs manquantes par classe de {target_col} ===")
        for col in self.data.columns:
            if self.data[col].isnull().sum() > 0:
                print(f"\nColonne : {col} ({self.data[col].isnull().sum()} valeurs manquantes)")
                subset = self.data.loc[self.data[col].isnull(), target_col]
                print(subset.value_counts(dropna=False))

    def plot_numeric_distributions(self, columns):
        print("\n=== Distributions numériques ===")
        for col in columns:
            if col in self.data.columns:
                plt.figure()
                sns.histplot(data=self.data, x=col, kde=True)
                plt.title(f'Distribution de {col}')
                plt.show()
            else:
                print(f"Colonne {col} introuvable dans le dataset.")

    def full_eda(self):
        print("\n=== Début de l'analyse exploratoire ===")

        self.overview()
        self.analyze_numerical()
        self.qualitative_summary()
        self.unique_values_report()
        self.correlation_matrix()
        self.filtered_correlation_matrix()
        self.loan_status_distribution()
        self.target_balance_check()
        self.chi2_categorical_vs_target()

        target_columns = ['loan_amnt', 'int_rate', 'dti', 'annual_inc', 'revol_util',
                          'term', 'grade', 'home_ownership', 'verification_status']
        self.relation_with_loan_status_selected(target_columns)

        colonnes_a_verifier = ['loan_amnt', 'dti', 'annual_inc', 'revol_util']
        self.detect_outliers(colonnes_a_verifier)

        self.missing_values_by_class()

        colonnes_numeriques = ['loan_amnt', 'dti', 'annual_inc', 'revol_util']
        self.plot_numeric_distributions(colonnes_numeriques)

        variables_historique = ['open_acc', 'total_acc', 'pub_rec', 'pub_rec_bankruptcies']
        self.plot_numeric_distributions(variables_historique)
        self.relation_with_loan_status_selected(variables_historique)

        print("\n=== Fin de l'analyse exploratoire ===")
