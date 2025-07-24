import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

class DataProcessing:

    def __init__(self, df):
        """
        Initialisation avec un DataFrame déjà chargé.
        """
        self.df = df.copy()
        
    def handle_missing_values(self):
        """
        Gestion des valeurs manquantes :
        - Imputation par 'Unknown' pour les catégoriques simples.
        - Imputation par médiane pour les variables numériques concernées.
        """
        cat_cols = ['emp_length', 'home_ownership', 'verification_status']
        num_cols = ['mort_acc', 'pub_rec_bankruptcies', 'revol_util']

        for col in cat_cols:
            if col in self.df.columns:
                self.df[col] = self.df[col].fillna("Unknown")

        for col in num_cols:
            if col in self.df.columns:
                median_val = self.df[col].median()
                self.df[col] = self.df[col].fillna(median_val)

        print("Missing values handled.")


    def treat_outliers(self):
        """
        Encadrement des valeurs extrêmes sur les variables sensibles.
        Méthode : Winsorisation aux 1er et 99e percentiles.
        """
        cols_to_winsorize = ['loan_amnt', 'dti', 'annual_inc', 'revol_util', 'revol_bal']

        for col in cols_to_winsorize:
            if col in self.df.columns:
                q1 = self.df[col].quantile(0.01)
                q99 = self.df[col].quantile(0.99)
                self.df[col] = np.clip(self.df[col], q1, q99)

        print("Outliers treated.")

    def transform_dates(self):
        """
        Transformation des dates brutes en variables exploitables :
        - Credit history length (in months).
        - Seasonality based on loan issue date.
        """
        if 'issue_d' in self.df.columns and 'earliest_cr_line' in self.df.columns:
            self.df['issue_d'] = pd.to_datetime(self.df['issue_d'], format='%d-%m-%y', errors='coerce')
            self.df['earliest_cr_line'] = pd.to_datetime(self.df['earliest_cr_line'], format='%d-%m-%y', errors='coerce')

            self.df['credit_history_months'] = (
                (self.df['issue_d'] - self.df['earliest_cr_line']).dt.days // 30
            ).clip(lower=0)

            self.df['loan_season'] = self.df['issue_d'].dt.month % 12 // 3 + 1
            season_map = {1: 'Winter', 2: 'Spring', 3: 'Summer', 4: 'Autumn'}
            self.df['loan_season'] = self.df['loan_season'].map(season_map)

            print("Date-derived variables created.")
        else:
            print("Date columns missing or not exploitable.")

    def create_new_variables(self):
        """
        Création de nouvelles variables dérivées.
        """
        if 'dti' in self.df.columns and 'annual_inc' in self.df.columns:
            self.df['estimated_debt_amount'] = (self.df['dti'] * self.df['annual_inc']) / 100

        if 'revol_util' in self.df.columns and 'annual_inc' in self.df.columns:
            self.df['revolving_credit_income_ratio'] = (self.df['revol_util'] * self.df['annual_inc']) / 100

        if 'emp_length' in self.df.columns:
            def classify_employment_length(x):
                if x in ['< 1 year', '1 year', '2 years', '3 years']:
                    return "Junior"
                elif x in ['4 years', '5 years', '6 years', '7 years']:
                    return "Intermediate"
                elif x in ['8 years', '9 years', '10+ years']:
                    return "Senior"
                else:
                    return "Unknown"

            self.df['employment_length_category'] = self.df['emp_length'].apply(classify_employment_length)

        if 'sub_grade' in self.df.columns:
            letter_mapping = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7}
            self.df['sub_grade_encoded'] = self.df['sub_grade'].apply(
                lambda x: int(x[1]) + (letter_mapping[x[0]] - 1) * 5
            )

        print("New variables created.")

    def encode_categoricals(self):
        """
        Encodage des variables catégoriques :
        - Ordinal encoding pour 'term' et 'grade'
        - One-hot encoding pour les autres catégoriques
        """
        if 'term' in self.df.columns:
            term_mapping = {' 36 months': 0, ' 60 months': 1}
            self.df['term_encoded'] = self.df['term'].map(term_mapping)

        if 'grade' in self.df.columns:
            grade_mapping = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7}
            self.df['grade_encoded'] = self.df['grade'].map(grade_mapping)

        one_hot_cols = [
            'home_ownership',
            'verification_status',
            'purpose',
            'initial_list_status',
            'application_type',
            'loan_season',
            'employment_length_category'
        ]

        self.df = pd.get_dummies(self.df, columns=one_hot_cols, drop_first=True)

        print("Categorical variables encoded.")

    def drop_useless_columns(self):
        """
        Suppression des variables inutiles ou redondantes, après transformation.
        """
        to_drop = [
            'title', 'emp_title', 'issue_d', 'earliest_cr_line', 'term',
            'emp_length', 'grade', 'sub_grade'
        ]
        self.df.drop(columns=[col for col in to_drop if col in self.df.columns], inplace=True)
        print("Useless columns dropped.")

    def standardize_numerical(self):
        """
        Standardisation des variables numériques pour harmoniser les échelles.
        """
        num_cols = ['loan_amnt', 'dti', 'annual_inc', 'revol_util', 'revol_bal',
                    'estimated_debt_amount']

        scaler = StandardScaler()
        for col in num_cols:
            if col in self.df.columns:
                self.df[col] = scaler.fit_transform(self.df[[col]])

        print("Numerical variables standardized.")

    def full_processing(self):
        """Processing complet AVEC sauvegarde des valeurs originales"""
        
        # ÉTAPE 1-6 : Vos traitements existants
        self.handle_missing_values()
        self.treat_outliers()
        self.transform_dates()
        self.create_new_variables()
        self.encode_categoricals()
        self.drop_useless_columns()
        
        # SAUVEGARDE DES VALEURS LISIBLES AVANT STANDARDISATION
        chatbot_columns = ['dti', 'annual_inc', 'loan_amnt', 'int_rate', 'revol_util', 
                          'installment', 'revol_bal', 'open_acc', 'total_acc', 'mort_acc']
        
        # Créer des versions "_original" pour le chatbot
        for col in chatbot_columns:
            if col in self.df.columns:
                self.df[f'{col}_original'] = self.df[col].copy()
        
        # ÉTAPE 7 : Standardisation (pour ML seulement)
        self.standardize_numerical()
        
        # À la fin du pipeline, ajoute le scoring et la classification
        if hasattr(self, "df"):
            # Prédire le score de défaut pour chaque ligne
            from modules.tools import predict_risk
            self.df["risk_score"] = self.df.apply(lambda row: predict_risk(row.to_dict())[0], axis=1)
            # Créer la colonne risk_level selon le seuil métier
            self.df["risk_level"] = self.df["risk_score"].apply(
                lambda x: "Risque Faible" if x < 0.4 else ("Risque Modéré" if x < 0.7 else "Risque Élevé")
            )
        
        print("Data preparation completed.")
        return self.df
