import pandas as pd
import numpy as np
from modules.data_analysis import DataAnalysis
import os

class DatasetAnalyzer:
    """Analyseur spécialisé pour les datasets traités du chatbot"""
    
    def __init__(self, dataset_path=None):
        self.dataset_path = dataset_path
        self.df = None
        
    def load_latest_dataset(self):
        """Charge automatiquement le dernier dataset traité"""
        data_dir = "DATA"
        if not os.path.exists(data_dir):
            print("Dossier DATA introuvable")
            return False
            
        # Trouver le fichier le plus récent
        files = [f for f in os.listdir(data_dir) if f.startswith('data_') and f.endswith('.csv')]
        if not files:
            print("Aucun dataset traité trouvé")
            return False
            
        latest_file = max(files, key=lambda x: os.path.getctime(os.path.join(data_dir, x)))
        self.dataset_path = os.path.join(data_dir, latest_file)
        
        try:
            self.df = pd.read_csv(self.dataset_path)
            print(f"Dataset chargé: {latest_file}")
            print(f"Dimensions: {self.df.shape}")
            return True
        except Exception as e:
            print(f"Erreur chargement: {e}")
            return False
    
    def analyze_risk_columns(self):
        """Analyse spécifique des colonnes de risque"""
        print("\n" + "="*50)
        print("ANALYSE DES COLONNES DE RISQUE")
        print("="*50)
        
        if self.df is None:
            print("Aucun dataset chargé")
            return
        
        # Vérifier risk_score
        if 'risk_score' in self.df.columns:
            print(" RISK_SCORE:")
            print(f"   Range: {self.df['risk_score'].min():.3f} → {self.df['risk_score'].max():.3f}")
            print(f"   Moyenne: {self.df['risk_score'].mean():.3f}")
            print(f"   Médiane: {self.df['risk_score'].median():.3f}")
            print(f"   Valeurs manquantes: {self.df['risk_score'].isnull().sum()}")
            
            # Distribution par quartiles
            quartiles = self.df['risk_score'].quantile([0.25, 0.5, 0.75])
            print(f"   Q1: {quartiles[0.25]:.3f}")
            print(f"   Q2: {quartiles[0.5]:.3f}")
            print(f"   Q3: {quartiles[0.75]:.3f}")
        else:
            print("Colonne 'risk_score' manquante")
        
        # Vérifier risk_level
        if 'risk_level' in self.df.columns:
            print("RISK_LEVEL:")
            risk_counts = self.df['risk_level'].value_counts()
            total = len(self.df)
            
            for level, count in risk_counts.items():
                pct = (count/total)*100
                print(f"   • {level}: {count:,} clients ({pct:.1f}%)")
            
            # Valeurs uniques
            unique_levels = self.df['risk_level'].unique()
            print(f"   Niveaux détectés: {list(unique_levels)}")
            print(f"   Valeurs manquantes: {self.df['risk_level'].isnull().sum()}")
        else:
            print("Colonne 'risk_level' manquante")

    def analyze_lgbm_features(self):
        """Analyse des 15 features LGBM selon JXP2309"""
        print("\n" + "="*50)
        print("ANALYSE DES FEATURES LGBM JXP2309")
        print("="*50)
        
        # Les 15 features exactes de votre modèle
        lgbm_features = [
            'sub_grade_encoded', 'dti', 'annual_inc', 'int_rate', 'term_encoded', 
            'loan_amnt', 'mort_acc', 'revol_util', 'revol_bal', 'open_acc', 
            'employment_length_category_Unknown', 'credit_history_months', 
            'total_acc', 'installment', 'home_ownership_RENT'
        ]
        
        present_features = []
        missing_features = []
        
        for feature in lgbm_features:
            if feature in self.df.columns:
                present_features.append(feature)
                print(f"✅ {feature}")
                
                # Stats basiques pour les colonnes numériques
                if self.df[feature].dtype in ['int64', 'float64']:
                    print(f"   Range: {self.df[feature].min():.2f} → {self.df[feature].max():.2f}")
                    print(f"   Moyenne: {self.df[feature].mean():.2f}")
                    
                    # Valeurs nulles/problématiques
                    null_count = self.df[feature].isnull().sum()
                    if null_count > 0:
                        print(f"  Valeurs manquantes: {null_count}")
                else:
                    unique_count = self.df[feature].nunique()
                    print(f"   Valeurs uniques: {unique_count}")
            else:
                missing_features.append(feature)
                print(f"{feature} - MANQUANT")
        
        print(f"Résumé features LGBM:")
        print(f"Présentes: {len(present_features)}/15")
        print(f"Manquantes: {len(missing_features)}/15")
        
        if missing_features:
            print(f"Features à créer: {missing_features}")
        
        return present_features, missing_features

    def detect_data_quality_issues(self):
        """Détecte les problèmes de qualité des données"""
        print("\n" + "="*50)
        print("DÉTECTION DES PROBLÈMES DE QUALITÉ")
        print("="*50)
        
        issues_found = []
        
        # 1. Revenus incohérents
        if 'annual_inc' in self.df.columns:
            negative_income = (self.df['annual_inc'] < 0).sum()
            zero_income = (self.df['annual_inc'] == 0).sum()
            very_low_income = (self.df['annual_inc'] < 1000).sum()
            
            if negative_income > 0:
                issues_found.append(f"Revenus négatifs: {negative_income} clients")
            if zero_income > 0:
                issues_found.append(f"Revenus nuls: {zero_income} clients")
            if very_low_income > 0:
                issues_found.append(f"Revenus < 1000€: {very_low_income} clients")
        
        # 2. DTI incohérent
        if 'dti' in self.df.columns:
            negative_dti = (self.df['dti'] < 0).sum()
            extreme_dti = (self.df['dti'] > 100).sum()
            
            if negative_dti > 0:
                issues_found.append(f"DTI négatif: {negative_dti} clients")
            if extreme_dti > 0:
                issues_found.append(f"DTI > 100%: {extreme_dti} clients")
        
        # 3. Scores de risque incohérents
        if 'risk_score' in self.df.columns:
            invalid_scores = ((self.df['risk_score'] < 0) | (self.df['risk_score'] > 1)).sum()
            if invalid_scores > 0:
                issues_found.append(f"Scores de risque hors [0,1]: {invalid_scores} clients")
        
        # Affichage des résultats
        if issues_found:
            print("PROBLÈMES DÉTECTÉS:")
            for issue in issues_found:
                print(f"   • {issue}")
        else:
            print("Aucun problème majeur détecté")
        
        return issues_found
    
    def chatbot_readiness_check(self):
        """Vérifie si le dataset est prêt pour le chatbot"""
        print("\n" + "="*50)
        print("VÉRIFICATION COMPATIBILITÉ CHATBOT")
        print("="*50)
        
        requirements = {
            'risk_score': 'Scores de risque calculés',
            'risk_level': 'Niveaux de risque classifiés',
            'annual_inc': 'Revenus pour analyse individuelle',
            'dti': 'DTI pour analyse individuelle'
        }
        
        readiness_score = 0
        total_requirements = len(requirements)
        
        for col, description in requirements.items():
            if col in self.df.columns:
                valid_data = self.df[col].notna().sum()
                pct_valid = (valid_data / len(self.df)) * 100
                
                if pct_valid >= 95:
                    print(f"{description}: {pct_valid:.1f}% données valides")
                    readiness_score += 1
                else:
                    print(f"{description}: {pct_valid:.1f}% données valides")
            else:
                print(f"{description}: colonne manquante")
        
        final_score = (readiness_score / total_requirements) * 100
        print(f"Score de compatibilité: {final_score:.1f}%")
        
        if final_score >= 75:
            print("Dataset prêt pour le chatbot")
        else:
            print("Dataset nécessite des corrections")
        
        return final_score >= 75
    
    def full_analysis(self):
        """Analyse complète du dataset traité"""
        print("="*60)
        print("ANALYSE COMPLÈTE DU DATASET TRAITÉ")
        print("="*60)
        
        if not self.load_latest_dataset():
            return False
        
        # Analyses spécialisées
        self.analyze_risk_columns()
        present_features, missing_features = self.analyze_lgbm_features()
        issues = self.detect_data_quality_issues()
        is_ready = self.chatbot_readiness_check()
        
        return {
            'dataset_loaded': True,
            'issues_found': issues,
            'chatbot_ready': is_ready,
            'total_clients': len(self.df),
            'file_path': self.dataset_path,
            'present_features': present_features,
            'missing_features': missing_features,
            'columns': list(self.df.columns)
        }

# Script d'utilisation directe
if __name__ == "__main__":
    analyzer = DatasetAnalyzer()
    results = analyzer.full_analysis()
    
    if results:
        print("\n" + "="*60)
        print("📋 RÉSUMÉ DE L'ANALYSE")
        print("="*60)
        print(f"✅ Dataset: {results['file_path']}")
        print(f"Clients: {results['total_clients']:,}")
        print(f"Prêt chatbot: {'Oui' if results['chatbot_ready'] else 'Non'}")
        print(f"Problèmes: {len(results['issues_found'])}")
        print(f"Features présentes: {len(results['present_features'])}/15")