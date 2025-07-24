import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from modules.model_analysis import ModelBenchmark
import pandas as pd
import joblib

class AutomatedPipeline:
    def __init__(self, data_path="DATA/raw/Classeur1.csv"):
        self.data_path = data_path
        self.results = {}
        
    def run_full_pipeline(self):
        """Pipeline automatisé complet"""
        print("PIPELINE AUTOMATISÉ - CRÉDIT POUR TOUS")
        print("=" * 50)
        
        # 1. Chargement données 
        print("Chargement des données...")
        df_brut = pd.read_csv(self.data_path, sep=";")
        self.results['data_loaded'] = len(df_brut)
        
        # 2. Preprocessing 
        print("Preprocessing...")
        from modules.data_processing import DataProcessing
        processing = DataProcessing(df_brut)
        df_prepared = processing.full_processing()
        self.results['data_processed'] = len(df_prepared)
        
        # 3. Split 
        print("Split train/valid/test...")
        from modules.data_split import DataSplit
        splitter = DataSplit(df_prepared)
        splitter.split()
        splitter.save()
        
        # 4. Benchmark modèles 
        print("Benchmark des modèles...")
        X_train = pd.read_parquet("DATA/X_train.parquet")
        y_train = pd.read_csv("DATA/y_train.csv").squeeze().map({"Fully Paid": 0, "Charged Off": 1})
        X_valid = pd.read_parquet("DATA/X_valid.parquet")
        y_valid = pd.read_csv("DATA/y_valid.csv").squeeze().map({"Fully Paid": 0, "Charged Off": 1})
        
        benchmark = ModelBenchmark(X_train, y_train, X_valid, y_valid)
        benchmark.run()
        self.results['benchmark'] = benchmark.get_results()
        
        # 5. Entraînement final 
        print("Entraînement modèle final...")
        from modules.feature_selection import FEATURES_FINAL
        from lightgbm import LGBMClassifier
        
        model_final = LGBMClassifier(random_state=42, class_weight="balanced", verbose=-1)
        model_final.fit(X_train[FEATURES_FINAL], y_train)
        
        # 6. Évaluation
        X_test = pd.read_parquet("DATA/X_test.parquet")
        y_test = pd.read_csv("DATA/y_test.csv").squeeze().map({"Fully Paid": 0, "Charged Off": 1})
        
        from modules.model_analysis import final_model_evaluation
        test_score = final_model_evaluation(model_final, X_test[FEATURES_FINAL], y_test, threshold=0.4)
        self.results['test_score'] = test_score
        
        # 7. Sauvegarde
        print("Sauvegarde du modèle...")
        from modules.model_analysis import save_model
        save_model(model_final, "DATA/models/final_model.pkl", FEATURES_FINAL)
        
        print("Pipeline terminé avec succès !")
        return self.results

def main():
    pipeline = AutomatedPipeline()
    results = pipeline.run_full_pipeline()
    
    print("RÉSULTATS PIPELINE:")
    print("=" * 30)
    for key, value in results.items():
        print(f"  {key}: {value}")

if __name__ == "__main__":
    main()