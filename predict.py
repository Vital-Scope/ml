import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import warnings
warnings.filterwarnings('ignore')

from create_dataset import extract_features_from_csv, detect_signal_type, process_patient_folder

def analyze_folder(folder_path, model_path='best_model.joblib'):
    model_data = joblib.load(model_path)
    patient_folder = Path(folder_path)
    features = process_patient_folder(patient_folder, patient_folder.name, None) 
    df_single = pd.DataFrame([features])
    non_feature_cols = ['patient_id', 'folder_path', 'csv_files_count', 'target']
    features_to_drop = [col for col in non_feature_cols if col in df_single.columns]
    features_df = df_single.drop(features_to_drop, axis=1)
    all_features = model_data['all_features']
    for feature in all_features:
        if feature not in features_df.columns:
            features_df[feature] = 0.0
    
    features_df = features_df[all_features]
    imputer = model_data['imputer']
    scaler = model_data['scaler']
    features_imputed = imputer.transform(features_df)
    features_scaled = scaler.transform(features_imputed)
    selected_features = model_data['selected_features']
    features_processed = pd.DataFrame(features_scaled, columns=all_features)
    features_final = features_processed[selected_features]
    
    # Предсказание
    model = model_data['model']
    probability = model.predict_proba(features_final)[0, 1]
    return probability

def main():    
    while True:
        folder_path = input("Введите путь к папке пациента или quit - для выхода: ").strip()
        
        if folder_path.lower() == 'quit':
            break
        
        if not Path(folder_path).exists():
            print("Папка не существует")
            continue
        
        probability = analyze_folder(folder_path)
        
        if probability is not None:
            print(f"\nРезультат:")
            print(f"Вероятность отклонения: {probability*100:.1f} %")
            
            if probability < 0.3:
                print("Отклонений вероятно нет")
            elif probability < 0.7:
                print("Могут быть риски гипоксии") 
            else:
                print("Высокая вероятность гипоксии")
        else:
            print("Не удалось проанализировать")

if __name__ == "__main__":
    main()
