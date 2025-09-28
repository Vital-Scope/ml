import pandas as pd
import numpy as np
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def analyze_annotations():
    try:
        # Читаем аннотации
        df_normal_anno = pd.read_excel('regular.xlsx')
        df_not_normal_anno = pd.read_excel('hypoxia.xlsx')
        print("Нормальные случаи - колонки:", df_normal_anno.columns.tolist())
        print("Количество записей normal:", len(df_normal_anno))
        print("\nПатологии - колонки:", df_not_normal_anno.columns.tolist())
        print("Количество записей not_normal:", len(df_not_normal_anno))
        print("\nПервые 3 строки normal.xlsx:")
        print(df_normal_anno.head(3))
        print("\nПервые 3 строки not_normal.xlsx:")
        print(df_not_normal_anno.head(3))
        
        return df_normal_anno, df_not_normal_anno
        
    except Exception as e:
        print(f"Ошибка при чтении аннотаций: {e}")
        return None, None

def explore_folder_structure(base_path):
    base_dir = Path(base_path)
    normal_path = base_dir / 'regular'
    not_normal_path = base_dir / 'hypoxia'
    
    print(f"Папка normal существует: {normal_path.exists()}")
    print(f"Папка not_normal существует: {not_normal_path.exists()}")
    
    if normal_path.exists():
        normal_folders = [f.name for f in normal_path.iterdir() if f.is_dir()]
        print(f"Папки в normal ({len(normal_folders)}): {normal_folders[:5]}...")  
        if normal_folders:
            first_folder = normal_path / normal_folders[0]
            print(f"\nСодержимое первой папки ({normal_folders[0]}):")
            for item in first_folder.iterdir():
                print(f"  {item.name}")
                if item.is_dir():
                    for subitem in item.iterdir():
                        print(f"    └── {subitem.name}")
    
    if not_normal_path.exists():
        not_normal_folders = [f.name for f in not_normal_path.iterdir() if f.is_dir()]
        print(f"\nПапки в not_normal ({len(not_normal_folders)}): {not_normal_folders[:5]}...")
    
    return normal_path, not_normal_path

def analyze_csv_structure(csv_file):
    try:
        df = pd.read_csv(csv_file, nrows=5)  
        print(f"Файл: {csv_file.name}")
        print(f"Колонки: {df.columns.tolist()}")
        print(f"Размер: {df.shape}")
        print(f"Первые значения:\n{df.head()}")
        print("-" * 50)
        return df.columns.tolist()
    except Exception as e:
        print(f"Ошибка чтения {csv_file}: {e}")
        return None

def extract_features_from_csv(csv_file, signal_type):
    try:
        df = pd.read_csv(csv_file)
        time_col = df.columns[0]
        value_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]
        
        values = df[value_col].dropna()
        
        if len(values) < 10:  # слишком мало данных
            return None
            
        # Базовые статистические признаки
        features = {
            f'{signal_type}_mean': np.mean(values),
            f'{signal_type}_std': np.std(values),
            f'{signal_type}_min': np.min(values),
            f'{signal_type}_max': np.max(values),
            f'{signal_type}_median': np.median(values),
            f'{signal_type}_q25': np.percentile(values, 25),
            f'{signal_type}_q75': np.percentile(values, 75),
            f'{signal_type}_skew': pd.Series(values).skew(),
            f'{signal_type}_kurtosis': pd.Series(values).kurtosis(),
            f'{signal_type}_count': len(values),
        }
        
        # Признаки вариабельности 
        if len(values) > 1:
            diff_values = np.diff(values)
            features.update({
                f'{signal_type}_rmssd': np.sqrt(np.mean(diff_values ** 2)),
                f'{signal_type}_nn50': np.sum(np.abs(diff_values) > 50),
                f'{signal_type}_cv': np.std(values) / np.mean(values) if np.mean(values) != 0 else 0,  # коэффициент вариации
            })
        
        return features
    except Exception as e:
        print(f"Ошибка обработки {csv_file}: {e}")
        return None

def detect_signal_type(folder_name):
    folder_lower = folder_name.lower()
    
    if any(word in folder_lower for word in ['bpm']):
        return 'bpm'
    elif any(word in folder_lower for word in ['uterus']):
        return 'uterus'
    else:
        return 'other'

def process_patient_folder(patient_folder, patient_id, target_class):
    features = {
        'patient_id': patient_id,
        'target': target_class,
        'folder_path': str(patient_folder)
    }
    
    csv_files_found = 0
    for csv_file in patient_folder.rglob('*.csv'):
        try:
            # Определяем тип сигнала по структуре папок
            signal_type = 'unknown'
            for parent in csv_file.parents:
                if parent != patient_folder:
                    signal_type = detect_signal_type(parent.name)
                    break
            
            # Извлекаем признаки
            signal_features = extract_features_from_csv(csv_file, signal_type)
            
            if signal_features:
                features.update(signal_features)
                csv_files_found += 1
                print(f"  Обработан: {csv_file.relative_to(patient_folder)}")
                
        except Exception as e:
            print(f"Ошибка обработки {csv_file}: {e}")
            continue
    
    features['csv_files_count'] = csv_files_found
    return features if csv_files_found > 0 else None

def create_complete_dataset(base_path):
    normal_path, not_normal_path = explore_folder_structure(base_path)
    df_normal_anno, df_not_normal_anno = analyze_annotations()
    
    all_features = []
    if normal_path.exists():
        print("\nОбработка normal случаев...")
        normal_folders = [f for f in normal_path.iterdir() if f.is_dir()]
        np.random.shuffle(normal_folders)
        print("Папки normal перемешаны перед обработкой")
        
        for i, patient_folder in enumerate(normal_folders):
            print(f"Обработка {i+1}/{len(normal_folders)}: {patient_folder.name}")
            
            features = process_patient_folder(patient_folder, patient_folder.name, 0)
            if features:
                all_features.append(features)
    
    if not_normal_path.exists():
        print("\nОбработка not_normal случаев...")
        not_normal_folders = [f for f in not_normal_path.iterdir() if f.is_dir()]
        np.random.shuffle(not_normal_folders)
        print("Папки not_normal перемешаны перед обработкой")
        
        for i, patient_folder in enumerate(not_normal_folders):
            print(f"Обработка {i+1}/{len(not_normal_folders)}: {patient_folder.name}")
            
            features = process_patient_folder(patient_folder, patient_folder.name, 1)
            if features:
                all_features.append(features)
    
    # Создаем DataFrame
    if all_features:
        df_dataset = pd.DataFrame(all_features)
        df_dataset = df_dataset.sample(frac=1, random_state=42).reset_index(drop=True)
        print("Весь датасет перемешан перед сохранением")
        output_file = 'medical_dataset.csv'
        df_dataset.to_csv(output_file, index=False, encoding='utf-8')
        print(f"Файл: {output_file}")
        print(f"Размер: {df_dataset.shape}")
        print(f"Колонки: {len(df_dataset.columns)}")
        print(f"Normal случаев: {len(df_dataset[df_dataset['target'] == 0])}")
        print(f"Not_normal случаев: {len(df_dataset[df_dataset['target'] == 1])}")
        print("\nСтатистика датасета:")
        print(df_dataset.describe())
        
        return df_dataset
    else:
        print("Не удалось извлечь данные из папок")
        return None

def main():
    np.random.seed(42)
    base_path = input("Введите путь к папке с данными: ").strip().strip('"')
    
    if not os.path.exists(base_path):
        print("Указанный путь не существует!")
        return
    df_dataset = create_complete_dataset(base_path)
    
    if df_dataset is not None:
        print("\nДатасет успешно создан и сохранен как 'medical_dataset.csv'")
        print("\nПервые 5 строк датасета:")
        print(df_dataset.head())
        with open('dataset_columns.txt', 'w', encoding='utf-8') as f:
            f.write("Колонки датасета:\n")
            for col in df_dataset.columns:
                f.write(f"{col}\n")
        print("\nСписок колонок сохранен в 'dataset_columns.txt'")
    else:
        print("Не удалось создать датасет")

if __name__ == "__main__":
    main()
