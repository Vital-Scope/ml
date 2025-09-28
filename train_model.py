import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
import joblib
import warnings
warnings.filterwarnings('ignore')

RANDOM_STATE = 42

def load_and_analyze_data():
    df = pd.read_csv('medical_dataset.csv')
    print(f"Размер данных: {df.shape}")
    target_counts = df['target'].value_counts()
    print(f"\nРаспределение целевой переменной:")
    print(f"Normal (0): {target_counts[0]} ({target_counts[0]/len(df)*100:.1f}%)")
    print(f"Not Normal (1): {target_counts[1]} ({target_counts[1]/len(df)*100:.1f}%)")
    features_df = df.drop(['patient_id', 'folder_path', 'csv_files_count', 'target'], 
                         axis=1, errors='ignore')
    print(f"\nКоличество признаков: {len(features_df.columns)}")
    missing_values = features_df.isnull().sum()
    if missing_values.sum() > 0:
        print("\nПропущенные значения:")
        for col, count in missing_values[missing_values > 0].items():
            print(f"  {col}: {count} ({count/len(df)*100:.1f}%)")
    else:
        print("\nПропущенных значений нет")

    print("\nСтатистика по ключевым признакам:")
    key_features = ['bpm_mean', 'bpm_std', 'bpm_rmssd', 'bpm_cv']
    for feat in key_features:
        if feat in features_df.columns:
            normal_vals = features_df[df['target'] == 0][feat].dropna()
            not_normal_vals = features_df[df['target'] == 1][feat].dropna()
            print(f"\n{feat}:")
            print(f"  Normal: mean={normal_vals.mean():.2f}, std={normal_vals.std():.2f}")
            print(f"  Not Normal: mean={not_normal_vals.mean():.2f}, std={not_normal_vals.std():.2f}")
    return df, features_df

def create_better_features(df):
    features_df = df.drop(['patient_id', 'folder_path', 'csv_files_count'], 
                         axis=1, errors='ignore')
    X = features_df.drop('target', axis=1)
    y = df['target']
    X_new = X.copy()
    
    # Признаки вариабельности
    if all(col in X.columns for col in ['bpm_std', 'bpm_mean']):
        X_new['bpm_cv_enhanced'] = X['bpm_std'] / (X['bpm_mean'] + 1e-8)
        X_new['bpm_stability'] = X['bpm_std'] / (X['bpm_max'] - X['bpm_min'] + 1e-8)
    
    # Отношения между статистиками
    if all(col in X.columns for col in ['bpm_q75', 'bpm_q25']):
        X_new['bpm_iqr'] = X['bpm_q75'] - X['bpm_q25']
        X_new['bpm_iqr_to_range'] = X_new['bpm_iqr'] / (X['bpm_max'] - X['bpm_min'] + 1e-8)
    
    # Признаки асимметрии распределения
    if 'bpm_skew' in X.columns:
        X_new['bpm_skew_abs'] = np.abs(X['bpm_skew'])
    
    # Взаимодействие ЧСС и Тонуса
    bpm_features = [col for col in X.columns if 'bpm' in col]
    uretus_features = [col for col in X.columns if 'uretus' in col]
    
    if bpm_features and uretus_features:
        for bpm_feat in ['bpm_mean', 'bpm_std']:
            for uretus_feat in ['uretus_mean', 'uretus_std']:
                if bpm_feat in X.columns and uretus_feat in X.columns:
                    X_new[f'{bpm_feat}_ratio_{uretus_feat}'] = X[bpm_feat] / (X[uretus_feat] + 1e-8)
    
    print(f"Создано {len(X_new.columns) - len(X.columns)} новых признаков")
    print(f"Общее количество признаков: {len(X_new.columns)}")
    
    return X_new, y

def select_best_features(X, y, k=20):
    # Удаляем признаки с нулевой дисперсией
    X_filtered = X.loc[:, X.std() > 0]
    
    if len(X_filtered.columns) <= k:
        print(f"Признаков меньше {k}, используем все")
        return X_filtered, X_filtered.columns
    selector = SelectKBest(score_func=f_classif, k=min(k, len(X_filtered.columns)))
    X_selected = selector.fit_transform(X_filtered, y)
    
    selected_features = X_filtered.columns[selector.get_support()]
    scores = selector.scores_[selector.get_support()]
    feature_scores = pd.DataFrame({
        'feature': selected_features,
        'score': scores
    }).sort_values('score', ascending=False)
    
    print("Топ-10 признаков")
    for i, row in feature_scores.head(10).iterrows():
        print(f"  {row['feature']}: {row['score']:.2f}")
    
    X_selected_df = pd.DataFrame(X_selected, columns=selected_features, index=X.index)
    
    return X_selected_df, selected_features

def create_advanced_model(X_train, y_train):
    # Балансировка классов
    scale_pos_weight = len(y_train[y_train==0]) / len(y_train[y_train==1])
    model = XGBClassifier(
        n_estimators=800,
        max_depth=4,  
        learning_rate=0.07,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_alpha=0.7, 
        reg_lambda=0.5,
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        eval_metric='logloss'
    )
    
    return model

def evaluate_model_comprehensively(model, X_train, X_test, y_train, y_test, model_name=""):
    # Обучение модели
    model.fit(X_train, y_train)
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    y_pred_proba_train = model.predict_proba(X_train)[:, 1]
    y_pred_proba_test = model.predict_proba(X_test)[:, 1]

    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    
    print(f"Accuracy на train: {train_accuracy:.4f}")
    print(f"Accuracy на test:  {test_accuracy:.4f}")
    
    # Анализ переобучения
    overfitting = train_accuracy - test_accuracy
    print(f"Переобучение: {overfitting:.4f}")

    print("\nClassification Report (Test):")
    print(classification_report(y_test, y_pred_test, target_names=['Normal', 'Not_Normal']))
    
    # Матрица ошибок
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    cm_train = confusion_matrix(y_train, y_pred_train)
    sns.heatmap(cm_train, annot=True, fmt='d', cmap='Blues', ax=ax1)
    ax1.set_title('Train Matrix')
    ax1.set_ylabel('True Label')
    ax1.set_xlabel('Predicted Label')
    
    cm_test = confusion_matrix(y_test, y_pred_test)
    sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues', ax=ax2)
    ax2.set_title('Test Matrix')
    ax2.set_ylabel('True Label')
    ax2.set_xlabel('Predicted Label')
    
    plt.tight_layout()
    plt.savefig(f'comprehensive_evaluation_{model_name}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return test_accuracy, model

def main():
    df, features_df = load_and_analyze_data()
    X, y = create_better_features(df)
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    X_df = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
    X_selected, selected_features = select_best_features(X_df, y, k=15)
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    
    print(f"\nФинальный размер данных:")
    print(f"Train: {X_train.shape}")
    print(f"Test: {X_test.shape}")
    print(f"Признаков: {X_train.shape[1]}")
    model = create_advanced_model(X_train, y_train)
    accuracy, trained_model = evaluate_model_comprehensively(
        model, X_train, X_test, y_train, y_test, "improved"
    )
    cv_scores = cross_val_score(trained_model, X_selected, y, cv=5, scoring='accuracy')
    print(f"Кросс-валидация: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    if hasattr(trained_model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'feature': selected_features,
            'importance': trained_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nТоп-10 самых важных признаков:")
        for i, row in importance_df.head(10).iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")
    model_data = {
        'model': trained_model,
        'imputer': imputer,
        'scaler': scaler,
        'selected_features': selected_features.tolist(),
        'all_features': X.columns.tolist(),
        'accuracy': accuracy,
        'cv_score': cv_scores.mean()
    }
    
    joblib.dump(model_data, 'best_model.joblib')
    print(f"Точность на тесте: {accuracy:.4f}")
    print(f"Кросс-валидация: {cv_scores.mean():.4f}")

if __name__ == "__main__":
    main()
