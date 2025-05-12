import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os

def preprocess_data(input_path, output_path):
    # Pastikan folder output tersedia
    os.makedirs(output_path, exist_ok=True)
    
    # Load dataset
    print("[INFO] Loading dataset...")
    df = pd.read_csv(input_path)
    
    # Drop kolom yang tidak relevan
    print("[INFO] Dropping unnecessary columns...")
    df = df.drop(columns=['id'])
    
    # Label Encoding pada kolom diagnosis
    print("[INFO] Encoding target variable...")
    le = LabelEncoder()
    df['diagnosis'] = le.fit_transform(df['diagnosis'])
    
    # Normalisasi fitur
    print("[INFO] Normalizing features...")
    scaler = StandardScaler()
    X = df.drop(columns=['diagnosis'])
    y = df['diagnosis']
    X_scaled = scaler.fit_transform(X)
    
    # Train Test Split
    print("[INFO] Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # Simpan dataset yang sudah diproses
    print("[INFO] Saving preprocessed datasets...")
    pd.DataFrame(X_train).to_csv(f'{output_path}/X_train.csv', index=False)
    pd.DataFrame(X_test).to_csv(f'{output_path}/X_test.csv', index=False)
    pd.DataFrame(y_train).to_csv(f'{output_path}/y_train.csv', index=False)
    pd.DataFrame(y_test).to_csv(f'{output_path}/y_test.csv', index=False)
    
    print("[SUCCESS] Preprocessing completed and files are saved at:", output_path)

# Pemanggilan fungsi
preprocess_data('/BreastCancer_raw/breast-cancer.csv', '/preprocessing/BreastCancer_preprocessing')