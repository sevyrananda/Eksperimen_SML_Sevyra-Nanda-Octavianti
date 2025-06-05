import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def preprocess_iris():
    # Path ke root folder (tempat iris_raw.csv berada)
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_path = os.path.join(BASE_DIR, 'iris_raw.csv')

    # Output di dalam folder ini (preprocessing)
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'iris_preprocessing.csv')

    # 1. Load dataset
    df = pd.read_csv(input_path)

    # 2. Hapus kolom ID jika ada
    if 'Id' in df.columns:
        df.drop('Id', axis=1, inplace=True)

    # 3. Label encoding untuk kolom target
    le = LabelEncoder()
    df['Species'] = le.fit_transform(df['Species'])

    # 4. Normalisasi fitur numerik
    numeric_cols = df.drop('Species', axis=1).columns
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    # 5. Simpan hasil preprocessing
    df.to_csv(output_path, index=False)

    print("Preprocessing selesai! Berikut contoh data:")
    print(df.head())
    return df

if __name__ == "__main__":
    preprocess_iris()
