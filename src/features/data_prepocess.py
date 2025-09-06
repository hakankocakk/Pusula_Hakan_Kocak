import pandas as pd
import os
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.preprocessing import MultiLabelBinarizer
import joblib

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import unicodedata
import joblib
import re

from unidecode import unidecode
from rapidfuzz import fuzz, process

from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA


def load_data_excel(path: str):
    return pd.read_excel(path)

def load_data(path: str):
    return pd.read_csv(path)


def normalize_text(text):
    if pd.isnull(text):
        return text 
    text = unicodedata.normalize('NFKD', text)
    text = "".join([c for c in text if not unicodedata.combining(c)])
    return text.lower().strip()


def fuzzy_replace(text, mapping, threshold=85):
    if pd.isna(text):
        return text
    words = text.lower().split()
    new_words = []
    for w in words:
        replaced = False
        for allergi in mapping:
            if fuzz.ratio(w.lower(), allergi) >= threshold:
                new_words.append(allergi)
                replaced = True
                break
        if not replaced:
            new_words.append(w)
    return " ".join(new_words)


def col_fix(df, cols_to_fix):
    for col in cols_to_fix:
        merged = (df.groupby('HastaNo')[col].apply(lambda x: ','.join(x.dropna().unique()) if x.notna().any() else np.nan).reset_index())
        df = df.drop(columns=[col]).merge(merged, on='HastaNo', how='left')
    return df


def create_referance_dataframe(df):
    ref_df = df[['HastaNo', 'Cinsiyet', 'KanGrubu', 'KronikHastalik', 'Alerji']].copy()
    
    ref_df = ref_df.groupby('HastaNo').agg({
        'Cinsiyet': 'first',
        'KanGrubu': 'first',
        'KronikHastalik': 'first',
        'Alerji': 'first'
    }).reset_index()
    return ref_df

def fill_missing_with_ref(df, ref_df, cols):

    ref_indexed = ref_df.set_index('HastaNo')
    
    for col in cols:
        if col in df.columns and col in ref_df.columns:
            df[col] = df[col].fillna(df['HastaNo'].map(ref_indexed[col]))
    return df

def replace_nan_yok(df, cols):
    for col in cols:
        df[col] = df[col].fillna("Yok")
    return df

def fill_tanilar_with_mode(row, mode_map):
    if pd.isna(row["Tanilar"]):
        key = row["TedaviAdi"]
        return mode_map.get(key, row["Tanilar"])
    return row["Tanilar"]

def fill_bolum_with_mode(row, mode_map):
    if pd.isna(row["Bolum"]):
        key = row["TedaviAdi"]
        return mode_map.get(key, row["Bolum"])
    return row["Bolum"]


def fill_uygulama_yeri_with_mode(row, mode_map):
    if pd.isna(row["UygulamaYerleri"]):
        key = (row["Tanilar"], row["TedaviAdi"])
        return mode_map.get(key, row["UygulamaYerleri"])
    return row["UygulamaYerleri"]

def fill_uygulama_yeri_2_with_mode(row, mode_map):
    if pd.isna(row["UygulamaYerleri"]):
        key = (row["HastaNo"], row["TedaviAdi"])
        return mode_map.get(key, row["UygulamaYerleri"])
    return row["UygulamaYerleri"]

def select_row(df):
    df = df[~(df["UygulamaYerleri"].isna() & df["Tanilar"].isna())]
    return df

def replace_nan_diğer(df, cols):
    for col in cols:
        df[col] = df[col].fillna("diğer")
    return df


def replace_nan_eksik_tanı(df, cols):
    for col in cols:
        df[col] = df[col].fillna("eksik_tanı")
    return df

def uygulama_suresi_replace(df):
    df["UygulamaSuresi"] = df["UygulamaSuresi"].str.replace("Dakika", "", regex=False).astype(int)
    return df

def drop_duplicate(df):
    df = df.drop(["HastaNo"], axis=1)
    unique_df = df.drop_duplicates().reset_index(drop=True)
    return unique_df


def onehotencoder(dataframe, model_path, train=True):
    one_hot_cat_cols = ["Cinsiyet", "KanGrubu", "Uyruk", "Bolum"]
    
    drop_map = {
        "Cinsiyet": ["diğer"], 
        "KanGrubu": ["diğer"],
        "Uyruk": ["Tokelau"],
        "Bolum": ["diğer"]
    }

    if train:
        ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False,drop=[drop_map.get(col, None) for col in one_hot_cat_cols])
        encoded_cols = ohe.fit_transform(dataframe[one_hot_cat_cols])
        joblib.dump(ohe, os.path.join(model_path, 'one_hot_encoder.pkl'))
        new_columns = ohe.get_feature_names_out(one_hot_cat_cols)
        encoded_df = pd.DataFrame(encoded_cols, columns=new_columns, index=dataframe.index)
        dataframe = pd.concat([dataframe, encoded_df], axis=1)
        dataframe.drop(columns=one_hot_cat_cols, inplace=True)
    else:
        loaded_ohe = joblib.load(os.path.join(model_path, 'one_hot_encoder.pkl'))
        encoded_test_data = loaded_ohe.transform(dataframe[one_hot_cat_cols])
        new_columns = loaded_ohe.get_feature_names_out(one_hot_cat_cols)
        encoded_test_df = pd.DataFrame(encoded_test_data, columns=new_columns, index=dataframe.index)
        dataframe = pd.concat([dataframe, encoded_test_df], axis=1)
        dataframe.drop(columns=one_hot_cat_cols, inplace=True)
    
    return dataframe


def multilabel_binarize(dataframe, multilabel_cols, model_path, train=True):
    mlb_dict = {}

    for col in multilabel_cols:
        dataframe[col] = dataframe[col].fillna('').apply(
            lambda x: [v.strip().lower() for v in str(x).split(',') if v.strip()]
        )

        if train:
            mlb = MultiLabelBinarizer()
            encoded = mlb.fit_transform(dataframe[col])
            joblib.dump(mlb, os.path.join(model_path, f'mlb_{col}.pkl'))
        else:
            mlb = joblib.load(os.path.join(model_path, f'mlb_{col}.pkl'))
            encoded = mlb.transform(dataframe[col])

        encoded_df = pd.DataFrame(
            encoded,
            columns=[f"{col}_{cls}" for cls in mlb.classes_],
            index=dataframe.index
        )

        dataframe = pd.concat([dataframe.drop(columns=[col]), encoded_df], axis=1)

        mlb_dict[col] = mlb

    return dataframe, mlb_dict


def replace_cols(df):
    df["Alerji_voltaren"] = df["Alerji_voltaren"] + df["Alerji_volteren"]
    df["KronikHastalik_hipotirodizm"] = df["KronikHastalik_hipotirodizm"] + df["KronikHastalik_hiportiroidizm"]
    df = df.drop(["Alerji_volteren", "KronikHastalik_hiportiroidizm"], axis=1)
    return df

def preprocess_text(text):
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()  
    text = re.sub(r'[^\w\s]', ' ', text)  
    text = re.sub(r'\s+', ' ', text).strip() 
    return text

def embed_text_columns(df, text_cols, model_path, model_name="dbmdz/bert-base-turkish-cased", reduced_dim=15, train=True):
    model = SentenceTransformer(model_name)
    
    for col in text_cols:
        # Ön işleme
        texts = df[col].fillna("").apply(preprocess_text).tolist()
        embeddings = model.encode(texts, convert_to_numpy=True)
        
        if train:
            pca = PCA(n_components=reduced_dim)
            reduced_embeddings = pca.fit_transform(embeddings)
            joblib.dump(pca, os.path.join(model_path, f'pca_{col}.pkl'))  # PCA modelini kaydet
        else:
            pca = joblib.load(os.path.join(model_path, f'pca_{col}.pkl'))
            reduced_embeddings = pca.transform(embeddings)
        
        emb_df = pd.DataFrame(
            reduced_embeddings,
            columns=[f"{col}_emb_{i}" for i in range(reduced_dim)],
            index=df.index
        )
        df = pd.concat([df.drop(columns=[col]), emb_df], axis=1)
        
    return df

def labelencoding_tedavisuresi(dataframe):
    tedavi_map = {
        '1 Seans': 0, '2 Seans': 1, '3 Seans': 2, '4 Seans': 3, '5 Seans': 4,
        '6 Seans': 5, '7 Seans': 6, '8 Seans': 7, '10 Seans': 8, '11 Seans': 9,
        '14 Seans': 10, '15 Seans': 11, '16 Seans': 12, '17 Seans': 13, '18 Seans': 14,
        '19 Seans': 15, '20 Seans': 16, '21 Seans': 17, '22 Seans': 18, '25 Seans': 19,
        '29 Seans': 20, '30 Seans': 21, '37 Seans': 22
    }

    dataframe["TedaviSuresi"] = dataframe["TedaviSuresi"].map(tedavi_map)
    return dataframe


def normalization(dataframe, num_cols, model_path, train=True):
    if train:
        scaler = StandardScaler()
        dataframe[num_cols] = scaler.fit_transform(dataframe[num_cols])
        joblib.dump(scaler, os.path.join(model_path, 'standardscaler.pkl'))
    else:
        loaded_scaler = joblib.load(os.path.join(model_path, 'standardscaler.pkl'))
        dataframe[num_cols] = loaded_scaler.transform(dataframe[num_cols])
        
    return dataframe


def save_data(data,  data_path: str):
    data.to_csv(data_path, index=False)


def main():
    raw_data_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "datas", "raw"
    )
    interim_data_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "datas", "interim"
    )
    processed_data_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "datas", "processed"
    )
    model_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "models"
    )
    os.makedirs(processed_data_path, exist_ok=True)
    os.makedirs(model_path, exist_ok=True)

    data = load_data_excel(
        os.path.join(raw_data_path, "Talent_Academy_Case_DT_2025.xlsx")
    )

    train = load_data(
        os.path.join(interim_data_path, "train.csv")
    )

    test = load_data(
        os.path.join(interim_data_path, "test.csv")
    )

    Allergies = ["polen", "toz", "arveles", "coraspin", "sucuk", "novalgin", "yer fistigi", "voltaren", "gripin"]
    train["Alerji"] = train["Alerji"].apply(lambda x: fuzzy_replace(x, Allergies))
    test["Alerji"] = test["Alerji"].apply(lambda x: fuzzy_replace(x, Allergies))
    train["Alerji"] = train["Alerji"].apply(normalize_text)
    test["Alerji"] = test["Alerji"].apply(normalize_text)

    cols_to_fix = ['KronikHastalik', 'Alerji']
    train = col_fix(train, cols_to_fix)
    test = col_fix(test, cols_to_fix)

    ref_df = create_referance_dataframe(data)

    cols_to_fill = ['Cinsiyet', 'KanGrubu', 'KronikHastalik', 'Alerji']
    train = fill_missing_with_ref(train, ref_df, cols_to_fill)
    test = fill_missing_with_ref(test, ref_df, cols_to_fill)

    cols = ["KronikHastalik", "Alerji"]
    train = replace_nan_yok(train, cols)
    test = replace_nan_yok(test, cols)

    mode_map = train.groupby("TedaviAdi")["Tanilar"].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else None).to_dict()
    train["Tanilar"] = train.apply(lambda row: fill_tanilar_with_mode(row, mode_map), axis=1)
    test["Tanilar"] = test.apply(lambda row: fill_tanilar_with_mode(row, mode_map), axis=1)

    bolum_mode_map = train.groupby("TedaviAdi")["Bolum"].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else None).to_dict()
    train["Bolum"] = train.apply(lambda row: fill_bolum_with_mode(row, bolum_mode_map), axis=1)
    test["Bolum"] = test.apply(lambda row: fill_bolum_with_mode(row, bolum_mode_map), axis=1)

    uygulama_yeri_mode_map = train.groupby(["Tanilar", "TedaviAdi"])["UygulamaYerleri"].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else None).to_dict()
    train["UygulamaYerleri"] = train.apply(lambda row: fill_uygulama_yeri_with_mode(row, uygulama_yeri_mode_map), axis=1)
    test["UygulamaYerleri"] = test.apply(lambda row: fill_uygulama_yeri_with_mode(row, uygulama_yeri_mode_map), axis=1)

    uygulama_yeri_2_mode_map = train.groupby(["HastaNo", "TedaviAdi"])["UygulamaYerleri"].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else None).to_dict()
    train["UygulamaYerleri"] = train.apply(lambda row: fill_uygulama_yeri_2_with_mode(row, uygulama_yeri_2_mode_map), axis=1)
    test["UygulamaYerleri"] = test.apply(lambda row: fill_uygulama_yeri_2_with_mode(row, uygulama_yeri_2_mode_map), axis=1)

    train = select_row(train)
    test = select_row(test)

    cols = ["Bolum", "UygulamaYerleri", "Cinsiyet", "KanGrubu"]
    train = replace_nan_diğer(train, cols)
    test = replace_nan_diğer(test, cols)

    cols = ["Tanilar"]
    train = replace_nan_eksik_tanı(train, cols)
    test = replace_nan_eksik_tanı(test, cols)

    train =  uygulama_suresi_replace(train)
    test =  uygulama_suresi_replace(test)

    unique_train = drop_duplicate(train)
    unique_test = drop_duplicate(test)

    unique_train = onehotencoder(unique_train, model_path, train=True)
    unique_test = onehotencoder(unique_test, model_path, train=False)

    multilabel_cols = ["Alerji", "KronikHastalik"]
    unique_train, mlb_models = multilabel_binarize(unique_train, multilabel_cols, model_path, train=True)
    unique_test, _ = multilabel_binarize(unique_test, multilabel_cols, model_path, train=False)


    unique_train = replace_cols(unique_train)
    unique_test = replace_cols(unique_test)

    text_cols = ["Tanilar", "TedaviAdi", "UygulamaYerleri"]
    unique_train = embed_text_columns(df = unique_train, text_cols = text_cols, model_path = model_path, reduced_dim = 15)
    unique_test = embed_text_columns(df = unique_test, text_cols = text_cols, model_path = model_path, reduced_dim = 15)


    unique_train = labelencoding_tedavisuresi(unique_train)
    unique_test = labelencoding_tedavisuresi(unique_test)

    num_cols = ["Yas", "UygulamaSuresi"]
    unique_train = normalization(unique_train, num_cols, model_path=model_path, train=True)
    unique_test = normalization(unique_test, num_cols, model_path=model_path, train=False)

    save_data(unique_train,
                  os.path.join(processed_data_path, "train.csv"))

    save_data(unique_test,
                os.path.join(processed_data_path, "test.csv"))



if __name__ == "__main__":
    main()

