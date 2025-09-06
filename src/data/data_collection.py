
import pandas as pd
import os
from sklearn.model_selection import train_test_split


def load_data(path: str):
    return pd.read_excel(path)


def data_split(dataframe, test_size=0.3, random_state=42):
    unique_patients = dataframe['HastaNo'].unique()
    train_patients, test_patients = train_test_split(
        unique_patients, test_size=test_size, random_state=random_state
    )
    
    train = dataframe[dataframe['HastaNo'].isin(train_patients)].reset_index(drop=True)
    test = dataframe[dataframe['HastaNo'].isin(test_patients)].reset_index(drop=True)
    
    return train, test


def save_data(data,  data_path: str):
    data.to_csv(data_path, index=False)

    
def main():

    raw_data_path = os.path.join(os.path.dirname(__file__),
                                 "..", "..", "datas", "raw",
                                 "Talent_Academy_Case_DT_2025.xlsx")
    interim_data_path = os.path.join(os.path.dirname(__file__),
                                     "..", "..", "datas", "interim")
    os.makedirs(interim_data_path, exist_ok=True)

    data = load_data(raw_data_path)

    train_data, test_data = data_split(data)

    save_data(train_data,
                os.path.join(interim_data_path, "train.csv"))
    save_data(test_data,
                os.path.join(interim_data_path, "test.csv"))



if __name__ == "__main__":
    main()