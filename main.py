from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from joblib import load
import pandas as pd
import logging
import re

app = FastAPI()

model = load('cars_prediction_model.joblib')


class Item(BaseModel):
    name: str
    year: int
    selling_price: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    torque: str
    seats: float


class Items(BaseModel):
    objects: List[Item]


@app.post("/predict_item")
def predict_item(item: Item) -> float:
    data = item.model_dump()

    df = pd.DataFrame([data])
    df = prepare_df(df)

    logging.error(df.head())

    prediction = model.predict(df)

    return prediction[0]


@app.post("/predict_items")
def predict_items(items: List[Item]) -> List[float]:
    df = pd.DataFrame([item.model_dump() for item in items])

    predictions = model.predict(df)

    return predictions.to_list()


def prepare_df(df):
    df['mileage'] = df['mileage'].str.extract(r'(\d+(\.\d+)?)', expand=False)[0].astype(float)
    df['engine'] = df['engine'].str.extract(r'(\d+(\.\d+)?)', expand=False)[0].astype(float)
    df['max_power'] = df['max_power'].str.extract(r'(\d+(\.\d+)?)', expand=False)[0].astype(float)
    df[['torque_value', 'max_torque_rpm']] = df['torque'].apply(lambda x: pd.Series(extract_torque_values(x)))
    df['torque_value'] = pd.to_numeric(df['torque_value'], errors='coerce')
    df['max_torque_rpm'] = pd.to_numeric(df['max_torque_rpm'], errors='coerce')

    categorical_columns_to_drop = ['name', 'fuel', 'seller_type', 'transmission', 'owner', 'selling_price', 'torque']

    df = df.drop(categorical_columns_to_drop, axis = 1)

    return df


def extract_torque_values(torque_str):
    if pd.isna(torque_str):
        return pd.NA, pd.NA

    match = re.search(r'(\d+\.?\d*)\s*(Nm|kgm)?\s*(?:@|at)?\s*([\d,]+\.?\d*)?\s*-?\s*([\d,]+)?\s*(rpm)?', torque_str,
                      re.IGNORECASE)
    if match:
        torque_value = match.group(1)
        torque_unit = match.group(2)
        rpm_value_start = match.group(3)
        rpm_value_end = match.group(4)

        if torque_unit and torque_unit.lower() == 'kgm':
            torque_value = float(torque_value) * 9.81
        else:
            torque_value = float(torque_value)

        if rpm_value_start and rpm_value_end:
            rpm_value_start = float(rpm_value_start.replace(',', ''))
            rpm_value_end = float(rpm_value_end.replace(',', ''))
            rpm_value = (rpm_value_start + rpm_value_end) / 2
        elif rpm_value_start:
            rpm_value = float(rpm_value_start.replace(',', ''))
        else:
            rpm_value = pd.NA

        return torque_value, rpm_value
    else:
        return pd.NA, pd.NA
