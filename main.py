from fastapi import FastAPI, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List
import pandas as pd
import numpy as np
import joblib
import re


app = FastAPI()


class Item(BaseModel):
    name: str
    year: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    seats: float


@app.get('/')
async def root():
    return {'message': 'Enter data to predict car price'}


mdn = joblib.load('median.pkl')
ohe = joblib.load('OHE.pkl')
scl = joblib.load('scaler.pkl')
ridge = joblib.load('ridge.pkl')


def rm_units(x, pattern=r"\d+.?\d+"):
    return "".join(re.findall(pattern, x))


def data_preprocessing(data):
    data = data.drop('name', axis=1)

    data['mileage'] = data['mileage'].apply(str).apply(rm_units)
    data['engine'] = data['engine'].apply(str).apply(rm_units)
    data['max_power'] = data['max_power'].apply(str).apply(rm_units)

    data['mileage'] = pd.to_numeric(data['mileage'], downcast='float', errors='coerce')
    data['engine'] = pd.to_numeric(data['engine'], downcast='float', errors='coerce')
    data['max_power'] = pd.to_numeric(data['max_power'], downcast='float', errors='coerce')

    data['mileage'] = data['mileage'].fillna(mdn['mileage'])
    data['engine'] = data['engine'].fillna(mdn['engine'])
    data['max_power'] = data['max_power'].fillna(mdn['max_power'])
    data['seats'] = data['seats'].fillna(mdn['seats'])

    data['power_per_volume'] = data['max_power'] / data['engine']
    data['year_squared'] = data['year'] ** 2
    data['owner_type'] = data['owner'] \
        .apply(lambda x: 1 if (x == 'First Owner' or x == 'Second Owner' or x == 'Test Drive Car') else 0)

    return data


def ohe_scl(data):
    data = data_preprocessing(data)
    data_ohe = ohe.transform(data.select_dtypes(include=object))
    data_oh = pd.DataFrame(data=data_ohe, columns=ohe.get_feature_names_out(list(data.select_dtypes(include=object))))
    data_onehot = data.select_dtypes(include=np.number).join(data_oh)
    data_scl = scl.transform(data_onehot)
    data_scaled = pd.DataFrame(data=data_scl, columns=list(data_onehot))

    return data_scaled


@app.post("/predict_item")
def predict_item(item: Item) -> float:
    data = pd.DataFrame([item.dict().copy()])
    data = data_preprocessing(data)
    data_scaled = ohe_scl(data)

    return ridge.predict(data_scaled)[0]


@app.post("/predict_items")
def predict_items(file: UploadFile):
    data = pd.read_csv(file.filename)
    data = data.drop(['torque', 'selling_price'], axis=1)
    data = data_preprocessing(data)
    data_scaled = ohe_scl(data)

    data['pred'] = ridge.predict(data_scaled)
    data.to_csv('prediction.csv')

    return FileResponse('prediction.csv')