from joblib import load
from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.datasets import load_iris

iris = load_iris()

#Chargement du model
loaded_model = load('logreg.joblib')

app = FastAPI()

#Definition d'un objet (une classe) pour realiser des requetes
class request_body(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float
    
# Definition du chemin du point d'entré(API)
@app.post("/predict")

def predict(data  : request_body):
    #Nouvelle donne sur lequel on fait la prediction
    new_data = [[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]]
    
    #Prediction
    prediction = loaded_model.predict(new_data)[0]
    
    #On retour les nom de l'espece iris
    return {'class' : iris.target_names[prediction]}