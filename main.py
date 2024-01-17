import fastapi
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from fastapi import FastAPI, HTTPException
import numpy as np
from pydantic import BaseModel

app = FastAPI()

df = pd.read_csv('data_new_df_metadatos_bars_fl.csv')

class Ubicacion(BaseModel):
    latitud: float
    longitud: float

#Primer modelo entrega recomendaciones por cercanía

@app.get('/r.cercania/{latitud}/{longitud}')
async def Recomendacion(latitud: float, longitud: float):
    try:
        columnasd_interes = ['latitude', 'longitude']
        data = df[columnasd_interes]

        # Creo el modelo KNN
        knn_model = NearestNeighbors(n_neighbors=10, metric='haversine')
        knn_model.fit(data)

        # Especifico la ubicación de búsqueda
        new_lat = float(latitud)  # Se reemplaza con input
        new_long = float(longitud)  #Se reemplaza con input

        # Se consultan los 10 lugares más cercanos
        distances, indices = knn_model.kneighbors([[new_lat, new_long]])

        # Imprimo los resultados
        nearest_places = df.iloc[indices[0]][['name','avg_rating','address','latitude','longitude']].to_dict(orient='index')

        return nearest_places
    except Exception as e:
        return {"error": str(e)}

#Segundo modelo entrega recomendaciones por cercanía luego de haber filtrado el dataset
#mediante el promedio de estrellas de los diferentes bares.

@app.get('/r.stars/{latitud1}/{longitud1}')
async def Recomendacion(latitud1: float, longitud1: float):
    try:
        filtered_df = df[df['avg_rating']>3.5]
             
        
        columnasd_interes = ['latitude', 'longitude']
        data = filtered_df[columnasd_interes]

        # Creo el modelo KNN
        knn_model = NearestNeighbors(n_neighbors=10, metric='haversine')
        knn_model.fit(data)

        # Especifico la ubicación de búsqueda
        new_lat = float(latitud1)  # Se reemplaza con input
        new_long = float(longitud1)  #Se reemplaza con input

        # Se consultan los 10 lugares más cercanos
        distances, indices = knn_model.kneighbors([[new_lat, new_long]])

        # Imprimo los resultados
        nearest_places1 = filtered_df.iloc[indices[0]][['name','avg_rating','address','latitude','longitude']].to_dict(orient='index')
      
        return nearest_places1
    except Exception as e:
        return {"error": str(e)}
        
