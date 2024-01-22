import fastapi
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

import numpy as np
from pydantic import BaseModel

app = FastAPI()

# CORS
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

df = pd.read_csv('data_new_df_metadatos_bars_fl.csv')

class Ubicacion(BaseModel):
    latitud: float
    longitud: float

def format_table(data):
    table_html = "<table border='1'><tr>"
    if data:
        columns = list(data.values())[0].keys()
        for col in columns:
            table_html += f"<th>{col}</th>"
        table_html += "</tr>"
        
        for row in data.values():
            table_html += "<tr>"
            for value in row.values():
                table_html += f"<td>{value}</td>"
            table_html += "</tr>"
    table_html += "</table>"
    return table_html

# Ruta para la página de inicio con formularios y resultados
@app.get("/", response_class=HTMLResponse)
async def read_root(latitud: float = None, longitud: float = None, latitud1: float = None, longitud1: float = None):
    result_cercania = None
    result_stars = None

    if latitud is not None and longitud is not None:
        result_cercania = await Recomendacion(latitud, longitud)

    if latitud1 is not None and longitud1 is not None:
        result_stars = await Recomendacion(latitud1, longitud1)

    return f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Recomendación de Bares en Tampa</title>
        <style>
            body {{
                background-color: #000;
                color: #fff;
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 0;
            }}

            header {{
                background-color: #007acc;
                padding: 20px;
                text-align: center;
            }}

            h1 {{
                margin: 0;
                font-size: 2em;
            }}

            section {{
                padding: 20px;
            }}

            code {{
                background-color: #333;
                color: #00bcd4;
                padding: 5px;
                border-radius: 3px;
                font-family: monospace;
            }}

            form {{
                margin-bottom: 20px;
            }}

            input {{
                padding: 8px;
                margin-right: 10px;
            }}

            button {{
                padding: 8px;
                background-color: #007acc;
                color: #fff;
                border: none;
                cursor: pointer;
            }}

            footer {{
                background-color: #007acc;
                padding: 10px;
                text-align: center;
                position: fixed;
                bottom: 0;
                width: 100%;
            }}

            .result-table {{
                margin-top: 20px;
            }}
        </style>
    </head>
    <body>
        <header>
            <h1>Recomendación de Bares</h1>

            <h2>Modelo de ML y FastAPI</h2>
        </header>
        
        <section>
            <h2>Modelo de Recomendación por Cercanía</h2>
            <form action="/" method="get">
                <label for="latitud">Latitud:</label>
                <input type="text" id="latitud" name="latitud" required>
                <label for="longitud">Longitud:</label>
                <input type="text" id="longitud" name="longitud" required>
                <button type="submit">Obtener Recomendaciones</button>
            </form>
            <div class="result-table" id="result_cercania">
                {format_table(result_cercania)}
            </div>
        </section>
        
        <section>
            <h2>Modelo de Recomendación por Cercanía con Filtro de Estrellas</h2>
            <form action="/" method="get">
                <label for="latitud1">Latitud:</label>
                <input type="text" id="latitud1" name="latitud1" required>
                <label for="longitud1">Longitud:</label>
                <input type="text" id="longitud1" name="longitud1" required>
                <button type="submit">Obtener Recomendaciones</button>
            </form>
            <div class="result-table" id="result_stars">
                {format_table(result_stars)}
            </div>
        </section>
        
        <footer>
            <p>&copy; 2024 Recomendación de Bares</p>
        </footer>
    </body>
    </html>
    """

# Primer modelo entrega recomendaciones por cercanía
@app.get('/r.cercania')
async def Recomendacion(latitud: float, longitud: float):
    try:
        columnasd_interes = ['latitude', 'longitude']
        data = df[columnasd_interes]

        # Creo el modelo KNN
        knn_model = NearestNeighbors(n_neighbors=10, metric='haversine')
        knn_model.fit(data)

        # Se consultan los 10 lugares más cercanos
        distances, indices = knn_model.kneighbors([[latitud, longitud]])

        # Imprimo los resultados
        nearest_places = df.iloc[indices[0]][['name', 'avg_rating', 'address', 'latitude', 'longitude']].to_dict(orient='index')

        return nearest_places
    except Exception as e:
        return {"error": str(e)}

# Segundo modelo entrega recomendaciones por cercanía luego de haber filtrado el dataset
# mediante el promedio de estrellas de los diferentes bares.
@app.get('/r.stars')
async def Recomendacion(latitud1: float, longitud1: float):
    try:
        filtered_df = df[df['avg_rating'] > 3.5]

        columnasd_interes = ['latitude', 'longitude']
        data = filtered_df[columnasd_interes]

        # Creo el modelo KNN
        knn_model = NearestNeighbors(n_neighbors=10, metric='haversine')
        knn_model.fit(data)

        # Se consultan los 10 lugares más cercanos
        distances, indices = knn_model.kneighbors([[latitud1, longitud1]])

        # Imprimo los resultados
        nearest_places1 = filtered_df.iloc[indices[0]][['name', 'avg_rating', 'address', 'latitude', 'longitude']].to_dict(orient='index')

        return nearest_places1
    except Exception as e:
        return {"error": str(e)}
