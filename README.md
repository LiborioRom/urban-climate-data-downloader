# Urban Climate Data Downloader

Aplicacion local para definir una o varias zonas de estudio, interpretar solicitudes mediante un modelo Ollama, descargar variables geoespaciales y meteorologicas y generar datasets listos para analisis. Incluye el flujo de downscaling Sentinel-3 a resolucion Landsat.

Este repositorio no contiene modelos de prediccion espaciotemporal de LST. Esa investigacion vive en `urban-lst-spatiotemporal-forecasting`.

## Funcionalidades

- interfaz Streamlit con conversacion y formulario editable;
- seleccion y ajuste de multiples ROI sobre mapa;
- catalogo explicable de variables;
- descargas Earth Engine, Copernicus Data Space, Sentinel Hub y CDS;
- ERA5-Land, Landsat, Sentinel-2, Sentinel-3 y variables urbanas;
- datasets unificados e informe de calidad;
- traduccion y downscaling Sentinel-3/Landsat;
- trabajos de descarga en segundo plano con progreso persistente.

## Instalacion local

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Instala Ollama y descarga el modelo local:

```powershell
ollama pull qwen3:4b
```

Copia `.env.example` como `.env` o `.streamlit/secrets.toml.example` como `.streamlit/secrets.toml` y completa las credenciales localmente. Esos archivos no deben subirse a Git.

Inicia la aplicacion:

```powershell
streamlit run app.py
```

## Notebooks auxiliares

- `01_descarga_datos_local_documentado.ipynb`: descarga local reproducible.
- `02_downscaling_xgboost.ipynb`: baseline de downscaling.
- `03_validacion_era5_cds_vs_gee.ipynb`: comparacion de ERA5-Land.
- `04_diagnostico_sentinel3.ipynb`: diagnostico de Sentinel-3.
- `05_traduccion_sentinel3_landsat.ipynb`: traduccion entre sensores.
- `06_downscaling_multizona.ipynb`: entrenamiento y aplicacion multizona.

## Datos y resultados

No se incluyen en Git. La aplicacion genera principalmente:

```text
runs/
data_downloads/
artifacts/
cache/
```

Transfiere estas carpetas por almacenamiento externo cuando necesites conservar ejecuciones completas.

## Credenciales

Las credenciales se leen desde el entorno o desde secretos locales de Streamlit. Nunca deben escribirse directamente en notebooks, commits, informes o mensajes del chatbot.

