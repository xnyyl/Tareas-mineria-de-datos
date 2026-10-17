# Minería de datos

Este proyecto está organizado por semanas, siguiendo la estructura de referencia del repositorio base pero con nombres adaptados:

- primero
- segundo
- tercero
- cuarto
- quinto
- sexto

## Estructura

- `csv/`: archivos CSV de trabajo
- `primero/`: limpieza y preparación de datos
- `segundo/`: análisis estadístico descriptivo
- `tercero/`: visualización de resultados
- `cuarto/`: pruebas estadísticas
- `quinto/`: regresiones lineales simples y múltiples
- `sexto/`: clasificación del resultado del partido con KNN

## Archivo principal

- `primero/limpieza_datos.py`: genera un CSV limpio a partir del dataset original

## Requisitos

Se recomienda usar Python 3.10+ y tener instaladas estas librerías:

- pandas
- numpy
- matplotlib
- scipy
- scikit-learn

## Ejecución

Desde la raíz del proyecto, puedes correr cada script así:

```bash
python primero/limpieza_datos.py
python segundo/estadistica_descriptiva.py
python tercero/visualizacion.py
python cuarto/pruebas_estadisticas.py
python quinto/modelos_lineales.py
python sexto/clasificacion_knn.py
```

El archivo base se encuentra en:

- `csv/fifa_world_cup_all_matches_1930_2026.csv`
