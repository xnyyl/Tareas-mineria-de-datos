# Minería de datos

Este proyecto está organizado por semanas, siguiendo la estructura de referencia del repositorio base pero con nombres adaptados:

- primero
- segundo
- tercero
- cuarto

## Estructura

- `csv/`: archivos CSV de trabajo
- `primero/`: limpieza y preparación de datos
- `segundo/`: análisis estadístico descriptivo
- `tercero/`: visualización de resultados
- `cuarto/`: pruebas estadísticas

## Archivo principal

- `primero/limpieza_datos.py`: genera un CSV limpio a partir del dataset original

## Requisitos

Se recomienda usar Python 3.10+ y tener instaladas estas librerías:

- pandas
- numpy
- matplotlib
- scipy

## Ejecución

Desde la raíz del proyecto, puedes correr cada script así:

```bash
python primero/limpieza_datos.py
python segundo/estadistica_descriptiva.py
python tercero/visualizacion.py
python cuarto/pruebas_estadisticas.py
```

El archivo base se encuentra en:

- `csv/fifa_world_cup_all_matches_1930_2026.csv`
