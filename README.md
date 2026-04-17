# Proyecto de Prediccion Minera en el Peru

Pipeline profesional de datos en Python para inspeccion, limpieza, validacion, transformacion, migracion a MongoDB Atlas, EDA, feature engineering y forecasting.

## Estado

- Fase 0: en implementacion (estructura y estandar operativo)
- Fase 1-8: pendientes de ejecucion por gate de aprobacion

## Requisitos

- Sistema operativo: Windows
- Python objetivo: 3.11.x
- Editor: VS Code

Nota: actualmente existe un entorno local .venv. Se recomienda recrearlo con Python 3.11 para alinearse con el estandar del proyecto.

## Setup rapido en PowerShell

1. Crear entorno virtual con Python 3.11:

```powershell
py -3.11 -m venv .venv
```

2. Activar entorno virtual:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```

3. Instalar dependencias:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

4. Verificar preparacion de Fase 0:

```powershell
python scripts/check_phase0_setup.py
```

## Estructura base

```text
.
|- data/
|  |- raw/
|  |- interim/
|  |- processed/
|- notebooks/
|- src/
|  |- ingestion/
|  |- profiling/
|  |- cleaning/
|  |- validation/
|  |- transforms/
|  |- mongodb/
|  |- eda/
|  |- features/
|  |- models/
|  |- utils/
|- scripts/
|- tests/
|- logs/
|- reports/
|- requirements.txt
|- .env.example
|- .gitignore
|- README.md
```

## Convencion de ejecucion

- Todos los scripts se ejecutan desde la raiz del repositorio.
- Los datos fuente van en data/raw.
- data/interim y data/processed se generan por pipeline.
- No hardcodear credenciales; usar variables de entorno desde .env.

## Iniciador principal (main)

Para ejecutar todo el pipeline con un solo comando, coloca el archivo fuente en:

- data/raw/DatosProduccionMinera.xlsx

Luego ejecuta:

```powershell
python main.py
```

Opciones utiles:

- Omitir MongoDB (solo pipeline local):

```powershell
python main.py --mongo-mode never
```

- Forzar MongoDB (requiere URI):

```powershell
python main.py --mongo-mode always --mongo-uri "<tu_uri_atlas>"
```

- Ejecutar sin correr fases, solo plan de pasos:

```powershell
python main.py --dry-run
```

El iniciador genera:

- logs/main_pipeline.log
- reports/main_pipeline_run_report.json

## Proximas fases

1. Fase 1: inspeccion y profiling automatizado.
2. Fase 2: limpieza automatizada.
3. Fase 3: validacion de esquema y calidad.
4. Fase 4: transformacion y clasificacion.
5. Fase 5: diseno y carga en MongoDB Atlas.
6. Fase 6: EDA.
7. Fase 7: feature engineering.
8. Fase 8: forecasting.
