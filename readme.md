# Sistema de Mantenimiento Predictivo - MVP

Sistema de prediccion de fallas de equipos industriales con 24 horas de anticipacion utilizando Machine Learning (LightGBM).

---

## Tabla de Contenidos

- [Descripcion](#descripcion)
- [Metricas de Rendimiento](#metricas-de-rendimiento)
- [Arquitectura](#arquitectura)
- [Dashboard para Stakeholders](#dashboard-para-stakeholders)
- [Requisitos](#requisitos)
- [Instalacion](#instalacion)
- [Configuracion](#configuracion)
- [Uso](#uso)
- [Estructura del Proyecto](#estructura-del-proyecto)
- [Notebooks - Fuente de Verdad](#notebooks---fuente-de-verdad)
- [Pipeline de Datos](#pipeline-de-datos)
- [Tabla de Predicciones](#tabla-de-predicciones)
- [Niveles de Alerta](#niveles-de-alerta)
- [Archivos del Modelo](#archivos-del-modelo)
- [Monitoreo y Logs](#monitoreo-y-logs)
- [Troubleshooting](#troubleshooting)
- [Mantenimiento](#mantenimiento)

---

## Descripcion

Este sistema analiza datos de produccion y mantenimiento de equipos industriales para predecir fallas con **24 horas de anticipacion**, permitiendo:

- **Reducir paradas no planificadas** mediante mantenimiento preventivo
- **Priorizar recursos** de mantenimiento segun nivel de riesgo
- **Optimizar costos** evitando fallas catastroficas
- **Medir rendimiento** comparando predicciones vs. fallas reales

### Caracteristicas principales

| Caracteristica | Descripcion |
|----------------|-------------|
| **Modelo** | LightGBM optimizado (21 arboles, best_iteration) |
| **Features** | 47 variables predictivas (seleccionadas por importancia) |
| **Horizonte** | Prediccion a 24 horas |
| **Granularidad** | 1 prediccion por maquina (estado actual) |
| **Actualizacion** | Ejecucion diaria automatizada |
| **Validacion** | Actualizacion diaria de targets reales |
| **Dashboard** | Interfaz Streamlit para stakeholders no tecnicos |

---

## Metricas de Rendimiento

Validacion realizada sobre datos de diciembre 2025 (no vistos durante entrenamiento jul-nov 2025):

### Validacion en datos frescos (diciembre 2025)

| Metrica | Test set (nov 2025) | Diciembre 2025 (frescos) |
|---------|---------------------|--------------------------|
| AUC-ROC | 0.8028 | **0.8219** |
| Precision | 0.3536 | 0.3486 |
| Recall | 0.7903 | 0.7671 |
| F1 | 0.5052 | 0.4794 |
| Avg Precision | 0.5540 | 0.5560 |

- **Deteccion de eventos:** 1,070 de 1,113 fallas detectadas = **96.1%**
- **Anticipacion promedio:** 15-20h antes de la falla
- **Alertas CRITICAL activas:** 14,270 ventanas (8.65%) con score >= 0.45

### Periodo de Entrenamiento

| Fase | Periodo |
|------|---------|
| Datos historicos | 2025-07-01 en adelante |
| Train | Hasta 2025-09-30 |
| Test | Hasta 2025-11-30 |
| Datos frescos (validacion) | Diciembre 2025 |

---

## Arquitectura

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Base de Datos │────>│   Extraccion    │────>│    Feature      │
│   (MySQL/Azure) │     │ (7d eventos,    │     │   Engineering   │
│                 │     │  60d mant.)     │     │   (17 pasos)    │
└─────────────────┘     └─────────────────┘     └────────┬────────┘
                                                         │
                                                         v
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│    Alertas      │<────│   Clasificacion │<────│    Modelo       │
│  (Dashboard)    │     │   (Umbrales)    │     │   LightGBM      │
└─────────────────┘     └─────────────────┘     └────────┬────────┘
                                                         │
                                                         v
                        ┌─────────────────┐     ┌─────────────────┐
                        │  Update Targets │<────│    Guardar      │
                        │   (diario)      │     │   Predicciones  │
                        └─────────────────┘     └─────────────────┘
                                │                        │
                                └────────────┬───────────┘
                                             v
                                ┌─────────────────────────┐
                                │  bui_predicciones_hora   │
                                │  (tabla de resultados)   │
                                └─────────────────────────┘
```

### Componentes

| Archivo | Descripcion |
|---------|-------------|
| `main.py` | Orquestador principal del pipeline |
| `save_predictions.py` | Guarda predicciones en BD |
| `update_targets.py` | Actualiza target_real despues de 24h |
| `dashboard_mvp.py` | Dashboard Streamlit para stakeholders |
| `src/db_connector.py` | Conexion y extraccion de datos desde MySQL/Azure |
| `src/feature_engineering.py` | Transformacion de datos raw a 48 features (17 pasos) |
| `src/predictor.py` | Carga del modelo y generacion de predicciones |
| `src/pipeline_utils.py` | Funciones auxiliares de pipeline (carga de features, transformaciones) |
| `src/utils.py` | Funciones auxiliares (logging, alertas) |
| `src/config_loader.py` | Carga configuracion y credenciales desde .env |

---

## Dashboard para Stakeholders

Dashboard interactivo (`dashboard_mvp.py`) orientado a usuarios no tecnicos para demostrar la capacidad predictiva del modelo.

### Ejecutar

```bash
python -m streamlit run dashboard_mvp.py
```

### Modos de carga de datos

| Modo | Descripcion | Cuando usar |
|------|-------------|-------------|
| **Cargar archivo** | Sube CSV/Parquet con datos preprocesados | Pruebas y demos sin acceso a BD |
| **Base de Datos** | Conecta a MySQL via .env | Produccion con credenciales |

### Tipos de archivo aceptados

**Resultados ya inferidos** (ej. `resultados_inferencia.parquet`):
- Columnas: `probabilidad_falla`, `prediccion_binaria`, `nivel_alerta`, `target_real`
- Se detecta automaticamente y se visualiza directo

**Dataset con features** (ej. `dataset_features_final.parquet`):
- Contiene las 48 features + `timestamp_hora` + `id_maquina_dfos`
- Se detecta automaticamente, ejecuta el modelo y muestra resultados

### Tabs del Dashboard

| Tab | Contenido |
|-----|-----------|
| **Resumen Ejecutivo** | KPIs (maquinas, alertas, deteccion), tabla de estado por maquina con semaforo, Top 10 riesgo |
| **Detalle por Maquina** | Selector de maquina, timeline de riesgo, tabla de acciones recomendadas, exportar Excel |
| **Validacion del Modelo** | Solo si hay target. Deteccion de fallas, fiabilidad, matriz de confusion visual, histograma de scores |

### Lenguaje no tecnico

El dashboard traduce conceptos tecnicos a lenguaje de negocio:

| Tecnico | En el Dashboard |
|---------|-----------------|
| Score/Probability | Nivel de riesgo |
| Precision | Fiabilidad de las alertas |
| Recall | Fallas detectadas a tiempo |
| False Positive | Falsa alarma |
| False Negative | Falla no detectada |
| Inference | Analisis predictivo |
| Feature | Indicador |

---

## Requisitos

### Sistema

- Python 3.10+
- Acceso a base de datos MySQL/Azure (para modo BD)
- 4GB RAM minimo (8GB recomendado)

### Dependencias principales

```
numpy==2.0.2
pandas==2.3.3
scikit-learn>=1.6.1
lightgbm>=4.6.0
sqlalchemy>=2.0.0
pymysql>=1.1.0
streamlit>=1.54.0
plotly>=6.5.2
openpyxl>=3.1.5
python-dotenv>=1.2.1
```

---

## Instalacion

### 1. Clonar repositorio

```bash
git clone <repository-url>
cd perdidasDiario
```

### 2. Crear entorno virtual

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# o
venv\Scripts\activate     # Windows
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 4. Crear tabla de predicciones

Ejecutar el DDL en la base de datos:

```bash
mysql -h your-host -u your-user -p your_database < scripts/ddl_bui_predicciones_hora.sql
```

### 5. Verificar archivos del modelo

Asegurarse de que existen los siguientes archivos en `model/`:

```
model/
├── model_optimized.pkl           # Modelo LightGBM
├── features_utiles.json          # Lista de 48 features
├── inference_config.json         # Umbrales y parametros
└── preprocessing_artifacts.pkl   # Artefactos de preprocesamiento
```

---

## Configuracion

### Variables de entorno (.env)

```bash
# Base de datos produccion
DB_PROD_HOST=your-server.mysql.database.azure.com
DB_PROD_USER=your_user
DB_PROD_PASSWORD=your_password
DB_PROD_NAME=your_database

# Base de datos desarrollo (opcional)
DB_DEV_HOST=localhost
DB_DEV_USER=dev_user
DB_DEV_PASSWORD=dev_password
DB_DEV_NAME=dev_database
```

### config.yaml

Configuracion central del sistema. Soporta variables de entorno con sintaxis `${VAR:default}` (usa el default si la variable no existe, permitiendo ejecutar sin `.env`).

```yaml
# Base de datos (source=lectura PROD, target=escritura DEV)
database:
  source:
    host: ${DB_PROD_HOST:localhost}
    port: ${DB_PROD_PORT:3306}
    database: ${DB_PROD_NAME:bui_production}
    user: ${DB_PROD_USER:readonly_user}
    password: ${DB_PROD_PASSWORD:CHANGE_ME}
  target:
    host: ${DB_DEV_HOST:localhost}
    ...

# Modelo
model:
  path: "./model/model_optimized.pkl"
  version: "v1.0"

# Umbrales (referencia - la fuente de verdad es inference_config.json)
thresholds:
  production: 0.3724709494525155
  alertas:
    critical: 0.7
    moderate: 0.31
    low: 0.10

# Ejecucion
execution:
  lookback_hours: 168              # 7 dias para bui_perdida
  lookback_days_maintenance: 60    # 60 dias para bui_pm_ewo
  timeout_minutes: 50
  fecha_referencia: null            # null=now(), o '2025-07-25' para testing
  fabrica_id: 21                    # Veszprem
```

---

## Uso

### Ejecucion del pipeline (produccion)

```bash
# Ejecutar predicciones (extrae, procesa, predice y guarda en BD)
python main.py

# Actualizar targets reales (despues de 24h)
python update_targets.py
```

### Dashboard para demos

```bash
# Lanzar dashboard Streamlit
python -m streamlit run dashboard_mvp.py

# En puerto especifico
python -m streamlit run dashboard_mvp.py --server.port 8502
```

### Ejecucion con fecha especifica (modo test)

Editar `config.yaml`:

```yaml
execution:
  fecha_referencia: '2025-07-25'
```

### Configurar ejecucion automatica (cron)

```bash
# Predicciones 1 vez al dia
0 6 * * * cd /path/to/perdidasDiario && /path/to/venv/bin/python main.py >> logs/cron.log 2>&1

# Actualizar targets 1 vez al dia (23:00)
0 23 * * * cd /path/to/perdidasDiario && /path/to/venv/bin/python update_targets.py >> logs/cron.log 2>&1
```

---

## Estructura del Proyecto

```
perdidasDiario/
│
├── .env                             # Credenciales (no commitear)
├── .streamlit/
│   └── config.toml                  # Config Streamlit (maxUploadSize)
├── main.py                          # Orquestador principal del pipeline
├── save_predictions.py              # Guarda predicciones en BD
├── update_targets.py                # Actualiza target_real y acierto
├── dashboard_mvp.py                 # Dashboard Streamlit para stakeholders
├── config.yaml                      # Configuracion central (BD, umbrales, ejecucion)
├── requirements.txt                 # Dependencias Python
├── readme.md                        # Esta documentacion
│
├── src/
│   ├── __init__.py
│   ├── config_loader.py             # Carga config.yaml con soporte ${VAR:default}
│   ├── db_connector.py              # Conexion y extraccion de BD
│   ├── feature_engineering.py       # Generacion de 48 features (17 pasos)
│   ├── predictor.py                 # Carga modelo y prediccion
│   ├── pipeline_utils.py            # Utilidades de pipeline (carga features, transformaciones)
│   └── utils.py                     # Funciones auxiliares
│
├── model/
│   ├── model_optimized.pkl          # Modelo LightGBM (pickle)
│   ├── features_utiles.json         # Lista de 48 features ordenadas
│   ├── inference_config.json        # Umbrales, metricas, hiperparametros
│   └── preprocessing_artifacts.pkl  # LabelEncoders, stats, thresholds
│
├── notebooks/
│   ├── pipeline_reentrenamiento.ipynb        # Fuente de verdad: entrenamiento
│   └── pipeline_inferencia_nuevos_datos.ipynb # Fuente de verdad: inferencia
│
├── logs/
│   ├── predictions_YYYY-MM-DD.log
│   └── errors.log
│
└── scripts/
    ├── ddl_bui_predicciones_hora.sql  # DDL para crear tabla en BD
    └── setup_cron.sh                  # Configura ejecucion automatica (Linux/cron)
```

---

## Notebooks - Fuente de Verdad

Los notebooks son la **fuente de verdad** del proyecto. El codigo en `src/` debe ser consistente con ellos.

| Notebook | Proposito |
|----------|-----------|
| `pipeline_reentrenamiento.ipynb` | Pipeline completo de reentrenamiento: ETL, feature engineering, optimizacion Optuna, evaluacion, exportacion de artifacts |
| `pipeline_inferencia_nuevos_datos.ipynb` | Pipeline de inferencia con datos nuevos: carga dataset, aplica modelo, evalua metricas, genera resultados |

### Artefactos generados por los notebooks

Los notebooks generan los archivos en `model/` que usa el sistema en produccion:
- `model_optimized.pkl` - Modelo entrenado
- `features_utiles.json` - 48 features seleccionadas (orden importa)
- `inference_config.json` - Umbrales, metricas referencia, hiperparametros
- `preprocessing_artifacts.pkl` - Encoders, estadisticas, thresholds

---

## Pipeline de Datos

### 1. Extraccion (db_connector.py)

Extrae datos de 3 tablas principales:

| Tabla | Descripcion | Ventana |
|-------|-------------|---------|
| `bui_perdida` | Eventos de perdida/produccion | 7 dias (168 horas) |
| `bui_pm_ewo` | Ordenes de mantenimiento | 60 dias |
| `bui_line` | Metadata de lineas | Todas |

### 2. Feature Engineering (feature_engineering.py)

Genera features en 17 pasos, seleccionando las 47 mas importantes:

| Paso | Descripcion |
|------|-------------|
| 1-4 | Pivoteo y agregacion base |
| 5 | Features detalladas por evento |
| 6 | Metricas agregadas |
| 7 | Features de mantenimiento |
| 8 | Rolling windows (2h, 6h, 12h, 24h) |
| 9 | Tendencias y aceleraciones |
| 10 | Ratios y proporciones |
| 11 | Features temporales |
| 12 | Variabilidad (std, cv) |
| 13 | Encoding categorias |
| 14 | Z-scores y anomalias |
| 15 | Comparacion con pares |
| 16 | Target y limpieza |
| 17 | Filtrar ultima ventana por maquina (1 prediccion por maquina) |

### 3. Prediccion (predictor.py)

- Carga modelo LightGBM desde `model_optimized.pkl`
- Valida presencia de 47 features en orden correcto (desde `features_utiles.json`)
- Genera scores de probabilidad (0-1)
- Clasifica en niveles de alerta segun umbrales de `inference_config.json`

### 4. Guardado (save_predictions.py)

- Inserta predicciones en `bui_predicciones_hora`
- Calcula `fl_pred_modelo` (1 si score >= 0.3724)
- Maneja duplicados con INSERT IGNORE

### 5. Actualizacion de Targets (update_targets.py)

- Se ejecuta 1 vez al dia
- Busca predicciones donde ya pasaron 24h
- Verifica en `bui_perdida` si hubo falla grave
- Actualiza `fl_target_real` y `fl_acierto`

---

## Tabla de Predicciones

### Estructura: `bui_predicciones_hora`

| Campo | Tipo | Descripcion |
|-------|------|-------------|
| `id_prediccion` | INT AUTO_INCREMENT | PK |
| `id_maquina_dfos` | VARCHAR(50) | ID de maquina |
| `id_linea` | INT | ID de linea de produccion |
| `fe_ventana` | DATETIME | Ventana horaria predicha |
| `nm_score` | DECIMAL(5,4) | Score del modelo (0.0000-1.0000) |
| `de_nivel_riesgo` | VARCHAR(20) | critico/moderado/bajo/normal |
| `fl_pred_modelo` | TINYINT | 1 si score >= 0.3724, 0 si no |
| `fl_target_real` | TINYINT NULL | NULL=pendiente, 0=no falla, 1=falla |
| `fl_acierto` | TINYINT NULL | NULL=pendiente, 0=error, 1=acierto |
| `de_modelo_version` | VARCHAR(20) | Version del modelo |
| `fe_creado` | TIMESTAMP | Fecha de insercion (auto) |
| `fe_actualizado` | TIMESTAMP NULL | Fecha de actualizacion |

### Criterio de Falla Grave (fl_target_real = 1)

Una falla se considera **grave** si cumple:

**Base:** `de_perdida_2 = 'Breakdown & Equipment Failure Time'`

**Y al menos uno de:**

| # | Criterio | Condicion |
|---|----------|-----------|
| 1 | Con EWO correctivo valido | duracion >= 10 min |
| 2 | Duracion prolongada | duracion >= 60 min |
| 3 | Tipo critico (Mechanical/Electrical/I&C) | duracion >= 20 min |

### Consultas utiles

```sql
-- Predicciones pendientes de actualizar
SELECT * FROM bui_predicciones_hora
WHERE fl_target_real IS NULL
  AND fe_ventana < NOW() - INTERVAL 24 HOUR;

-- Matriz de confusion
SELECT
    fl_pred_modelo,
    fl_target_real,
    COUNT(*) as cantidad
FROM bui_predicciones_hora
WHERE fl_target_real IS NOT NULL
GROUP BY fl_pred_modelo, fl_target_real;

-- Metricas de precision
SELECT
    SUM(CASE WHEN fl_pred_modelo = 1 AND fl_target_real = 1 THEN 1 ELSE 0 END) as TP,
    SUM(CASE WHEN fl_pred_modelo = 1 AND fl_target_real = 0 THEN 1 ELSE 0 END) as FP,
    SUM(CASE WHEN fl_pred_modelo = 0 AND fl_target_real = 1 THEN 1 ELSE 0 END) as FN,
    SUM(CASE WHEN fl_pred_modelo = 0 AND fl_target_real = 0 THEN 1 ELSE 0 END) as TN
FROM bui_predicciones_hora
WHERE fl_target_real IS NOT NULL;
```

---

## Niveles de Alerta

| Nivel | Umbral | Accion recomendada |
|-------|--------|-------------------|
| **CRITICO** | Score >= 0.45 | Mantenimiento inmediato, coordinar con produccion |
| **MODERADO** | 0.31 - 0.45 | Inspeccion prioritaria en 24h |
| **BAJO** | 0.10 - 0.31 | Monitoreo intensivo, verificar en proximo turno |
| **NORMAL** | < 0.10 | Operacion normal, mantenimiento segun plan |

**Umbral de produccion:** 0.31 (punto optimo recall/precision en test set)

---

## Archivos del Modelo

### model_optimized.pkl

Modelo LightGBM serializado con pickle.

```python
import pickle
with open('model/model_optimized.pkl', 'rb') as f:
    model = pickle.load(f)
```

### features_utiles.json

Lista de las 47 features requeridas por el modelo (seleccionadas por importancia > 0). **El orden es critico** para la correcta inferencia.

```json
{
  "features": [
    "concurrent_breakdown_equipment_failure_area_24h",
    "breakdown_equipment_failure_zscore",
    "breakdown_equipment_failure_cv_7d",
    "..."
  ],
  "n_features": 47
}
```

### inference_config.json

Configuracion completa de umbrales, metricas referencia e hiperparametros.

```json
{
  "umbrales": {
    "produccion": 0.31,
    "alertas": {
      "critical": 0.45,
      "moderate": 0.31,
      "low": 0.10
    }
  },
  "modelo": {
    "algoritmo": "LightGBM",
    "archivo": "model_optimized.pkl",
    "n_features": 47,
    "best_iteration": 21
  },
  "fechas_entrenamiento": {
    "inicio_historico": "2025-07-01",
    "fecha_fin_train": "2025-09-30",
    "fecha_fin_test": "2025-11-30"
  }
}
```

### preprocessing_artifacts.pkl

Artefactos para garantizar consistencia entre entrenamiento e inferencia:

| Artefacto | Proposito |
|-----------|-----------|
| `label_encoders` | LabelEncoders para `area` y `fabrica` + mappings |
| `onehot_columns` | Columnas one-hot fijas de subcategorias |
| `thresholds` | Umbral alto riesgo (0.256) y breakdown rates |
| `stats_por_grupo` | Media/Std por maquina para Z-scores |
| `defaults` | Valores default para features sin historial |
| `features_esperadas` | Lista completa de 264 features intermedias |

---

## Monitoreo y Logs

### Estructura de logs

```
logs/
├── predictions_2025-12-05.log   # Log diario de predicciones
├── errors.log                    # Errores del sistema
└── cron.log                      # Salida del cron job
```

### Formato de log

```
2025-12-05 06:00:15 - INFO - INICIANDO PREDICCION DIARIA
2025-12-05 06:00:16 - INFO - Conexion a BD establecida
2025-12-05 06:05:23 - INFO - Feature engineering completado: 830 maquinas
2025-12-05 06:05:24 - INFO - Distribucion de alertas:
2025-12-05 06:05:24 - INFO -    MODERADO: 150 (18.1%)
2025-12-05 06:05:24 - INFO -    BAJO: 480 (57.8%)
2025-12-05 06:05:24 - INFO -    NORMAL: 200 (24.1%)
2025-12-05 06:05:25 - INFO - Predicciones guardadas: 830 registros
```

### Monitorear en tiempo real

```bash
tail -f logs/predictions_$(date +%Y-%m-%d).log
```

---

## Troubleshooting

### Error: "Features faltantes"

**Causa**: El feature engineering no genero todas las 48 features.

**Solucion**:
1. Verificar que `preprocessing_artifacts.pkl` existe
2. Verificar datos de entrada (minimo 7 dias)
3. Revisar logs para identificar paso fallido

### Error: "Conexion a BD fallida"

**Causa**: Credenciales incorrectas o servidor no accesible.

**Solucion**:
1. Verificar `.env` con las variables DB_PROD_*
2. Probar conexion manualmente: `mysql -h host -u user -p`
3. Verificar firewall/VPN si aplica

### Error: "Duplicate entry"

**Causa**: Prediccion duplicada para misma maquina y ventana.

**Solucion**: Normal si se re-ejecuta para la misma hora. El sistema usa INSERT IGNORE.

### Error: "Score maximo muy bajo" / Sin alertas criticas

**Causa**: El modelo no generaliza a datos con distribucion diferente (data drift).

**Solucion**:
1. Verificar que se usa `preprocessing_artifacts.pkl` correcto
2. Comparar distribucion de features vs entrenamiento
3. Si persiste, reentrenar modelo con datos mas recientes

### Warning: "sklearn version mismatch"

**Causa**: Version de sklearn diferente entre entrenamiento e inferencia.

**Solucion**:
1. Instalar misma version: `pip install scikit-learn==1.8.0`
2. O regenerar artifacts con version actual

---

## Mantenimiento

### Actualizacion del modelo

1. Ejecutar notebook `pipeline_reentrenamiento.ipynb` con datos actualizados
2. Los artifacts se generan automaticamente:
   - `model_optimized.pkl`
   - `features_utiles.json`
   - `inference_config.json`
   - `preprocessing_artifacts.pkl`
3. Copiar archivos generados a `model/`
4. Validar con `pipeline_inferencia_nuevos_datos.ipynb`
5. Verificar metricas en el dashboard (Tab Validacion)
6. Desplegar

### Re-entrenamiento recomendado

- **Frecuencia**: Cada 3-6 meses o cuando hay data drift
- **Trigger**: Si recall cae por debajo del 60% o max score no supera 0.5
- **Datos**: Minimo 6 meses de historia

### Monitoreo de drift

Verificar periodicamente:
- Distribucion de scores (debe mantenerse estable)
- Ratio de alertas por nivel
- Precision/Recall en fallas conocidas

### Consulta de rendimiento historico

```sql
SELECT
    DATE(fe_creado) as fecha,
    COUNT(*) as total_predicciones,
    SUM(fl_target_real) as fallas_reales,
    SUM(fl_acierto) as aciertos,
    ROUND(AVG(fl_acierto) * 100, 2) as accuracy_pct
FROM bui_predicciones_hora
WHERE fl_target_real IS NOT NULL
GROUP BY DATE(fe_creado)
ORDER BY fecha DESC
LIMIT 30;
```

---

## Changelog

### v1.3 (2026-02)

#### Modelo re-entrenado post-fix (2026-02-27)

| Metrica | Valor |
|---------|-------|
| AUC-ROC test | 0.803 |
| Avg Precision test | 0.554 |
| Features usadas | 47 |
| Best iteration | 21 |
| Umbral produccion | 0.31 (recall 79%) |

Artefacto `stats_por_maquina`: claves `(id_linea, id_maquina_dfos)` — 885 combinaciones unicas.

#### Umbrales de alerta (`inference_config.json`)

| Nivel | Umbral | Precision | Recall |
|-------|--------|-----------|--------|
| CRITICAL | >= 0.45 | 68.7% | 27.4% |
| MODERATE | >= 0.31 | 35.4% | 79.0% |
| LOW | >= 0.10 | — | — |

#### Fix: Aislamiento de maquinas por fabrica+linea en feature engineering

**Problema detectado (v2):** Dentro de la misma fabrica, el mismo `id_maquina_dfos` puede
existir en distintas lineas de produccion (`id_linea`). El agrupamiento por solo
`['id_fabrica', 'id_maquina_dfos']` mezclaba datos de maquinas homonimas de lineas distintas.

**Clave compuesta correcta:** `['id_fabrica', 'id_linea', 'id_maquina_dfos']`

**Problema detectado (v1):** Las operaciones agrupaban por `id_maquina_dfos` solo,
mezclando maquinas con el mismo nombre en fabricas distintas.

**Archivos corregidos:**

**`src/feature_engineering.py`**
- Paso 8 (rolling windows 2h/6h/12h/24h): `groupby('id_maquina_dfos')` → `groupby(['id_fabrica', 'id_linea', 'id_maquina_dfos'])`
- Paso 9 (tendencias/aceleraciones): `.shift(12)` ahora agrupa por fabrica+linea+maquina
- Paso 12 (variabilidad std/cv 7 dias): mismo fix en transform de std y mean
- Paso 12 (horas_desde_ultimo_breakdown): loop itera por `(fabrica, linea, maquina)`
- Paso 17 (ultima ventana por maquina): `idxmax()` agrupa por fabrica+linea+maquina

```python
# ANTES - mezclaba maquinas de distintas fabricas con mismo nombre
df.groupby('id_maquina_dfos')[evento].transform(...)

# DESPUES - cada maquina se trata de forma aislada por fabrica+linea+maquina
df.groupby(['id_fabrica', 'id_linea', 'id_maquina_dfos'])[evento].transform(...)
```

**`pipeline_reentrenamiento.ipynb`** (celdas 14, 17, 21, 22)
- Celda 14 (pivoteo): `maquinas_unicas` ahora usa combinaciones unicas de
  `['id_fabrica', 'id_linea', 'id_maquina_dfos']`; el filtrado de chunks usa merge
  por las 3 columnas; el loop interno itera sobre `(fabrica, linea, maquina)` en lugar
  de solo `maquina`
- Celda 17 (target): loop de marcado de fallas y exclusion de primeras horas ahora
  itera por `(fabrica, linea, maquina)` — evita que fallas de MAQUINA_A/linea_7 marquen
  erroneamente ventanas de MAQUINA_A/linea_5 como target=1
- Celda 21: sort, rolling windows, shift, std/cv 7 dias y loop de horas_desde_ultimo_breakdown
  actualizados a clave compuesta de 3 columnas
- Celda 22: z-scores (anomalias) actualizados a clave compuesta de 3 columnas

**`pipeline_inferencia_nuevos_datos.ipynb`** (celdas 15, 24, 26)

- Celda 15 (pivoteo): misma correccion que reentrenamiento celda 14 — `maquinas_unicas`
  ahora usa combinaciones unicas de `['id_fabrica', 'id_linea', 'id_maquina_dfos']`,
  filtrado de chunks via merge por las 3 columnas, loop interno itera sobre `(fabrica, linea, maquina)`
- Celda 24: rolling windows, shift (tendencias), std/cv 7 dias y loop de
  horas_desde_ultimo_breakdown actualizados a clave compuesta de 3 columnas
- Celda 26: z-scores (anomalias) actualizados a `groupby(['id_fabrica', 'id_linea', 'id_maquina_dfos'])`

**`pipeline_reentrenamiento.ipynb`** (celda 26 — artifacts)

- `stats_por_maquina` ahora agrupa por `['id_linea', 'id_maquina_dfos']`, generando
  claves `(id_linea, 'MAQUINA_A')` en lugar de solo `'MAQUINA_A'`
- Coordinado con el lookup en `feature_engineering.py` PASO 13 que ahora usa
  `key = (id_linea, maquina)` para buscar estadisticas por maquina en los artifacts

**Nota:** En produccion (`main.py`) el riesgo ya estaba mitigado porque se filtra a
fabrica 21 (Veszprem) antes del feature engineering. El fix es critico para
reentrenamiento con datos multi-fabrica y multi-linea.

#### Fix: Errores en generacion de preprocessing artifacts (notebook entrenamiento)
- `stats_por_maquina` (z-scores): revertido a `groupby('id_maquina_dfos')` porque
  `id_fabrica` no esta disponible en `dataset_features_final.parquet` (fue eliminado en FE)
- Threshold `is_high_risk_subcategoria`: corregido para usar `df_final` en lugar de
  `df_con_target` (que no tiene columnas `subcategoria_*`)
- Fix instalacion Optuna (celda 31): reemplazado `python_exe = r"c:\Python39\python.exe"`
  por `sys.executable` para usar el entorno Python activo

#### Validacion en datos frescos (diciembre 2025)

Ejecutado `pipeline_inferencia_nuevos_datos.ipynb` con datos de diciembre 2025 (nunca vistos):

| Metrica | Test set | Diciembre 2025 |
|---------|----------|----------------|
| AUC-ROC | 0.8028 | **0.8219** |
| Precision | 0.3536 | 0.3486 |
| Recall | 0.7903 | 0.7671 |
| F1 | 0.5052 | 0.4794 |

- Deteccion de eventos: 1,070 / 1,113 = **96.1%**
- Alertas CRITICAL activas (>= 0.45): 14,270 ventanas (8.65%)
- Anticipacion tipica: 15-24h antes de la falla

Fix celda 37 (`pipeline_inferencia_nuevos_datos.ipynb`): eliminadas advertencias de
diagnostico heredadas (chequeo residual contra umbral 0.70 y falso positivo
"recall sospechoso > 90%"). Los mensajes ahora reflejan el comportamiento correcto
del modelo con los umbrales actualizados.

#### Estructura de carpetas para reentrenamiento
- `data/raw/` — input: `bui_perdida.parquet`, `bui_pm_ewo.parquet`, `bui_line.parquet`
- `data/interim/` — generado: dataset pivoteado y con target
- `data/processed/` — generado: features final, artifacts, splits
- `models/` — salidas del reentrenamiento (separado de `model/` produccion)
- `outputs/plots/` — graficos generados por el notebook

### v1.2 (2026-02)
- Dashboard MVP (`dashboard_mvp.py`) para stakeholders no tecnicos
- 3 tabs: Resumen Ejecutivo, Detalle por Maquina, Validacion del Modelo
- Carga de datos via archivo (CSV/Parquet) con deteccion automatica
- Modo BD preparado (requiere .env con credenciales)
- Tabla compacta de estado de maquinas con semaforo y acciones
- Matriz de confusion visual con porcentajes
- Exportar resultados a Excel
- `config.yaml` reestructurado: DB source/target, umbrales consistentes, sintaxis `${VAR:default}`
- `config_loader.py` mejorado: soporte defaults, no crashea sin `.env`
- `scripts/setup_cron.sh` recreado: auto-detecta directorio y Python, instala/quita/verifica cron jobs
- Limpieza: eliminados scripts obsoletos, tests vacios, .bat con rutas hardcodeadas
- Fix `update_targets.py`: ventana temporal corregida (±1h → proximas 24h), criterios de severidad
  replicados del notebook (EWO>=10min, duracion>=60min, critico>=20min), `ewos_validas` ahora se usa
- Fix `save_predictions.py`: usa `config_loader` para resolver variables de entorno,
  manejo de `id_linea` nulo (NaN → 0)
- Fix: Step 17 feature engineering - 1 prediccion por maquina (era 24)
- Fix: `pipeline_utils.py` - agregada funcion `cargar_features_ordenadas()`
- Fix: `inference_config.json` - referencia a `.pkl` (era `.txt`)
- Fix: `main.py` - llamada a `generar_predicciones` simplificada
- Regenerado `requirements.txt` (encoding UTF-8, agregado streamlit/plotly/openpyxl)

### v1.1 (2025-12-06)
- `save_predictions.py` para guardar predicciones en BD
- `update_targets.py` para actualizar target_real
- Nueva tabla `bui_predicciones_hora` para tracking
- Soporte para fecha_referencia configurable (modo test)
- Extraccion de 60 dias para bui_pm_ewo (antes 7 dias)

### v1.0 (2025-12-04)
- Release inicial
- Modelo LightGBM con 48 features
- Pipeline completo de extraccion a prediccion
