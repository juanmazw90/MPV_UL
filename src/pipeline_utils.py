"""
================================================================================
UTILIDADES DEL PIPELINE DE REENTRENAMIENTO
================================================================================
Funciones auxiliares para procesamiento, feature engineering y evaluación.

================================================================================
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# CARGA DE FEATURES DEL MODELO
# ============================================================================

def cargar_features_ordenadas():
    """
    Carga la lista de features en el ORDEN CORRECTO desde features_utiles.json.

    CRÍTICO: LightGBM requiere que las features se pasen en el mismo orden
    en que fue entrenado. Esta función garantiza ese orden.

    Returns:
        list: Lista ordenada de nombres de features.
    """
    posibles_rutas = [
        Path('models/inference_export/features_utiles.json'),
        Path('model/features_utiles.json'),
        Path(__file__).parent.parent / 'model' / 'features_utiles.json',
        Path(__file__).parent.parent / 'models' / 'inference_export' / 'features_utiles.json',
    ]

    for ruta in posibles_rutas:
        if ruta.exists():
            with open(ruta, 'r') as f:
                data = json.load(f)
            features = data['features']
            print(f"   Features cargadas desde: {ruta} ({len(features)} features)")
            return features

    rutas_str = "\n".join(f"     - {r}" for r in posibles_rutas)
    raise FileNotFoundError(
        f"features_utiles.json no encontrado en ninguna ruta:\n{rutas_str}"
    )


# ============================================================================
# FUNCIONES DE TRANSFORMACIÓN - BUI_PERDIDA
# ============================================================================

def calcular_variables_temporales(df):
    """Crea variables derivadas de fechas."""
    print("   ⏰ Calculando variables temporales...")
    
    df['duracion_minutos'] = (df['fe_fin'] - df['fe_inicio']).dt.total_seconds() / 60
    df['hora_inicio'] = df['fe_inicio'].dt.hour
    df['dia_semana'] = df['fe_inicio'].dt.dayofweek
    df['semana_año'] = df['fe_inicio'].dt.isocalendar().week
    df['mes'] = df['fe_inicio'].dt.month
    df['trimestre'] = df['fe_inicio'].dt.quarter
    
    def asignar_turno(hora):
        if 6 <= hora < 14:
            return 'MAÑANA'
        elif 14 <= hora < 22:
            return 'TARDE'
        else:
            return 'NOCHE'
    
    df['turno'] = df['hora_inicio'].apply(asignar_turno).astype('category')
    df['es_fin_semana'] = df['dia_semana'].isin([5, 6])
    df['retraso_registro_min'] = (df['fe_creado'] - df['fe_inicio']).dt.total_seconds() / 60
    
    return df


def imputar_duraciones_nulas(df):
    """Imputa fe_fin nulos."""
    print("   🔧 Imputando duraciones nulas...")
    
    nulos_inicial = df['fe_fin'].isna().sum()
    if nulos_inicial > 0:
        duracion_mediana = df.groupby('de_perdida_1')['duracion_minutos'].median()
        
        def imputar_fe_fin(row):
            if pd.isna(row['fe_fin']):
                duracion = duracion_mediana.get(row['de_perdida_1'], 10)
                return row['fe_inicio'] + pd.Timedelta(minutes=duracion)
            return row['fe_fin']
        
        df['fe_fin'] = df.apply(imputar_fe_fin, axis=1)
        df['duracion_minutos'] = (df['fe_fin'] - df['fe_inicio']).dt.total_seconds() / 60
    
    return df


def filtrar_outliers_duracion(df, max_dias=7):
    """Filtra duraciones extremas."""
    print("   🔍 Filtrando outliers de duración...")
    
    max_minutos = max_dias * 24 * 60
    outliers = df['duracion_minutos'] > max_minutos
    df = df[~outliers].copy()
    
    return df


def categorizar_wop(df):
    """Categoriza nm_wop."""
    print("   📊 Categorizando nm_wop...")
    
    def asignar_categoria_wop(wop):
        if pd.isna(wop) or wop == 100.0:
            return 'SIN_PRIORIDAD'
        elif wop <= 10:
            return 'PRIORIDAD_CRITICA'
        elif wop <= 25:
            return 'PRIORIDAD_ALTA'
        elif wop <= 50:
            return 'PRIORIDAD_MEDIA'
        else:
            return 'PRIORIDAD_BAJA'
    
    df['categoria_wop'] = df['nm_wop'].apply(asignar_categoria_wop).astype('category')
    return df


def corregir_clasificacion_run(df):
    """Corrige clasificación incorrecta de RUN."""
    print("   🔧 Corrigiendo clasificación RUN...")
    
    mask_run_incorrecto = (
        (df['de_perdida_1'] == 'RUN') &
        (df['duracion_minutos'] > 5)
    )
    
    df.loc[mask_run_incorrecto, 'de_perdida_1'] = 'DESCONOCIDO'
    df.loc[mask_run_incorrecto, 'de_perdida_2'] = 'DESCONOCIDO'
    df.loc[mask_run_incorrecto, 'de_perdida_3'] = 'DESCONOCIDO'
    
    return df


def imputar_jerarquia_perdidas(df):
    """Imputa jerarquía de pérdidas nulas."""
    print("   🔧 Imputando jerarquía de pérdidas...")
    
    # Imputación de nivel 1
    mask_nivel1_nulo = df['de_perdida_1'].isna()
    df.loc[mask_nivel1_nulo, 'de_perdida_1'] = 'DESCONOCIDO'
    
    # Imputación de nivel 2
    mask_nivel2_nulo = df['de_perdida_2'].isna()
    df.loc[mask_nivel2_nulo, 'de_perdida_2'] = 'DESCONOCIDO'
    
    # Imputación de nivel 3
    mask_nivel3_nulo = df['de_perdida_3'].isna()
    df.loc[mask_nivel3_nulo, 'de_perdida_3'] = 'DESCONOCIDO'
    
    return df


def imputar_velocidad(df):
    """Imputa velocidad faltante con mediana por línea."""
    print("   🔧 Imputando velocidad...")
    
    velocidad_mediana_linea = df.groupby('id_linea')['nm_velocidad'].median()
    df['nm_velocidad'] = df.apply(
        lambda row: velocidad_mediana_linea.get(row['id_linea'], 0) 
        if pd.isna(row['nm_velocidad']) else row['nm_velocidad'],
        axis=1
    )
    
    return df


def crear_variables_maquina(df):
    """Crea variables relacionadas con máquinas."""
    print("   🔧 Creando variables de máquina...")
    
    # Flags de existencia
    df['tiene_maquina_dfos'] = df['id_maquina_dfos'].notna()
    df['tiene_maquina_causal'] = df['id_maquina_causal_dfos'].notna()
    
    # Imputar nulos
    df['id_maquina_dfos'] = df['id_maquina_dfos'].fillna('SIN_MAQUINA')
    df['de_maquina_dfos'] = df['de_maquina_dfos'].fillna('SIN_DESCRIPCION')
    df['id_maquina_causal_dfos'] = df['id_maquina_causal_dfos'].fillna('SIN_CAUSAL')
    df['de_maquina_causal_dfos'] = df['de_maquina_causal_dfos'].fillna('SIN_CAUSAL')
    
    return df


def imputar_sku(df):
    """Imputa SKU faltante."""
    print("   🔧 Imputando SKU...")
    
    df['id_sku'] = df['id_sku'].fillna('SIN_SKU')
    df['tiene_sku_asignado'] = (df['id_sku'] != 'SIN_SKU')
    
    return df


def crear_variable_origen(df):
    """Crea variable de origen de pérdida."""
    print("   🔧 Creando variable origen...")
    
    def clasificar_origen(row):
        if pd.notna(row['id_perdida_dfos']):
            return 'SISTEMA_DFOS'
        elif pd.notna(row['id_perdida_manual']):
            return 'MANUAL'
        else:
            return 'DESCONOCIDO'
    
    df['origen_perdida'] = df.apply(clasificar_origen, axis=1).astype('category')
    return df


def agrupar_perdida_3(df):
    """Agrupa categorías de de_perdida_3."""
    print("   📊 Agrupando categorías de pérdida...")
    
    def agrupar_perdidas(valor):
        if pd.isna(valor) or valor == 'DESCONOCIDO':
            return 'Sin_Clasificar'
        elif any(x in str(valor) for x in ['Breakdown', 'Equipment Failure', 'Emergency Stop']):
            return 'Breakdown_Fallas_Equipos'
        elif 'Changeover' in str(valor):
            return 'Cambio_Formato'
        elif 'Minor Stoppages' in str(valor):
            return 'Paradas_Menores'
        elif any(x in str(valor) for x in ['Speed Loss', 'Reduced Speed']):
            return 'Perdida_Velocidad'
        elif 'Setup' in str(valor):
            return 'Setup_Ajustes'
        elif any(x in str(valor) for x in ['Planned', 'Planned Maintenance']):
            return 'Mantenimiento_Planificado'
        elif any(x in str(valor) for x in ['Start', 'Shutdown', 'No Production']):
            return 'Arranque_Parada'
        elif any(x in str(valor) for x in ['Material', 'Raw Material']):
            return 'Problemas_Materiales'
        elif any(x in str(valor) for x in ['Human Error', 'Meal', 'Tea Break']):
            return 'Recursos_Humanos'
        elif any(x in str(valor) for x in ['Utility', 'Utilities', 'Infrastructure']):
            return 'Utilidades_Infraestructura'
        elif any(x in str(valor) for x in ['Weekends', 'Bank Holidays']):
            return 'Tiempo_Programado'
        elif any(x in str(valor) for x in ['Out of Specifications', 'FG Out']):
            return 'Problemas_Calidad'
        else:
            return 'Otros'
    
    df['perdida_agrupada'] = df['de_perdida_3'].apply(agrupar_perdidas).astype('category')
    return df


def crear_variables_analisis_averias(df):
    """Crea variables para análisis de averías."""
    print("   🔬 Creando variables de complejidad...")
    
    def clasificar_complejidad(row):
        if row['fl_asignado']:
            if row['duracion_minutos'] > 60:
                return 'COMPLEJA_LARGA'
            else:
                return 'COMPLEJA_CORTA'
        else:
            if row['duracion_minutos'] > 720:
                return 'AUTOMATICA_LARGA'
            else:
                return 'AUTOMATICA_CORTA'
    
    df['complejidad_resolucion'] = df.apply(clasificar_complejidad, axis=1).astype('category')
    
    return df


# ============================================================================
# FUNCIONES DE EVALUACIÓN
# ============================================================================

def evaluar_modelo(y_true, y_pred, y_pred_proba, dataset_name, logger=None):
    """
    Evalúa el modelo y muestra métricas completas.
    
    Parameters:
    -----------
    y_true : array-like
        Etiquetas verdaderas
    y_pred : array-like
        Predicciones binarias (0/1)
    y_pred_proba : array-like
        Probabilidades predichas
    dataset_name : str
        Nombre del conjunto (TRAIN/VAL/TEST)
    logger : logging.Logger, optional
        Logger para registrar métricas
        
    Returns:
    --------
    dict : Diccionario con todas las métricas
    """
    from sklearn.metrics import (
        confusion_matrix, roc_auc_score, average_precision_score,
        precision_score, recall_score, f1_score
    )
    
    print(f"\n{'='*60}")
    print(f"📈 MÉTRICAS - {dataset_name}")
    print(f"{'='*60}")
    
    # Métricas básicas
    auc_roc = roc_auc_score(y_true, y_pred_proba)
    avg_precision = average_precision_score(y_true, y_pred_proba)
    
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    print(f"\n🎯 Métricas con umbral 0.5:")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall: {recall:.4f}")
    print(f"   F1-Score: {f1:.4f}")
    
    print(f"\n📊 Métricas generales:")
    print(f"   AUC-ROC: {auc_roc:.4f}")
    print(f"   Average Precision: {avg_precision:.4f}")
    
    # Matriz de confusión
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    print(f"\n📋 Matriz de Confusión:")
    print(f"   TN: {tn:,}  |  FP: {fp:,}")
    print(f"   FN: {fn:,}  |  TP: {tp:,}")
    
    print(f"\n💡 Interpretación:")
    print(f"   Verdaderos Negativos (TN): {tn:,} - Ventanas sin falla predichas correctamente")
    print(f"   Falsos Positivos (FP): {fp:,} - Falsas alarmas")
    print(f"   Falsos Negativos (FN): {fn:,} - Fallas NO detectadas ⚠️")
    print(f"   Verdaderos Positivos (TP): {tp:,} - Fallas detectadas correctamente ✓")
    
    # Tasas
    tasa_deteccion = tp / (tp + fn) if (tp + fn) > 0 else 0
    tasa_falsas_alarmas = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    print(f"\n📈 Tasas:")
    print(f"   Tasa de Detección: {tasa_deteccion*100:.2f}% (de todas las fallas, cuántas detectamos)")
    print(f"   Tasa de Falsas Alarmas: {tasa_falsas_alarmas*100:.2f}% (de ventanas normales, cuántas marcamos mal)")
    
    # Precision@K (top 10% predicciones)
    k = int(len(y_true) * 0.1)
    top_k_indices = np.argsort(y_pred_proba)[-k:]
    precision_at_k = y_true.iloc[top_k_indices].mean() if hasattr(y_true, 'iloc') else y_true[top_k_indices].mean()
    
    print(f"\n🎯 Precision@10% (top 10% predicciones más confiables):")
    print(f"   {precision_at_k*100:.2f}% de las top predicciones son fallas reales")
    
    # Logging si está disponible
    if logger:
        logger.info(f"Métricas {dataset_name} - AUC: {auc_roc:.4f}, Recall: {recall:.4f}, Precision: {precision:.4f}")
    
    return {
        'auc_roc': auc_roc,
        'avg_precision': avg_precision,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'precision_at_10': precision_at_k,
        'tasa_deteccion': tasa_deteccion,
        'tasa_falsas_alarmas': tasa_falsas_alarmas
    }


# ============================================================================
# FUNCIONES DE UTILIDAD
# ============================================================================

def mostrar_resumen_dataframe(df, nombre="DataFrame"):
    """Muestra resumen completo de un DataFrame."""
    print(f"\n{'='*80}")
    print(f"📊 RESUMEN: {nombre}")
    print(f"{'='*80}")
    print(f"Shape: {df.shape}")
    print(f"Memoria: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print(f"\nColumnas ({len(df.columns)}):")
    print(df.dtypes.value_counts())
    print(f"\nNulos por columna:")
    nulos = df.isna().sum()
    if nulos.sum() > 0:
        print(nulos[nulos > 0])
    else:
        print("✅ Sin valores nulos")
    print("="*80)


def guardar_checkpoint(df, filepath, descripcion="Checkpoint"):
    """Guarda checkpoint de DataFrame."""
    print(f"\n💾 Guardando {descripcion}...")
    df.to_parquet(filepath, compression='snappy', index=False)
    tamaño_mb = filepath.stat().st_size / 1024**2
    print(f"✅ Guardado: {filepath.name} ({tamaño_mb:.1f} MB)")


def cargar_checkpoint(filepath, descripcion="Checkpoint"):
    """Carga checkpoint de DataFrame."""
    print(f"\n📂 Cargando {descripcion}...")
    df = pd.read_parquet(filepath)
    print(f"✅ Cargado: {filepath.name} - Shape: {df.shape}")
    return df
