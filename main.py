"""
main.py - Sistema de Mantenimiento Predictivo
==============================================

Orquestador principal del sistema de predicción.
Configurado para procesar exclusivamente la fábrica de Veszprém.

Genera 1 predicción por máquina basada en el estado actual (features históricas).
El timestamp_ventana indica desde cuándo aplica la predicción de 24h.
Para granularidad horaria, ejecutar vía cron cada hora.
"""

import sys
import os

if sys.platform == 'win32':
    try:
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8')
            sys.stderr.reconfigure(encoding='utf-8')
        else:
            import io
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    except Exception as e:
        print(f"Advertencia: No se pudo configurar UTF-8: {e}")

import logging
import yaml
import signal
import tempfile
import warnings
import pandas as pd
from pandas.errors import PerformanceWarning
from save_predictions import save_predictions_to_db
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv

warnings.simplefilter(action='ignore', category=PerformanceWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

load_dotenv('.env')

try:
    import fcntl
except ImportError:
    fcntl = None

BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))

from src.db_connector import DatabaseConnector
from src.feature_engineering import FeatureEngineer
from src.predictor import Predictor
from src.utils import setup_logging, send_email_alert
from src.config_loader import load_config


def setup_timeout(timeout_minutes):
    """Configura timeout para evitar ejecuciones infinitas (Solo Unix)."""
    if not hasattr(signal, 'SIGALRM'):
        logging.warning("[TIMEOUT] No soportado en Windows. Se omitirá el límite de tiempo.")
        return

    def timeout_handler(signum, frame):
        raise TimeoutError(f"Ejecución superó {timeout_minutes} minutos")

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_minutes * 60)


def acquire_lock():
    """Adquiere lock para evitar ejecuciones simultáneas."""
    lockfile_path = os.path.join(tempfile.gettempdir(), 'predictive_maintenance.lock')
    lockfile = open(lockfile_path, 'w')

    if fcntl is None:
        logging.warning("[LOCK] No soportado en Windows. Se ejecutará sin exclusión mutua.")
        return lockfile

    try:
        fcntl.flock(lockfile, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return lockfile
    except IOError:
        logging.error("[LOCK] Otra instancia está corriendo. Saliendo...")
        sys.exit(0)


def release_lock(lockfile):
    """Libera el lock de forma segura."""
    if lockfile and not lockfile.closed:
        if fcntl:
            fcntl.flock(lockfile, fcntl.LOCK_UN)
        lockfile.close()


def generar_predicciones(predictor, feature_engineer,
                         bui_perdida, bui_pm_ewo, bui_line,
                         fecha_referencia, logger):
    """
    Genera 1 predicción por máquina basada en el estado actual de features.

    El modelo predice: "habrá falla grave en las próximas 24h desde este momento".
    El timestamp_ventana se asigna a la hora actual (fecha_referencia).

    Para obtener granularidad horaria, ejecutar este pipeline cada hora vía cron.
    """
    logger.info(f"\nGenerando predicciones para: {fecha_referencia}")

    # Feature Engineering
    logger.info("[FEATURES] Calculando matriz de características...")
    try:
        X = feature_engineer.transform(bui_perdida, bui_pm_ewo, bui_line)
        logger.info(f"   Matriz X: {X.shape} ({X['id_maquina_dfos'].nunique()} máquinas)")
    except Exception as e:
        logger.error(f"   Error en feature engineering: {e}", exc_info=True)
        return pd.DataFrame()

    if len(X) == 0:
        logger.error("   Feature engineering devolvió matriz vacía")
        return pd.DataFrame()

    # Predicción (1 vez, 1 resultado por máquina)
    predictions = predictor.predict(X)

    if len(predictions) == 0:
        logger.error("   No se generaron predicciones")
        return pd.DataFrame()

    # Asignar timestamp de la ventana de predicción
    predictions['timestamp_ventana'] = fecha_referencia

    return predictions


def main():
    """Función principal de ejecución."""
    
    try:
        config = load_config(str(BASE_DIR / 'config.yaml'))
    except Exception as e:
        print(f"[ERROR] Error crítico cargando config: {e}")
        sys.exit(1)
    
    logger = setup_logging(config)
    
    logger.info("=" * 80)
    logger.info("[INICIO] Mantenimiento Predictivo - Fábrica VESZPRÉM")
    logger.info("=" * 80)
    
    lockfile = acquire_lock()
    timeout_minutes = config.get('execution', {}).get('timeout_minutes', 120)
    setup_timeout(timeout_minutes)
    
    db = None
    
    try:
        # =====================================================================
        # PASO 1: EXTRAER DATOS
        # =====================================================================
        db = DatabaseConnector(config)
        db.connect_source()
        
        lookback_hours = config.get('execution', {}).get('lookback_hours', 168)
        lookback_days_maintenance = config.get('execution', {}).get('lookback_days_maintenance', 60)
        fecha_referencia = config.get('execution', {}).get('fecha_referencia', None)
        fecha_hoy = pd.to_datetime(fecha_referencia).date() if fecha_referencia else datetime.now().date()
        
        # Extracción inicial
        bui_perdida = db.extract_events_data(lookback_hours=lookback_hours, fecha_referencia=fecha_referencia)
        bui_pm_ewo = db.extract_maintenance_data(lookback_days=lookback_days_maintenance, fecha_referencia=fecha_referencia)
        bui_line = db.extract_line_metadata()

        # ---------------------------------------------------------------------
        # NUEVO: FILTRO OBLIGATORIO - SOLO VESZPRÉM
        # ---------------------------------------------------------------------
        logger.info("\n[FILTRO] Aplicando filtro de fábrica: Veszprém")
        bui_line = bui_line[bui_line['id_fabrica'] == 21].copy()
        
        if bui_line.empty:
            logger.error("No se encontraron líneas activas para 'Veszprém'. Abortando.")
            return 0
            
        veszprem_line_ids = bui_line['id_linea'].unique().tolist()
        logger.info(f" - Líneas detectadas: {len(veszprem_line_ids)} {veszprem_line_ids}")
        
        # Filtrar el resto de DataFrames por los IDs de línea de Veszprém
        bui_perdida = bui_perdida[bui_perdida['id_linea'].isin(veszprem_line_ids)].copy()
        bui_pm_ewo = bui_pm_ewo[bui_pm_ewo['id_linea'].isin(veszprem_line_ids)].copy()
        # ---------------------------------------------------------------------

        if bui_perdida.empty:
            logger.warning("[SIN DATOS] No hay eventos de producción para Veszprém en el período.")
            return 0
        
        logger.info(f"[INFO] Procesando {bui_perdida['id_maquina_dfos'].nunique()} máquinas de Veszprém")

        # =====================================================================
        # PASO 2: INICIALIZAR COMPONENTES
        # =====================================================================
        logger.info("\n[PASO 2] Inicializando Feature Engineer y Predictor...")
        fe = FeatureEngineer(config)

        # =====================================================================
        # PASO 3: PREDICCIÓN
        # =====================================================================
        logger.info("\n[PASO 3] Cargando Predictor...")
        predictor = Predictor(config)
        
        # =====================================================================
        # PASO 4: PREDICCIÓN (1 por máquina, estado actual)
        # =====================================================================
        logger.info("\n[PASO 4] Generando predicciones (1 por máquina)...")
        fecha_referencia_dt = datetime.combine(fecha_hoy, datetime.min.time())

        predictions_df = generar_predicciones(
            predictor, fe, bui_perdida, bui_pm_ewo, bui_line,
            fecha_referencia_dt, logger
        )
        
        if predictions_df.empty:
            logger.error("[ERROR] No se generaron predicciones.")
            return 1
        
        # =====================================================================
        # PASO 5: GUARDAR EN BD
        # =====================================================================
        logger.info("\n[PASO 5] Guardando en bui_predicciones_hora_dia...")
        registros_guardados = save_predictions_to_db(predictions_df, config)
        
        logger.info(f"\n[ÉXITO] {registros_guardados:,} registros guardados para Veszprém")
        return 0
    
    except Exception as e:
        logger.error(f"\n[ERROR CRÍTICO] {e}", exc_info=True)
        try:
            send_email_alert(config, subject="[ERROR] Predicciones Veszprém Fallidas", body=str(e))
        except: pass
        return 1
    
    finally:
        if db: db.close_connections()
        release_lock(lockfile)
        logger.info("[FIN] Proceso finalizado")


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)