#!/bin/bash
# =============================================================================
# setup_cron.sh - Configurar ejecucion automatica del pipeline
# Sistema de Mantenimiento Predictivo - Fabrica Veszprem
# =============================================================================
# Uso:
#   bash scripts/setup_cron.sh           -> instala los cron jobs
#   bash scripts/setup_cron.sh --remove  -> elimina los cron jobs
#   bash scripts/setup_cron.sh --status  -> muestra estado actual
# =============================================================================

set -e

# Detectar directorio raiz del proyecto (un nivel arriba de /scripts)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"

# ─────────────────────────────────────────────────────────────────────────────
# Detectar Python
# ─────────────────────────────────────────────────────────────────────────────
detect_python() {
    # Prioridad: venv del proyecto > conda > python3 del sistema
    if [ -f "$PROJECT_DIR/venv/bin/python" ]; then
        echo "$PROJECT_DIR/venv/bin/python"
    elif [ -f "$PROJECT_DIR/.venv/bin/python" ]; then
        echo "$PROJECT_DIR/.venv/bin/python"
    elif [ -f "$PROJECT_DIR/.conda/bin/python" ]; then
        echo "$PROJECT_DIR/.conda/bin/python"
    elif command -v python3 &>/dev/null; then
        echo "$(command -v python3)"
    else
        echo ""
    fi
}

PYTHON_PATH=$(detect_python)

# ─────────────────────────────────────────────────────────────────────────────
# Configuracion de horarios (parametrizables via argumentos)
# ─────────────────────────────────────────────────────────────────────────────
#
# LOGICA TEMPORAL:
#   main.py corre a HORA_PRED y genera predicciones con horizonte 24h.
#   update_targets.py debe correr DESPUES de que la ventana de 24h se cierre.
#   Si main.py corre a las 06:00, la ventana cubre [06:00 hoy → 06:00 mañana].
#   update_targets corre a HORA_UPDATE (por defecto 07:00 del dia siguiente),
#   con la query filtrando fe_ventana < NOW() - INTERVAL 24 HOUR.
#   Esto garantiza que la ventana completa de 24h ya transcurrio.
#
# Uso con horarios personalizados:
#   bash scripts/setup_cron.sh --hora-pred 5 --hora-update 6
#
HORA_PRED=${HORA_PRED:-6}        # Predicciones: 06:00 AM (inicio turno manana)
HORA_UPDATE=${HORA_UPDATE:-7}    # Evaluacion: 07:00 AM (25h despues de prediccion)

# Parsear argumentos --hora-pred y --hora-update
while [[ $# -gt 0 ]]; do
    case "$1" in
        --hora-pred)   HORA_PRED="$2";   shift 2 ;;
        --hora-update) HORA_UPDATE="$2"; shift 2 ;;
        *) break ;;
    esac
done

CRON_PREDICCIONES="0 $HORA_PRED * * * cd $PROJECT_DIR && $PYTHON_PATH main.py >> $PROJECT_DIR/logs/cron_predicciones.log 2>&1"
CRON_TARGETS="0 $HORA_UPDATE * * * cd $PROJECT_DIR && $PYTHON_PATH update_targets.py >> $PROJECT_DIR/logs/cron_targets.log 2>&1"

CRON_MARKER="# predictive-maintenance-veszprem"

# ─────────────────────────────────────────────────────────────────────────────
# Funciones
# ─────────────────────────────────────────────────────────────────────────────

show_status() {
    echo ""
    echo "=== Estado actual de cron jobs ==="
    if crontab -l 2>/dev/null | grep -q "$CRON_MARKER"; then
        echo "INSTALADOS:"
        crontab -l 2>/dev/null | grep -A1 "$CRON_MARKER" | grep -v "^--$"
    else
        echo "No hay cron jobs instalados para este proyecto."
    fi
    echo ""
}

remove_crons() {
    echo "Eliminando cron jobs del proyecto..."
    if crontab -l 2>/dev/null | grep -q "$CRON_MARKER"; then
        # Eliminar bloques marcados con CRON_MARKER (la linea del marker y la siguiente)
        crontab -l 2>/dev/null | grep -v "$CRON_MARKER" | grep -v "main.py" | \
            grep -v "update_targets.py" | crontab -
        echo "Cron jobs eliminados."
    else
        echo "No habia cron jobs instalados para este proyecto."
    fi
}

install_crons() {
    # Crear directorio de logs si no existe
    mkdir -p "$PROJECT_DIR/logs"

    # Verificar Python
    if [ -z "$PYTHON_PATH" ]; then
        echo "ERROR: No se encontro Python. Verifica que el entorno virtual este creado."
        echo "  Opciones:"
        echo "    python -m venv $PROJECT_DIR/venv"
        echo "    source $PROJECT_DIR/venv/bin/activate && pip install -r requirements.txt"
        exit 1
    fi

    # Verificar que los scripts existen
    if [ ! -f "$PROJECT_DIR/main.py" ]; then
        echo "ERROR: No se encontro main.py en $PROJECT_DIR"
        exit 1
    fi
    if [ ! -f "$PROJECT_DIR/update_targets.py" ]; then
        echo "ERROR: No se encontro update_targets.py en $PROJECT_DIR"
        exit 1
    fi

    # Verificar si ya estan instalados
    if crontab -l 2>/dev/null | grep -q "$CRON_MARKER"; then
        echo "Los cron jobs ya estan instalados. Usando --remove primero para reemplazar."
        remove_crons
    fi

    # Instalar
    (
        crontab -l 2>/dev/null
        echo ""
        echo "$CRON_MARKER"
        echo "$CRON_PREDICCIONES"
        echo "$CRON_MARKER"
        echo "$CRON_TARGETS"
    ) | crontab -

    echo ""
    echo "=== Cron jobs instalados correctamente ==="
    echo ""
    echo "  PREDICCIONES  →  0${HORA_PRED}:00 diario"
    echo "    $PYTHON_PATH main.py"
    echo "    Log: $PROJECT_DIR/logs/cron_predicciones.log"
    echo ""
    echo "  UPDATE TARGETS →  0${HORA_UPDATE}:00 diario (25h despues → ventana 24h cerrada)"
    echo "    $PYTHON_PATH update_targets.py"
    echo "    Log: $PROJECT_DIR/logs/cron_targets.log"
    echo ""
    echo "Proyecto:  $PROJECT_DIR"
    echo "Python:    $PYTHON_PATH"
    echo ""
    echo "Para verificar: crontab -l"
    echo "Para monitorear: tail -f $PROJECT_DIR/logs/cron_predicciones.log"
}

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
case "${1:-}" in
    --remove)
        remove_crons
        ;;
    --status)
        show_status
        ;;
    "")
        echo "=== Setup Cron - Sistema de Mantenimiento Predictivo ==="
        echo "Proyecto: $PROJECT_DIR"
        echo "Python:   ${PYTHON_PATH:-NO ENCONTRADO}"
        echo ""
        install_crons
        show_status
        ;;
    *)
        echo "Uso: bash scripts/setup_cron.sh [--remove|--status]"
        exit 1
        ;;
esac
