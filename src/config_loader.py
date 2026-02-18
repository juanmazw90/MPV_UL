"""
Cargador de configuracion con soporte para variables de entorno.

Soporta dos sintaxis en el YAML:
  ${VAR_NAME}           -> requiere la variable (error si no existe)
  ${VAR_NAME:default}   -> usa 'default' si la variable no existe
"""
import os
import yaml
import re
import logging
from pathlib import Path
from dotenv import load_dotenv

log = logging.getLogger(__name__)

# Patron: ${VAR_NAME} o ${VAR_NAME:valor_default}
_ENV_PATTERN = re.compile(r'\$\{([^}:]+)(?::([^}]*))?\}')


def load_config(config_path: str = "config.yaml") -> dict:
    """
    Carga configuracion desde YAML y reemplaza variables de entorno.

    Sintaxis soportada:
        ${VAR}            -> Reemplaza con os.environ[VAR]. Error si no existe.
        ${VAR:default}    -> Reemplaza con os.environ[VAR] o 'default' si no existe.
    """
    # Cargar .env si existe
    env_path = Path(config_path).parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        log.info(f"Archivo .env cargado desde: {env_path}")

    # Leer YAML
    with open(config_path, 'r', encoding='utf-8') as f:
        config_text = f.read()

    missing_vars = []

    def replace_env_var(match):
        var_name = match.group(1)
        default_value = match.group(2)  # None si no hay :default
        value = os.environ.get(var_name)

        if value is not None:
            return value
        if default_value is not None:
            return default_value

        missing_vars.append(var_name)
        return match.group(0)  # Devolver sin reemplazar

    config_text = _ENV_PATTERN.sub(replace_env_var, config_text)

    if missing_vars:
        log.warning(
            f"Variables de entorno no definidas (sin default): {missing_vars}. "
            f"Funcionalidades que las requieran no estaran disponibles."
        )

    # Parsear YAML
    config = yaml.safe_load(config_text)

    return config


# Uso simple
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    config = load_config()
    print("Configuracion cargada correctamente")
    db_src = config.get('database', {}).get('source', {})
    print(f"   DB Source Host: {db_src.get('host', 'N/A')}")
    print(f"   DB Source User: {db_src.get('user', 'N/A')}")
    print(f"   Thresholds: {config.get('thresholds', {})}")
    print(f"   Execution: {config.get('execution', {})}")