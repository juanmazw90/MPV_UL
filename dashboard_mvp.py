"""
Dashboard MVP - Sistema de Prediccion de Mantenimiento
======================================================
Dashboard orientado a stakeholders no tecnicos.
Demuestra la capacidad predictiva del modelo con datos reales.
"""

import sys
import os

if sys.platform == 'win32':
    try:
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8')
            sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

import json
import logging
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from datetime import datetime, timedelta
from io import BytesIO

# Suppress verbose logs from pipeline modules
logging.getLogger('src.predictor').setLevel(logging.WARNING)
logging.getLogger('src.feature_engineering').setLevel(logging.WARNING)
logging.getLogger('src.db_connector').setLevel(logging.WARNING)
logging.basicConfig(level=logging.WARNING)

BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))

# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="Prediccion de Mantenimiento",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CSS STYLES
# ============================================================================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="st-"] {
        font-family: 'Inter', sans-serif;
    }

    .main-header {
        background: linear-gradient(135deg, #1a237e 0%, #0d47a1 50%, #01579b 100%);
        padding: 24px 32px;
        border-radius: 16px;
        margin-bottom: 16px;
        color: white;
    }
    .main-header h1 {
        margin: 0; font-size: 1.8rem; font-weight: 700; color: white;
    }
    .main-header p {
        margin: 4px 0 0 0; font-size: 0.9rem; opacity: 0.85; color: #e3f2fd;
    }

    .training-banner {
        background: #FFF3E0;
        border-left: 4px solid #FF9800;
        padding: 14px 24px;
        border-radius: 0 8px 8px 0;
        margin-bottom: 20px;
        font-size: 1.1rem;
        color: #E65100;
    }
    .training-banner b {
        font-size: 1.15rem;
    }
    .training-banner .horizonte-badge {
        background: #FF9800;
        color: white;
        padding: 2px 10px;
        border-radius: 12px;
        font-weight: 700;
        font-size: 0.95rem;
    }

    .periodo-banner {
        background: linear-gradient(90deg, #FFF8E1, #FFE0B2);
        border: 2px solid #FFB74D;
        border-radius: 10px;
        padding: 12px 20px;
        margin: 8px 0 12px 0;
        text-align: center;
    }
    .periodo-banner .periodo-label { font-size: 0.8rem; color: #E65100; font-weight: 600; }
    .periodo-banner .periodo-fechas { font-size: 1.15rem; font-weight: 800; color: #E65100; }

    .kpi-card {
        background: white;
        border-radius: 16px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        transition: transform 0.3s, box-shadow 0.3s;
        height: 160px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .kpi-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.12);
    }
    .kpi-icon { font-size: 2rem; margin-bottom: 4px; }
    .kpi-value { font-size: 2rem; font-weight: 800; line-height: 1.1; }
    .kpi-title { font-size: 0.85rem; color: #666; font-weight: 600; text-transform: uppercase; margin-top: 4px; }
    .kpi-subtitle { font-size: 0.8rem; color: #777; margin-top: 4px; font-weight: 500; }

    .semaforo-grid {
        display: flex; flex-wrap: wrap; gap: 8px; justify-content: center;
        padding: 16px; background: #f8f9fa; border-radius: 12px;
    }
    .semaforo-item {
        text-align: center; padding: 8px 10px; border-radius: 10px;
        background: white; box-shadow: 0 2px 6px rgba(0,0,0,0.06);
        min-width: 80px;
    }
    .semaforo-dot {
        width: 20px; height: 20px; border-radius: 50%;
        margin: 0 auto 4px; box-shadow: 0 0 10px;
    }
    .semaforo-label { font-size: 0.6rem; font-weight: 600; color: #333; word-break: break-all; }

    .matriz-cell {
        border-radius: 12px; padding: 20px; text-align: center;
        min-height: 120px; display: flex; flex-direction: column;
        justify-content: center;
    }
    .matriz-cell h3 { margin: 0 0 8px 0; font-size: 2rem; font-weight: 800; }
    .matriz-cell .matriz-pct { font-size: 1.1rem; font-weight: 600; opacity: 0.85; margin: 4px 0 0 0; }
    .matriz-cell p { margin: 0; font-size: 0.8rem; }

    .progress-container {
        background: #E0E0E0; border-radius: 10px; height: 28px;
        overflow: hidden; margin: 8px 0;
    }
    .progress-bar {
        height: 100%; border-radius: 10px;
        display: flex; align-items: center; justify-content: center;
        color: white; font-weight: 700; font-size: 0.85rem;
        transition: width 1.5s ease-out;
    }

    .sidebar-info {
        background: #f0f4ff; border-radius: 8px; padding: 12px;
        font-size: 0.75rem; color: #555; margin-top: 16px;
    }

    div[data-testid="stTabs"] button {
        font-weight: 600 !important;
        font-size: 0.9rem !important;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# MONTH NAMES (Spanish)
# ============================================================================

MESES_ES = {
    1: 'enero', 2: 'febrero', 3: 'marzo', 4: 'abril',
    5: 'mayo', 6: 'junio', 7: 'julio', 8: 'agosto',
    9: 'septiembre', 10: 'octubre', 11: 'noviembre', 12: 'diciembre'
}

NIVEL_COLORES = {
    'critico': '#EF5350', 'CRITICAL': '#EF5350',
    'moderado': '#FF9800', 'MODERATE': '#FF9800',
    'bajo': '#66BB6A', 'LOW': '#66BB6A',
    'normal': '#4CAF50', 'NONE': '#4CAF50',
}

ACCIONES = {
    'critico': 'Mantenimiento INMEDIATO - Coordinar con produccion',
    'CRITICAL': 'Mantenimiento INMEDIATO - Coordinar con produccion',
    'moderado': 'Inspeccion PRIORITARIA en 24h',
    'MODERATE': 'Inspeccion PRIORITARIA en 24h',
    'bajo': 'Monitoreo INTENSIVO - Verificar en proximo turno',
    'LOW': 'Monitoreo INTENSIVO - Verificar en proximo turno',
    'normal': 'Operacion NORMAL - Mantenimiento segun plan',
    'NONE': 'Operacion NORMAL - Mantenimiento segun plan',
}

# ============================================================================
# HELPERS: LOAD MODEL METADATA
# ============================================================================

@st.cache_data
def load_model_metadata():
    config_path = BASE_DIR / 'model' / 'inference_config.json'
    with open(config_path, 'r') as f:
        cfg = json.load(f)

    dt = datetime.strptime(cfg['fechas_entrenamiento']['fecha_fin_test'], '%Y-%m-%d %H:%M:%S')
    fecha_es = f"{dt.day} de {MESES_ES[dt.month]} de {dt.year}"

    return {
        'fecha_entrenamiento': fecha_es,
        'n_features': cfg['modelo']['n_features'],
        'version': cfg.get('version', '1.0'),
        'umbral_produccion': cfg['umbrales']['produccion'],
        'umbrales_alertas': cfg['umbrales']['alertas'],
        'metricas_ref': cfg.get('metricas_referencia', {}),
    }


@st.cache_data
def load_required_features():
    path = BASE_DIR / 'model' / 'features_utiles.json'
    with open(path, 'r') as f:
        return json.load(f)['features']


# ============================================================================
# HELPERS: DATA DETECTION AND LOADING
# ============================================================================

def detect_file_type(df):
    """Detect if uploaded file is pre-computed results or raw features."""
    if 'probabilidad_falla' in df.columns and 'prediccion_binaria' in df.columns:
        return 'resultados'

    required = load_required_features()
    present = [f for f in required if f in df.columns]
    if len(present) >= 40:  # Allow some tolerance
        return 'features'

    return 'desconocido'


def validate_features(df):
    required = load_required_features()
    meta_cols = ['timestamp_hora', 'id_maquina_dfos', 'id_linea']

    missing_features = [f for f in required if f not in df.columns]
    missing_meta = [m for m in meta_cols if m not in df.columns]

    return {
        'valid': len(missing_features) == 0 and len(missing_meta) == 0,
        'missing_features': missing_features,
        'missing_meta': missing_meta,
        'has_target': 'target' in df.columns,
    }


def normalize_results(df):
    """Normalize column names from notebook results to standard format."""
    col_map = {
        'probabilidad_falla': 'score',
        'prediccion_binaria': 'pred',
        'target_real': 'target',
    }
    df = df.rename(columns=col_map)
    df['timestamp_hora'] = pd.to_datetime(df['timestamp_hora'])

    # Normalize nivel_alerta to lowercase
    if 'nivel_alerta' in df.columns:
        nivel_map = {'CRITICAL': 'critico', 'MODERATE': 'moderado', 'LOW': 'bajo', 'NONE': 'normal'}
        df['nivel_alerta'] = df['nivel_alerta'].map(nivel_map).fillna(df['nivel_alerta'])

    return df


# ============================================================================
# HELPERS: RUN INFERENCE
# ============================================================================

@st.cache_resource
def get_predictor():
    from src.predictor import Predictor
    config = {
        'model': {
            'path': str(BASE_DIR / 'model' / 'model_optimized.pkl'),
            'version': '1.0'
        }
    }
    return Predictor(config)


def run_inference(df):
    """Run model inference on a DataFrame with features."""
    predictor = get_predictor()
    meta = load_model_metadata()

    results = predictor.predict(df)

    # Rename to standard columns
    results = results.rename(columns={'score': 'score'})

    # Add binary prediction
    results['pred'] = (results['score'] >= meta['umbral_produccion']).astype(int)

    # Add target if available
    if 'target' in df.columns:
        target_map = df.set_index(['id_maquina_dfos', 'timestamp_hora'])['target']
        results['target'] = results.set_index(['id_maquina_dfos', 'timestamp_hora']).index.map(
            lambda x: target_map.get(x, np.nan)
        )
        # Fallback merge if map fails
        if results['target'].isna().all():
            results = results.drop(columns=['target'])
            results = results.merge(
                df[['id_maquina_dfos', 'timestamp_hora', 'target']],
                on=['id_maquina_dfos', 'timestamp_hora'],
                how='left'
            )

    return results


# ============================================================================
# HELPERS: COMPUTE NON-TECHNICAL METRICS
# ============================================================================

def compute_metrics(df):
    """Compute non-technical metrics from predictions with target."""
    has_target = 'target' in df.columns and df['target'].notna().any()

    total_machines = df['id_maquina_dfos'].nunique()

    # Count alerts by level
    alertas = df[df['nivel_alerta'].isin(['critico', 'moderado', 'bajo'])]
    n_alertas = len(alertas)
    n_criticas = len(df[df['nivel_alerta'] == 'critico'])

    metrics = {
        'total_machines': total_machines,
        'n_alertas': n_alertas,
        'n_criticas': n_criticas,
        'has_target': has_target,
    }

    if has_target:
        pred_col = 'pred' if 'pred' in df.columns else 'prediccion_binaria'
        target_col = 'target' if 'target' in df.columns else 'target_real'

        tp = int(((df[pred_col] == 1) & (df[target_col] == 1)).sum())
        fp = int(((df[pred_col] == 1) & (df[target_col] == 0)).sum())
        fn = int(((df[pred_col] == 0) & (df[target_col] == 1)).sum())
        tn = int(((df[pred_col] == 0) & (df[target_col] == 0)).sum())

        total_fallas = tp + fn
        tasa_deteccion = tp / total_fallas if total_fallas > 0 else 0
        fiabilidad = tp / (tp + fp) if (tp + fp) > 0 else 0

        metrics.update({
            'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
            'total_fallas': total_fallas,
            'tasa_deteccion': tasa_deteccion,
            'fiabilidad': fiabilidad,
        })

    return metrics


# ============================================================================
# RENDER HELPERS
# ============================================================================

def render_kpi(icon, value, title, subtitle, color):
    st.markdown(f"""
    <div class="kpi-card" style="border-top: 4px solid {color};">
        <div class="kpi-icon">{icon}</div>
        <div class="kpi-value" style="color: {color};">{value}</div>
        <div class="kpi-title">{title}</div>
        <div class="kpi-subtitle">{subtitle}</div>
    </div>
    """, unsafe_allow_html=True)


def render_progress_bar(detected, total, label=""):
    pct = detected / total * 100 if total > 0 else 0
    color = '#4CAF50' if pct >= 70 else '#FF9800' if pct >= 50 else '#EF5350'
    st.markdown(f"""
    <div style="margin: 12px 0;">
        <div style="font-size: 1rem; font-weight: 600; margin-bottom: 6px;">{label}</div>
        <div class="progress-container">
            <div class="progress-bar" style="width: {pct:.0f}%; background: linear-gradient(90deg, {color}dd, {color});">
                {pct:.0f}%
            </div>
        </div>
        <div style="font-size: 0.8rem; color: #666; margin-top: 4px;">
            {detected:,} de {total:,}
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_matriz_cell(value, title, desc, bg_color, text_color, pct=None):
    pct_html = f'<p class="matriz-pct">({pct:.1f}%)</p>' if pct is not None else ''
    st.markdown(f"""
    <div class="matriz-cell" style="background: {bg_color}; color: {text_color};">
        <h3>{value:,}</h3>
        {pct_html}
        <p><b>{title}</b></p>
        <p style="font-size: 0.7rem; opacity: 0.8;">{desc}</p>
    </div>
    """, unsafe_allow_html=True)


def export_to_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Resultados', index=False)
    return output.getvalue()


# ============================================================================
# TAB 1: RESUMEN EJECUTIVO
# ============================================================================

def tab_resumen(df, metrics):
    # KPIs
    c1, c2, c3 = st.columns(3)
    with c1:
        render_kpi("🏭", f"{metrics['total_machines']}", "Maquinas Analizadas",
                   "en el periodo seleccionado", "#1565C0")
    with c2:
        render_kpi("⚠️", f"{metrics['n_alertas']:,}", "Alertas Generadas",
                   "requieren atencion", "#FF9800")
    with c3:
        if metrics['has_target']:
            pct = f"{metrics['tasa_deteccion']*100:.0f}%"
            render_kpi("🎯", pct, "Fallas Detectadas",
                       f"{metrics['tp']} de {metrics['total_fallas']} fallas", "#4CAF50")
        else:
            render_kpi("🎯", "N/A", "Fallas Detectadas",
                       "sin datos de validacion", "#9E9E9E")

    st.markdown("<br>", unsafe_allow_html=True)

    # Semaforo de maquinas (tabla compacta)
    st.subheader("Estado de Maquinas")

    score_col = 'score' if 'score' in df.columns else 'probabilidad_falla'

    # Get max score per machine
    group_cols = ['id_maquina_dfos']
    has_linea = 'id_linea' in df.columns
    if has_linea:
        group_cols.append('id_linea')

    machine_status = df.groupby(group_cols).agg(
        max_score=(score_col, 'max'),
        nivel=('nivel_alerta', lambda x: x.mode().iloc[0] if len(x) > 0 else 'normal')
    ).reset_index()

    # Re-classify by max score
    meta = load_model_metadata()
    umbrales = meta['umbrales_alertas']

    def classify(score):
        if score >= umbrales['critical']:
            return 'critico'
        elif score >= umbrales['moderate']:
            return 'moderado'
        elif score >= umbrales['low']:
            return 'bajo'
        return 'normal'

    machine_status['nivel'] = machine_status['max_score'].apply(classify)

    # Extract line group for segmentation
    if has_linea:
        machine_status['grupo'] = 'Linea ' + machine_status['id_linea'].astype(str)
    else:
        import re
        def extract_group(name):
            name = name.strip()
            m = re.match(r'^([A-Za-z_]+)', name)
            if m and len(m.group(1)) >= 2:
                return m.group(1).rstrip('_')
            return 'Otros'
        machine_status['grupo'] = machine_status['id_maquina_dfos'].apply(extract_group)

    # Sort by severity then score
    orden = {'critico': 0, 'moderado': 1, 'bajo': 2, 'normal': 3}
    machine_status['orden'] = machine_status['nivel'].map(orden)
    machine_status = machine_status.sort_values(['orden', 'max_score'], ascending=[True, False])

    # Build display table
    tabla = machine_status[['id_maquina_dfos', 'grupo', 'max_score', 'nivel']].copy()
    tabla.columns = ['Maquina', 'Linea/Grupo', 'Riesgo Max', 'Nivel']
    tabla['Riesgo Max'] = (tabla['Riesgo Max'] * 100).round(1)
    tabla['Accion'] = tabla['Nivel'].map(ACCIONES).fillna('-')

    # Color dot column
    tabla['Estado'] = tabla['Nivel'].map({
        'critico': '🔴', 'moderado': '🟠', 'bajo': '🟡', 'normal': '🟢'
    })

    # Show only machines with alerts by default, toggle to show all
    n_with_alerts = tabla[tabla['Nivel'].isin(['critico', 'moderado', 'bajo'])].shape[0]
    show_all = st.checkbox(
        f"Mostrar todas las maquinas ({len(tabla)})",
        value=False,
        help=f"{n_with_alerts} maquinas con alertas de {len(tabla)} totales"
    )

    display_tabla = tabla if show_all else tabla[tabla['Nivel'].isin(['critico', 'moderado', 'bajo'])]

    if display_tabla.empty:
        st.info("No hay maquinas con alertas activas en este periodo.")
    else:
        # Reorder columns for display
        display_tabla = display_tabla[['Estado', 'Maquina', 'Linea/Grupo', 'Riesgo Max', 'Nivel', 'Accion']].copy()
        display_tabla['Maquina'] = display_tabla['Maquina'].str.strip()

        st.dataframe(
            display_tabla,
            use_container_width=True,
            hide_index=True,
            height=min(500, len(display_tabla) * 38 + 40),
            column_config={
                'Estado': st.column_config.TextColumn('', width=40),
                'Maquina': st.column_config.TextColumn('Maquina', width=180),
                'Linea/Grupo': st.column_config.TextColumn('Linea/Grupo', width=120),
                'Riesgo Max': st.column_config.ProgressColumn(
                    'Riesgo Max (%)', min_value=0, max_value=100, format='%.1f%%'
                ),
                'Nivel': st.column_config.TextColumn('Nivel', width=90),
                'Accion': st.column_config.TextColumn('Accion Recomendada', width=300),
            }
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Top 10 risk ranking
    st.subheader("Top 10 - Maquinas con Mayor Riesgo")

    top10 = machine_status.nlargest(10, 'max_score').copy()
    top10['Riesgo (%)'] = (top10['max_score'] * 100).round(1)

    fig = px.bar(
        top10, x='Riesgo (%)', y='id_maquina_dfos', orientation='h',
        color='max_score',
        color_continuous_scale=['#FFE0B2', '#FF9800', '#E65100'],
        range_color=[0, 100],
        labels={'id_maquina_dfos': '', 'max_score': 'Riesgo'}
    )
    fig.update_layout(
        yaxis={'categoryorder': 'total ascending'},
        height=400,
        showlegend=False,
        coloraxis_showscale=False,
        margin=dict(l=0, r=20, t=10, b=10),
        xaxis_title="Nivel de Riesgo (%)",
    )
    fig.update_traces(
        text=top10['Riesgo (%)'].apply(lambda x: f"{x:.0f}%"),
        textposition='outside'
    )
    st.plotly_chart(fig, use_container_width=True)


# ============================================================================
# TAB 2: DETALLE POR MAQUINA
# ============================================================================

def tab_detalle(df):
    machines = sorted(df['id_maquina_dfos'].unique())
    selected = st.selectbox("Seleccionar maquina:", machines, key='machine_detail')

    df_machine = df[df['id_maquina_dfos'] == selected].copy()
    df_machine = df_machine.sort_values('timestamp_hora')

    score_col = 'score' if 'score' in df_machine.columns else 'probabilidad_falla'

    # Score timeline
    st.subheader(f"Evolucion del Riesgo - {selected}")

    fig = go.Figure()

    meta = load_model_metadata()
    umbrales = meta['umbrales_alertas']

    # Color bars by alert level
    colors = df_machine['nivel_alerta'].map(NIVEL_COLORES).fillna('#9E9E9E')

    fig.add_trace(go.Bar(
        x=df_machine['timestamp_hora'],
        y=df_machine[score_col] * 100,
        marker_color=colors.tolist(),
        hovertemplate='<b>%{x}</b><br>Riesgo: %{y:.1f}%<extra></extra>',
        name='Nivel de Riesgo'
    ))

    # Threshold lines
    fig.add_hline(y=umbrales['critical']*100, line_dash="dash", line_color="#EF5350",
                  annotation_text="Critico", annotation_position="right")
    fig.add_hline(y=umbrales['moderate']*100, line_dash="dash", line_color="#FF9800",
                  annotation_text="Moderado", annotation_position="right")

    fig.update_layout(
        height=350,
        yaxis_title="Nivel de Riesgo (%)",
        xaxis_title="",
        margin=dict(l=0, r=60, t=10, b=10),
        yaxis_range=[0, 100],
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Actions table
    st.subheader("Acciones Recomendadas")

    display_df = df_machine[['timestamp_hora', 'nivel_alerta', score_col]].copy()
    display_df.columns = ['Fecha/Hora', 'Nivel', 'Riesgo (%)']
    display_df['Riesgo (%)'] = (display_df['Riesgo (%)'] * 100).round(1)
    display_df['Accion Recomendada'] = display_df['Nivel'].map(ACCIONES).fillna('-')

    # Only show rows with alerts
    display_alerts = display_df[display_df['Nivel'].isin(['critico', 'moderado', 'bajo'])]

    if display_alerts.empty:
        st.info("Esta maquina no presenta alertas en el periodo seleccionado.")
    else:
        st.dataframe(
            display_alerts.sort_values('Fecha/Hora', ascending=False),
            use_container_width=True,
            hide_index=True,
            height=min(400, len(display_alerts) * 40 + 40)
        )

    # Export
    col1, col2 = st.columns([1, 4])
    with col1:
        excel_data = export_to_excel(display_df)
        st.download_button(
            "Descargar Excel",
            data=excel_data,
            file_name=f"detalle_{selected}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )


# ============================================================================
# TAB 3: VALIDACION DEL MODELO
# ============================================================================

def tab_validacion(df, metrics):
    if not metrics['has_target']:
        st.info("La validacion requiere datos con fallas reales conocidas. "
                "Sube un archivo que contenga la columna 'target' o 'target_real'.")
        return

    tp, fp, fn, tn = metrics['tp'], metrics['fp'], metrics['fn'], metrics['tn']
    total_fallas = metrics['total_fallas']
    tasa = metrics['tasa_deteccion']
    fiab = metrics['fiabilidad']

    # Celebration
    if tasa >= 0.80:
        st.balloons()
        st.success(f"Excelente: el sistema detecto el {tasa*100:.0f}% de las fallas a tiempo.")

    # KPIs
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_kpi("🎯", f"{tasa*100:.0f}%", "Deteccion Anticipada",
                   f"{tp} de {total_fallas} fallas detectadas", "#4CAF50")
    with c2:
        render_kpi("✅", f"{fiab*100:.0f}%", "Fiabilidad de Alertas",
                   f"de cada 10 alertas, {fiab*10:.0f} correctas", "#1565C0")
    with c3:
        render_kpi("⚠️", f"{fp:,}", "Falsas Alarmas",
                   "alertas sin falla real", "#FF9800")
    with c4:
        render_kpi("❌", f"{fn:,}", "Fallas No Detectadas",
                   "fallas sin alerta previa", "#EF5350")

    st.markdown("<br>", unsafe_allow_html=True)

    # Detection progress bar
    render_progress_bar(tp, total_fallas, "Fallas detectadas a tiempo")

    st.markdown("<br>", unsafe_allow_html=True)

    # Confusion matrix (non-technical)
    st.subheader("Resultados del Sistema")

    total_obs = tp + fp + fn + tn
    c1, c2 = st.columns(2)
    with c1:
        render_matriz_cell(tp, "Deteccion Exitosa",
                          "Se alerto y efectivamente hubo falla",
                          "#E8F5E9", "#2E7D32",
                          pct=tp/total_obs*100 if total_obs else 0)
    with c2:
        render_matriz_cell(fp, "Falsa Alarma",
                          "Se alerto pero no hubo falla",
                          "#FFF8E1", "#F57F17",
                          pct=fp/total_obs*100 if total_obs else 0)

    c3, c4 = st.columns(2)
    with c3:
        render_matriz_cell(fn, "Falla No Detectada",
                          "Hubo falla sin alerta previa",
                          "#FFEBEE", "#C62828",
                          pct=fn/total_obs*100 if total_obs else 0)
    with c4:
        render_matriz_cell(tn, "Operacion Normal",
                          "Sin falla y sin alerta (correcto)",
                          "#F5F5F5", "#616161",
                          pct=tn/total_obs*100 if total_obs else 0)

    st.markdown("<br>", unsafe_allow_html=True)

    # Distribution by alert level
    st.subheader("Distribucion de Alertas")

    score_col = 'score' if 'score' in df.columns else 'probabilidad_falla'

    if score_col in df.columns:
        fig = px.histogram(
            df, x=score_col, nbins=50,
            color_discrete_sequence=['#1565C0'],
            labels={score_col: 'Nivel de Riesgo'},
        )

        meta = load_model_metadata()
        umbrales = meta['umbrales_alertas']

        fig.add_vline(x=umbrales['critical'], line_dash="dash", line_color="#EF5350",
                      annotation_text="Critico")
        fig.add_vline(x=umbrales['moderate'], line_dash="dash", line_color="#FF9800",
                      annotation_text="Moderado")
        fig.add_vline(x=umbrales['low'], line_dash="dash", line_color="#66BB6A",
                      annotation_text="Bajo")

        fig.update_layout(
            height=300,
            yaxis_title="Cantidad de predicciones",
            xaxis_title="Nivel de Riesgo (probabilidad)",
            margin=dict(l=0, r=20, t=10, b=10),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)


# ============================================================================
# MAIN
# ============================================================================

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🏭 Sistema de Prediccion de Mantenimiento</h1>
        <p>Analisis predictivo de averias para optimizar la produccion</p>
    </div>
    """, unsafe_allow_html=True)

    # Training banner
    meta = load_model_metadata()
    st.markdown(f"""
    <div class="training-banner">
        <b>Modelo entrenado con datos hasta: {meta['fecha_entrenamiento']}</b>
        &nbsp;|&nbsp; {meta['n_features']} indicadores analizados por maquina
        &nbsp;|&nbsp; Version {meta['version']}
        &nbsp;|&nbsp; <span class="horizonte-badge">Horizonte: proximas 24 horas</span>
    </div>
    """, unsafe_allow_html=True)

    # ================================================================
    # SIDEBAR
    # ================================================================
    with st.sidebar:
        st.markdown("### Configuracion")

        data_source = st.radio(
            "Fuente de datos",
            ["Cargar archivo", "Base de Datos"],
            index=0,
            help="Sube un archivo CSV o Parquet con datos preprocesados"
        )

        st.markdown("---")

        df_results = None
        has_target = False

        if data_source == "Cargar archivo":
            uploaded = st.file_uploader(
                "Sube tu archivo de datos",
                type=["csv", "parquet"],
                help="Acepta datasets con features (para ejecutar modelo) "
                     "o resultados ya inferidos (para visualizar)"
            )

            if uploaded is not None:
                # Load file
                with st.spinner("Leyendo archivo..."):
                    if uploaded.name.endswith('.parquet'):
                        df_raw = pd.read_parquet(uploaded)
                    else:
                        df_raw = pd.read_csv(uploaded)

                file_type = detect_file_type(df_raw)

                if file_type == 'resultados':
                    st.success(f"Resultados detectados: {len(df_raw):,} registros")

                    df_norm = normalize_results(df_raw)

                    # Date range
                    date_min = df_norm['timestamp_hora'].min().date()
                    date_max = df_norm['timestamp_hora'].max().date()

                    st.markdown(f"""
                    <div class="periodo-banner">
                        <div class="periodo-label">Periodo disponible</div>
                        <div class="periodo-fechas">{date_min} a {date_max}</div>
                    </div>
                    """, unsafe_allow_html=True)

                    col1, col2 = st.columns(2)
                    with col1:
                        fecha_desde = st.date_input("Desde", value=date_min,
                                                     min_value=date_min, max_value=date_max)
                    with col2:
                        fecha_hasta = st.date_input("Hasta", value=date_max,
                                                     min_value=date_min, max_value=date_max)

                    if st.button("Analizar", type="primary", use_container_width=True):
                        mask = (
                            (df_norm['timestamp_hora'].dt.date >= fecha_desde) &
                            (df_norm['timestamp_hora'].dt.date <= fecha_hasta)
                        )
                        df_results = df_norm[mask].copy()
                        has_target = 'target' in df_results.columns and df_results['target'].notna().any()
                        st.session_state['df_results'] = df_results
                        st.session_state['has_target'] = has_target

                elif file_type == 'features':
                    validation = validate_features(df_raw)

                    if not validation['valid']:
                        if validation['missing_meta']:
                            st.error(f"Faltan columnas identificadoras: {validation['missing_meta']}")
                        if validation['missing_features']:
                            st.error(f"Faltan {len(validation['missing_features'])} indicadores del modelo")
                            with st.expander("Ver indicadores faltantes"):
                                for f in validation['missing_features']:
                                    st.text(f"- {f}")
                    else:
                        st.success(f"Dataset valido: {len(df_raw):,} registros, "
                                   f"{df_raw['id_maquina_dfos'].nunique()} maquinas")

                        df_raw['timestamp_hora'] = pd.to_datetime(df_raw['timestamp_hora'])
                        date_min = df_raw['timestamp_hora'].min().date()
                        date_max = df_raw['timestamp_hora'].max().date()

                        st.markdown(f"""
                        <div class="periodo-banner">
                            <div class="periodo-label">Periodo disponible</div>
                            <div class="periodo-fechas">{date_min} a {date_max}</div>
                        </div>
                        """, unsafe_allow_html=True)

                        col1, col2 = st.columns(2)
                        with col1:
                            # Default to Dec data (post-training)
                            default_from = max(date_min, datetime(2025, 12, 1).date())
                            fecha_desde = st.date_input("Desde", value=default_from,
                                                         min_value=date_min, max_value=date_max)
                        with col2:
                            fecha_hasta = st.date_input("Hasta", value=date_max,
                                                         min_value=date_min, max_value=date_max)

                        if st.button("Analizar", type="primary", use_container_width=True):
                            with st.spinner("Ejecutando analisis predictivo..."):
                                mask = (
                                    (df_raw['timestamp_hora'].dt.date >= fecha_desde) &
                                    (df_raw['timestamp_hora'].dt.date <= fecha_hasta)
                                )
                                df_filtered = df_raw[mask].copy()

                                if len(df_filtered) == 0:
                                    st.error("No hay datos en el rango seleccionado.")
                                else:
                                    df_results = run_inference(df_filtered)
                                    # Merge target back
                                    if 'target' in df_filtered.columns:
                                        target_series = df_filtered.set_index(
                                            ['id_maquina_dfos', 'timestamp_hora']
                                        )['target']
                                        df_results = df_results.merge(
                                            df_filtered[['id_maquina_dfos', 'timestamp_hora', 'target']],
                                            on=['id_maquina_dfos', 'timestamp_hora'],
                                            how='left',
                                            suffixes=('', '_orig')
                                        )
                                        if 'target_orig' in df_results.columns:
                                            df_results['target'] = df_results['target'].fillna(
                                                df_results['target_orig'])
                                            df_results.drop(columns=['target_orig'], inplace=True)

                                    has_target = 'target' in df_results.columns and df_results['target'].notna().any()
                                    st.session_state['df_results'] = df_results
                                    st.session_state['has_target'] = has_target

                else:
                    st.error("Formato de archivo no reconocido. "
                             "Se esperan resultados de inferencia o dataset con features del modelo.")

        else:
            # DB mode
            st.info("Requiere credenciales de base de datos en el archivo .env")
            st.markdown("Variables necesarias:")
            st.code("DB_PROD_HOST\nDB_PROD_USER\nDB_PROD_PASSWORD\nDB_PROD_NAME\nDB_DEV_HOST\n...")

            # Check if env vars exist
            if os.getenv('DB_PROD_HOST'):
                st.success("Credenciales detectadas")
                # TODO: implement DB mode with factory/line selectors
            else:
                st.warning("No se detectaron credenciales. Configura el archivo .env")

        # Model info
        st.markdown("---")
        st.markdown(f"""
        <div class="sidebar-info">
            <b>Sobre el modelo</b><br>
            Algoritmo: LightGBM<br>
            Indicadores: {meta['n_features']}<br>
            <span style="color:#E65100;font-weight:700;">Horizonte: proximas 24 horas</span><br>
            Version: {meta['version']}
        </div>
        """, unsafe_allow_html=True)

    # ================================================================
    # MAIN CONTENT
    # ================================================================

    # Check session state for persisted results
    if 'df_results' in st.session_state and st.session_state['df_results'] is not None:
        df_results = st.session_state['df_results']
        has_target = st.session_state.get('has_target', False)

    if df_results is not None and len(df_results) > 0:
        metrics = compute_metrics(df_results)

        # Create tabs
        tab_names = ["Resumen Ejecutivo", "Detalle por Maquina"]
        if has_target:
            tab_names.append("Validacion del Modelo")

        tabs = st.tabs(tab_names)

        with tabs[0]:
            tab_resumen(df_results, metrics)

        with tabs[1]:
            tab_detalle(df_results)

        if has_target and len(tabs) > 2:
            with tabs[2]:
                tab_validacion(df_results, metrics)
    else:
        # Empty state
        st.markdown("""
        <div style="text-align: center; padding: 80px 20px; color: #999;">
            <div style="font-size: 4rem; margin-bottom: 16px;">📊</div>
            <h3 style="color: #666;">Sube un archivo para comenzar el analisis</h3>
            <p>Selecciona un archivo CSV o Parquet en el panel lateral izquierdo</p>
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
