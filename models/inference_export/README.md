# Archivos del Modelo - Proyecto de Inferencia

## 📦 Archivos Generados

Fecha de generación: 2026-02-27 18:14:22

### 1. features_utiles.json (1.8 KB)
- Lista de 47 features requeridas para inferencia
- **CRÍTICO**: Mantener el orden exacto de las features

### 2. inference_config.json (1.6 KB)
- Configuración completa del modelo
- Umbrales de alertas
- Métricas de referencia
- Hiperparámetros

### 3. model_optimized.pkl (0.10 MB)
- Modelo LightGBM en formato pickle
- Carga más rápida que .txt

### 4. model_optimized.txt (0.10 MB)
- Modelo LightGBM en formato nativo
- Más portable entre versiones

### 5. preprocessing_artifacts.pkl (0.27 MB)
- Encoders (label encoding, one-hot)
- Estadísticas para z-scores
- Thresholds y valores por defecto

```

## 📊 Métricas del Modelo

- **AUC-ROC Test**: 0.8028
- **Precision**: 35.4%
- **Recall**: 79.0%
- **F1-Score**: 0.5052

## 🚦 Sistema de Alertas

| Nivel | Umbral | Acción |
|-------|--------|--------|
| 🔴 CRITICAL | ≥ 0.7 | Mantenimiento INMEDIATO - Coordinar con producción |
| 🟡 MODERATE | 0.31 - 0.7 | Inspección PRIORITARIA en 24h |
| 🟢 LOW | 0.1 - 0.31 | Monitoreo INTENSIVO - Verificar en próximo turno |
| ⚪ NONE | < 0.1 | Operación NORMAL - Mantenimiento según plan |

## ⚙️ Configuración del Pipeline

- **Horizonte de predicción**: 24h
- **Ventanas temporales**: 24h, 7d, 30d
- **Rolling windows**: [2, 6, 12, 24]

## 📅 Información de Entrenamiento

- **Datos desde**: 2025-07-01
- **Train hasta**: 2025-09-30 23:59:59
- **Test hasta**: 2025-11-30 23:59:59
- **Features**: 47
- **Best iteration**: 21

## 🔄 Re-entrenamiento

Se recomienda re-entrenar el modelo:

- Si las métricas en producción caen > 5%
- Después de cambios significativos en el proceso productivo

## 📝 Notas Importantes

1. **Features**: Deben calcularse EXACTAMENTE igual que en entrenamiento
2. **Orden**: El orden de las features en `features_utiles.json` es CRÍTICO
3. **Preprocessing**: Usar siempre `preprocessing_artifacts.pkl` para consistencia
4. **Umbrales**: Ajustar según feedback operativo y costos reales

---
Generado automáticamente el 2026-02-27 18:14:22
