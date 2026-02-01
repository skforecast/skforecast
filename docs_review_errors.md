# Revisión de Documentación de Skforecast - Errores Encontrados

**Fecha de revisión:** 29 de enero de 2026  
**Versión revisada:** 0.20.0

Este documento lista todos los errores encontrados en la documentación de skforecast tras una revisión exhaustiva de las carpetas `user_guides`, `introduction-forecasting`, `quick-start`, `more`, `faq` y el `README.md`.

---

## Resumen Ejecutivo

| Categoría | Errores Encontrados |
|-----------|---------------------|
| Typos/Errores ortográficos | 35+ |
| Errores gramaticales | 8 |
| Imports incorrectos | 1 |
| Referencias obsoletas | 2 |
| URLs/Enlaces incorrectos | 3 |
| Texto duplicado | 2 |
| Parámetros/Variables incorrectos | 5 |
| Inconsistencias texto/código | 4 |
| **Total** | **60+** |

---

# SEGUNDA PASADA - ERRORES ADICIONALES ENCONTRADOS

---

## 7. Typos/Errores Ortográficos Adicionales

### 7.6 "tunning" en forecasting-sarimax-arima.ipynb

**Archivo:** `docs/user_guides/forecasting-sarimax-arima.ipynb`  
**Línea:** 1562

**Texto incorrecto:** `"## Model tunning"`  
**Corrección:** `"## Model tuning"`

## 9. Variables/Parámetros con Nombres Incorrectos

### 9.1 "error_mse" usado con MAE en forecasting-sarimax-arima.ipynb

**Archivo:** `docs/user_guides/forecasting-sarimax-arima.ipynb`  
**Líneas:** 856, 861, 1159, 1164

**Problema:** La variable se llama `error_mse` y el print dice `"Test error (mse)"` pero se calcula `mean_absolute_error` (MAE).

**Código incorrecto:**
```python
error_mse = mean_absolute_error(...)
print(f"Test error (mse): {error_mse}")
```

**Corrección:**
```python
error_mae = mean_absolute_error(...)
print(f"Test error (mae): {error_mae}")
```

---
