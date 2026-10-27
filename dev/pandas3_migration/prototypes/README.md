# Prototipos (monkeypatch, fuera de la librería)

Cada módulo de esta carpeta expone `apply()` y `restore()`. `apply()` vuelve a
ejecutar el código fuente de un método o función de skforecast con una
sustitución de texto (`_source_patch.patch_source`) y lo instala en su clase o
módulo; `restore()` devuelve el original. La librería no se modifica: el
cambio vive solo en el proceso que lo importa.

Uso con los scripts de la carpeta padre:

```bash
# A/B intercalado "current" frente a "patched:<nombre>" en la misma sesión
python dev/pandas3_migration/bench_hotspots.py --patch no_copy_keyword
# Huellas de todas las salidas con el prototipo aplicado desde el arranque
python dev/pandas3_migration/identity_cases.py --patch no_copy_keyword --tag no_copy_keyword
```

| Prototipo | Cambio que simula | Versión en la que actúa |
|---|---|---|
| `no_copy_keyword` | Quitar `copy=False` de `pd.concat` y `astype` en pandas 3 (en pandas 2 se mantiene), es decir, la función de compatibilidad `concat`/`astype` propuesta en el informe | Solo pandas 3 (en pandas 2 es un no-op) |
| `dataframe_copy_false` | `pd.DataFrame(data=ndarray, ..., copy=False)` en `create_train_X_y` de los cuatro forecasters que envuelven la matriz de numpy | Ambas (pandas 2 ya no copiaba) |

Al añadir un prototipo, comprobar primero que `apply()` y `restore()` dejan las
salidas idénticas (ver `identity_cases.py --patch`) y luego medir.
