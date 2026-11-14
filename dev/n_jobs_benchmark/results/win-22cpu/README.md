# Estrategia de `n_jobs` en Windows (22 CPU)

Portátil con Windows 11, Intel de la familia Meteor Lake (familia 6, modelo
170), 22 CPU lógicos. Python 3.12.16, numpy 2.5.4, pandas 2.3.3,
scikit-learn 1.9.1, joblib 1.6.0, xgboost 3.4.1, catboost 1.2.10. Máquina
libre durante las medidas (una primera tanda, con la máquina ocupada, se
descartó).

- `before/`: `0.28.x`, commit 046e511 (sin ninguno de los cambios).
- `after/`: rama `perf/n-jobs-auto-strategy`, commit f35c611 (propuestas 1, 2
  y 3, y la serialización única en multiserie).

```bash
python bench_windows.py run --out <carpeta>
python bench_windows.py compare before after
```

Una repetición por medida, cada una en un proceso nuevo con tres llamadas: la
primera arranca los workers (en frío) y las otras dos los reutilizan (en
caliente, mediana). Salida completa en `compare.txt`. Las medidas con un
proceso varían hasta un 15 % entre llamadas: solo se comentan diferencias
mayores. Predicciones idénticas en todas las medidas.

## Búsquedas y refit (propuestas 1 y 2)

Segundos en frío / en caliente.

| Caso | Antes | Después | Un proceso |
|---|---|---|---|
| Búsqueda de XGBoost con refit (3 candidatos de 6 folds), `'auto'`: 21 → 6 procesos | 26,9 / 20,1 | 16,3 / 13,7 | 21,3 / 23,6 |
| Backtesting de XGBoost con refit (6 folds), `n_jobs=21` → 6 procesos | 9,7 / 8,8 | 8,2 / 4,7 | 7,8 / 7,7 |
| Multiserie de XGBoost con refit (8 folds), `n_jobs=21` → 8 procesos | 7,3 / 6,2 | 6,2 / 3,3 | 7,0 / 6,7 |
| HistGB sin refit (100 folds), `n_jobs=21` | 10,0 / 6,3 | 7,1 / 2,5 | 10,0 / 9,7 |
| HistGB sin refit (40 folds), `n_jobs=21` | 7,2 / 3,5 | 6,5 / 1,8 | 4,5 / 4,1 |
| Multiserie de XGBoost sin refit (52 folds), `n_jobs=21` | 8,0 / 2,9 | 7,0 / 1,9 | 2,1 / 2,0 |

- La búsqueda con `'auto'` era más lenta que un proceso en frío (26,9 frente a
  20,9) y ahora es 1,3 veces más rápida en frío y 1,7 en caliente.
- El backtesting y la búsqueda con `'auto'` que ya usaban 6 u 8 procesos no
  cambian, como se esperaba.
- **El ajuste posterior de XGBoost no se ralentiza en Windows**, ni antes ni
  después (columna `fit after`, de 0,8 a 1,3 s). El problema venía de libgomp
  (Linux); XGBoost en Windows usa la librería OpenMP de Microsoft. La nota de
  versión lo limita a Linux.

## Sin refit: el umbral de 12 s (propuesta 3)

| Caso, sin refit | Un proceso | Paralelo (21) | `'auto'` después |
|---|---|---|---|
| HistGB, 100 folds (predicción de unos 10 s) | 10,0 / 9,7 | 7,1 / 2,5 | 7,2 / 8,0 |
| HistGB, 40 folds (unos 4 s) | 4,5 / 4,1 | 6,5 / 1,8 | 4,4 / 4,0 |
| CatBoost, 100 folds (unos 3 s) | 3,4 / 3,2 | 6,7 / 2,1 | 3,3 / 3,0 |

- Con unos 4 s de predicción, el paralelo pierde en frío; con unos 10 s, gana
  también en frío. Interpolando estos dos casos, en esta máquina compensa en
  frío a partir de unos 7 s, por debajo de los 12 s del umbral de Windows.
- **Se mantiene el umbral de 12 s.** Una sola máquina y una repetición no
  bastan para bajarlo, y el riesgo es asimétrico: un umbral alto solo deja de
  ganar algo entre 7 y 12 s, y uno bajo, en un Windows que arranque procesos
  más despacio (menos núcleos, antivirus), haría el backtesting más lento que
  con un proceso. Con 12 s, cuando `'auto'` eligió el paralelo, ganó.

### La decisión cambia entre llamadas cerca del umbral

Procesos usados por `'auto'` en cada una de las tres llamadas (`after/`):

| Caso | Llamadas |
|---|---|
| HistGB, 100 folds | 21, 1, 1 |
| HistGB, 40 folds | 1, 1, 1 |
| CatBoost, 100 folds | 1, 1, 1 |
| XGBoost, búsqueda / backtesting / multiserie con refit | 6, 6, 6 / 6, 6, 6 / 8, 8, 8 |

En la primera llamada, el primer fold (predicho en el proceso principal) es
más lento porque el proceso acaba de arrancar, y la estimación de la
predicción total pasa de 12 s; en las siguientes queda por debajo y `'auto'`
vuelve a un proceso. No hace daño (la desviación solo empuja hacia el paralelo
entre unos 7 y 12 s, donde ya gana), pero deja ganancia: en las llamadas 2 y
3 los 21 workers ya estaban arrancados y el paralelo habría tardado 2,5 s en
lugar de 8,0. Es la limitación conocida de `'auto'`, que decide como si cada
backtesting fuera el único. Mejora futura, fuera de esta pull request: no
contar el arranque de los workers cuando ya están en marcha.
