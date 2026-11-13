# Propuesta 3 mínima (decidir por el tiempo de predicción sin refit): EC2

Máquina: EC2 r5a.4xlarge, AMD EPYC 7571, 16 CPU (8 núcleos); 8 CPU emulados
con la afinidad. Python 3.12.13, numpy 2.5.4, pandas 2.3.3,
scikit-learn 1.9.1, joblib 1.6.0, lightgbm 4.7.0, xgboost 3.4.1,
catboost 1.2.10.

- `before/`: rama `perf/serialize-forecaster-once` (propuesta 1).
- `after/`: rama `perf/n-jobs-auto-prediction-workload` (commit 21fd860).

El script se ejecutó desde otra copia (`chore/n-jobs-benchmark`, commit
b721d08, que añade CatBoost con `'auto'` al plan), mientras el clon instalado
cambiaba de commit entre las dos pasadas. `skforecast_file` confirma que se
midió el código del clon. Por eso `git_commit` es el mismo en las dos.

```bash
python bench_n_jobs.py run --cpus 16 8 --estimators hgb catboost xgb lgbm_1t --reps 2 --blocks bt_norefit --max-priority 4 --out <carpeta>
```

Backtesting de 100 folds de 24 pasos sin refit. Mediana de las repeticiones
(2, salvo `'auto'`, con 1); segundos en frío / en caliente.

| CPU | Estimador | `'auto'` antes | `'auto'` después | Mejor a mano |
|---|---|---|---|---|
| 16 | HistGB | 1 proceso: 19,2 / 19,1 | 15 procesos: 8,1 / 5,0 | 7,6 / 4,9 |
| 8 | HistGB | 1 proceso: 20,3 / 20,4 | 7 procesos: 8,0 / 5,8 | 7,8 / 5,6 |
| 16 | CatBoost | 1 proceso: 3,9 / 3,8 | 1 proceso: 3,9 / 3,8 | 4,7 / 2,3 |
| 8 | CatBoost | 1 proceso: 4,1 / 4,0 | 1 proceso: 4,1 / 4,0 | 5,0 / 2,7 |
| 16 | XGBoost | 1 proceso: 3,0 / 2,9 | 1 proceso: 2,9 / 2,9 | 4,7 / 2,3 |
| 8 | XGBoost | 1 proceso: 3,3 / 3,2 | 1 proceso: 3,3 / 3,2 | 5,0 / 2,7 |

- HistGB con `'auto'`: 2,4 a 2,5 veces más rápido en frío y 3,5 a 3,8 en
  caliente, lo mismo que la mejor configuración puesta a mano. Predecir el
  primer fold en el proceso principal no se nota.
- CatBoost con `'auto'` sigue en un proceso: el primer fold no lo empuja a
  paralelo (3,9 s de predicción, por debajo del mínimo de 6 s en Linux).
- XGBoost y LightGBM con un hilo no pasan por la medida y no cambian.
- Las medidas que no usan `'auto'` son iguales antes y después (diferencias
  por debajo del 2 %). 0 fallos, carga externa máxima 6 %, predicciones
  idénticas en las 50 medidas.
