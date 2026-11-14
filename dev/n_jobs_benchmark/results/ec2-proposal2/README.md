# Propuesta 2 (no arrancar más procesos que folds): EC2

Máquina: EC2 r5a.4xlarge, AMD EPYC 7571, 16 CPU (8 núcleos); 8 CPU emulados
con la afinidad. Python 3.12.13, numpy 2.5.4, pandas 2.3.3,
scikit-learn 1.9.1, joblib 1.6.0, lightgbm 4.7.0, xgboost 3.4.1.

- `before/`: rama `perf/n-jobs-auto-strategy`, commit 62dd4aa (propuestas 1
  y 3).
- `after/`: rama `perf/n-jobs-auto-strategy`, commit 8e9223b (propuestas 1,
  2 y 3).

El script se ejecutó desde otra copia (`chore/n-jobs-benchmark`, commit
09bdbe6), mientras el clon instalado cambiaba de commit entre las dos
pasadas. Por eso `git_commit` es el mismo en las dos. Los procesos usados
confirman que se midió código distinto: antes, 15 con 16 CPU y 7 con 8;
después, 6 en todas las medidas en paralelo.

```bash
python bench_n_jobs.py run --cpus 16 8 --estimators xgb hgb lgbm_1t --reps 2 --blocks gs_refit_small bt_refit_small --max-priority 4 --out <carpeta>
```

Mediana de 2 repeticiones; segundos en frío / en caliente.

## 16 CPU

Búsqueda con refit (3 candidatos de 6 folds) con `'auto'`, que pasa de 15
procesos a 6:

| Estimador | Antes | Después | Un proceso |
|---|---|---|---|
| XGBoost | 37,3 / 31,0 | 25,9 / 23,6 | 32,8 / 32,8 |
| HistGB | 29,2 / 22,5 | 23,5 / 21,2 | 61,3 / 61,3 |
| LightGBM, 1 hilo | 24,6 / 18,5 | 20,7 / 18,4 | 52,6 / 52,8 |

Backtesting con refit (6 folds) con `n_jobs=15`, que pasa a 6 procesos:

| Estimador | Antes | Después | Un proceso |
|---|---|---|---|
| XGBoost | 12,6 / 12,3 | 9,9 / 7,8 | 10,7 / 10,7 |
| HistGB | 9,6 / 9,3 | 9,0 / 6,6 | 20,1 / 20,1 |
| LightGBM, 1 hilo | 8,1 / 7,7 | 8,1 / 5,8 | 16,7 / 16,7 |

Memoria máxima de los workers: de 2,3 a 3,1 GB antes, de 1,1 a 1,3 GB después.

## Lectura

- **Búsquedas con `'auto'`:** del 16 al 30 % más rápidas en frío y hasta un
  24 % en caliente (XGBoost). El arranque de los workers baja de 6,1 a 6,7 s
  a 2,3 s. Con XGBoost, la búsqueda en paralelo pasa de empatar con un proceso
  (1,03 veces) a ganarle (1,39 veces).
- **`n_jobs` entero:** el backtesting con 15 procesos pasa a 6 y es del 25 al
  37 % más rápido en caliente.
- **Backtesting con `'auto'`:** sin cambios, porque ya usaba 6 procesos
  (`limit_n_jobs_by_workload`).
- **8 CPU:** de 7 procesos a 6, sin diferencias (por debajo del 1 %).
- **Controles:** las medidas con un proceso son iguales antes y después
  (diferencias por debajo del 2 %). 0 fallos, carga externa máxima 0,3 %,
  predicciones idénticas en todas las medidas.

Las causas encajan con las dos hipótesis del informe, aunque ninguna se ha
medido por separado:

- **Workers sin estrenar.** Con más procesos que folds, cada llamada puede
  caer en workers que aún no han importado skforecast ni el estimador. LightGBM
  de un hilo gana un 25 % en caliente en el backtesting (cada llamada reparte 6
  folds entre 15 workers) y nada en la búsqueda, donde los 15 workers ya se han
  estrenado en los candidatos anteriores.
- **Más hilos por worker.** joblib da `cpu_count() // n_jobs` hilos a cada
  worker: 2 en lugar de 1. XGBoost y HistGB, que usan esos hilos, ganan también
  en caliente en la búsqueda (sus segundos de CPU suben de 210 a 253 y de 133 a
  171); LightGBM de un hilo, no.
