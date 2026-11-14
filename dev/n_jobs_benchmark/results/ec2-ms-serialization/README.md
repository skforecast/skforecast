# Serialización única en el backtesting multiserie: EC2

Máquina: EC2 r5a.4xlarge, AMD EPYC 7571, 16 CPU (8 núcleos); 8 CPU emulados
con la afinidad. Python 3.12.13, numpy 2.5.4, pandas 2.3.3,
scikit-learn 1.9.1, joblib 1.6.0, lightgbm 4.7.0, xgboost 3.4.1.

- `before/`: rama `perf/n-jobs-auto-strategy`, commit 8e9223b (propuestas 1,
  2 y 3; la propuesta 1 solo en el backtesting de una serie).
- `after/`: commit e3eadd9 (el forecaster también se serializa una vez en
  `_backtesting_forecaster_multiseries`).

El script se ejecutó desde otra copia (`chore/n-jobs-benchmark`, commit
364fd57), mientras el clon instalado cambiaba de commit entre las dos
pasadas. Por eso `git_commit` es el mismo en las dos. Los procesos usados son
los mismos antes y después (15, 8 o 7 en paralelo): solo cambia la
serialización.

```bash
python bench_n_jobs.py run --cpus 16 8 --estimators xgb hgb lgbm_1t --reps 2 --blocks ms_refit ms_norefit --max-priority 4 --out <carpeta>
```

50 series diarias. Sin refit: 52 folds de 7 pasos. Con refit: 8 folds de 28
pasos. Mediana de 2 repeticiones; segundos en frío / en caliente, con
`n_jobs=cpu_count() - 1` (tope de un proceso por fold). `fit después`: ajuste
de XGBoost en el proceso principal después del backtesting en paralelo,
relativo al mismo ajuste después del backtesting con un proceso.

| CPU | Caso | Estimador | Antes | Después | Un proceso | `fit después` |
|---|---|---|---|---|---|---|
| 16 | sin refit (15 procesos) | XGBoost | 6,7 / 8,9 | 6,3 / 3,2 | 3,3 / 3,2 | 5,3x → 1,0x |
| 16 | sin refit | HistGB | 8,0 / 5,7 | 6,4 / 3,2 | 4,4 / 4,3 | |
| 16 | sin refit | LightGBM, 1 hilo | 10,5 / 9,1 | 7,9 / 4,8 | 5,2 / 5,1 | |
| 16 | con refit (8 procesos) | XGBoost | 7,5 / 10,1 | 7,5 / 5,4 | 9,4 / 9,5 | 5,1x → 1,0x |
| 16 | con refit | HistGB | 5,6 / 3,1 | 5,2 / 3,0 | 6,5 / 6,4 | |
| 16 | con refit | LightGBM, 1 hilo | 9,3 / 7,2 | 8,9 / 6,6 | 23,9 / 23,9 | |
| 8 | sin refit (7 procesos) | XGBoost | 6,3 / 6,1 | 5,5 / 3,2 | 3,4 / 3,3 | 2,9x → 1,0x |
| 8 | sin refit | HistGB | 7,5 / 5,0 | 5,6 / 3,2 | 4,3 / 4,2 | |
| 8 | sin refit | LightGBM, 1 hilo | 10,6 / 7,8 | 7,1 / 4,9 | 5,2 / 5,1 | |
| 8 | con refit (7 procesos) | XGBoost | 8,4 / 8,2 | 8,2 / 5,9 | 10,1 / 10,1 | 2,9x → 1,0x |
| 8 | con refit | HistGB | 5,7 / 3,4 | 5,4 / 3,2 | 6,5 / 6,5 | |
| 8 | con refit | LightGBM, 1 hilo | 9,2 / 7,2 | 8,8 / 6,6 | 23,9 / 23,9 | |

## Lectura

- **XGBoost:** el mismo síntoma que la propuesta 1 corrigió en una serie. Antes,
  la segunda llamada en paralelo era más lenta que la primera (8,9 frente a 6,7
  sin refit; 10,1 frente a 7,5 con refit) y los ajustes posteriores de XGBoost
  en la sesión, hasta 5,3 veces más lentos. Después, en caliente, de 8,9 a 3,2
  sin refit y de 10,1 a 5,4 con refit, y el ajuste posterior no cambia.
- **Sin refit, en caliente,** paralelo pasa de perder a empatar o ganar: de 1,6
  a 2,8 veces más rápido que antes. En frío sigue perdiendo frente a un proceso
  (arranque de los workers), y `'auto'` sigue eligiendo un proceso. Con
  HistGB, en caliente, paralelo gana ahora 1,33 veces a un proceso (3,2 frente
  a 4,3): es el único caso medido en que `'auto'` queda por detrás del mejor.
  No compensa cambiar la regla: en frío pierde 1,5 veces, y la predicción total
  (4,3 s) queda por debajo del mínimo de la propuesta 3 (6 s en Linux).
- **Con refit,** `'auto'` ya usaba el paralelo (8 o 7 procesos), así que la
  mejora le llega directamente: XGBoost de 10,1 a 5,4 en caliente con 16 CPU.
  HistGB y LightGBM de un hilo ganan entre un 3 y un 8 % en caliente.
- **Controles:** las medidas con un proceso son iguales antes y después
  (diferencias por debajo del 3 %, salvo una primera llamada de HistGB con
  8 CPU, 4,9 frente a 4,3). Predicciones idénticas en todas las medidas.
