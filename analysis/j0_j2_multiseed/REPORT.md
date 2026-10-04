# Validación de T_J0_J2 en V1 con cinco semillas de demanda

## Protocolo

Se conservaron sin cambios V1, el PPO determinista, la red y sus offsets, el horizonte de 3600 s y el paso de 1 s. El generador del proyecto `tools/randomTrips.py` produjo demandas con periodo 2 s, intervalo 0–3600 s y semillas **42–46**; se enrutan con `duarouter --validate`. La demanda generada con semilla 42 coincide vehículo por vehículo y ruta por ruta con `maps/master_slave.rou.xml` (1126 vehículos). En SUMO se mantuvo `--seed 42` en las 20 corridas para aislar la variación de demanda. Cada semilla usó **su mismo archivo de rutas** para T=9, 10, 11 y 12 s. Ninguna corrida dejó sin completar un vehículo recto de la cohorte.

El tiempo J0→J2 se mide igual que en el experimento V1 previo: primera muestra en E1 hasta primera muestra en E5. Waiting E1 es la suma de segundos con velocidad <0,1 m/s de la cohorte recta `-E0→E1→E5`. Waiting secundario J2 suma los vehículos detenidos por segundo en los accesos secundarios `-E5`, `-E6` y `-E7`. Los archivos de demanda y la tabla por vehículo se conservan junto a este informe.

| Semilla | T (s) | Rectos sin parar | Waiting E1 (vehículo-s) | Tiempo E1→E5 medio (s) | Waiting secundario J2 (vehículo-s) |
|---:|---:|---:|---:|---:|---:|
| 42 | 9 | 24/42 (57,14 %) | 35 | 14,12 | 10167 |
| 42 | 10 | 27/42 (64,29 %) | 24 | 14,43 | 10246 |
| 42 | 11 | 31/42 (73,81 %) | 19 | 14,52 | 9999 |
| 42 | 12 | 29/42 (69,05 %) | 23 | 14,17 | 10145 |
| 43 | 9 | 39/50 (78,00 %) | 19 | 13,32 | 9345 |
| 43 | 10 | 39/50 (78,00 %) | 22 | 13,48 | 9305 |
| 43 | 11 | 33/50 (66,00 %) | 34 | 13,90 | 9831 |
| 43 | 12 | 34/50 (68,00 %) | 30 | 14,14 | 9708 |
| 44 | 9 | 33/48 (68,75 %) | 24 | 13,58 | 9780 |
| 44 | 10 | 30/48 (62,50 %) | 28 | 14,33 | 10127 |
| 44 | 11 | 29/48 (60,42 %) | 45 | 14,75 | 9698 |
| 44 | 12 | 30/48 (62,50 %) | 28 | 14,23 | 9978 |
| 45 | 9 | 31/45 (68,89 %) | 30 | 14,31 | 9087 |
| 45 | 10 | 27/45 (60,00 %) | 28 | 14,91 | 8630 |
| 45 | 11 | 31/45 (68,89 %) | 25 | 13,51 | 8767 |
| 45 | 12 | 36/45 (80,00 %) | 14 | 13,40 | 8670 |
| 46 | 9 | 33/47 (70,21 %) | 30 | 14,13 | 8643 |
| 46 | 10 | 27/47 (57,45 %) | 46 | 14,55 | 8659 |
| 46 | 11 | 30/47 (63,83 %) | 37 | 14,87 | 9039 |
| 46 | 12 | 27/47 (57,45 %) | 36 | 15,15 | 8942 |

## Media y variabilidad entre semillas

Los valores son **media ± desviación estándar muestral** de las cinco corridas por T. Cada semilla aporta una observación; las cohortes rectas son de 42, 50, 48, 45 y 47 vehículos respectivamente.

| T (s) | Sin parada (%) | Waiting E1 total (vehículo-s) | Waiting E1 por recto (s) | Tiempo E1→E5 medio (s) | Waiting secundario J2 (vehículo-s) |
|---:|---:|---:|---:|---:|---:|
| 9 | **68,60 ± 7,46** | 27,6 ± 6,2 | 0,60 ± 0,17 | **13,89 ± 0,42** | 9404 ± 593 |
| 10 | 64,45 ± 8,00 | 29,6 ± 9,5 | 0,64 ± 0,20 | 14,34 ± 0,53 | **9393 ± 774** |
| 11 | 66,59 ± 5,09 | 32,0 ± 10,2 | 0,68 ± 0,19 | 14,31 ± 0,58 | 9467 ± 534 |
| 12 | 67,40 ± 8,44 | **26,2 ± 8,3** | **0,56 ± 0,16** | 14,22 ± 0,62 | 9489 ± 650 |

## Interpretación

**T=11 s no conserva la ventaja de la semilla 42.** Frente a T=9, su diferencia emparejada de progresión es +16,67 puntos en la semilla 42, pero −12,00, −8,33 y −6,38 puntos en las semillas 43, 44 y 46; empata en la 45. El cambio medio emparejado es −2,01 puntos con desviación estándar de 11,31 puntos. T=11 logra la mayor progresión y el menor waiting E1 sólo en la semilla 42.

El mejor T según progresión y waiting E1 es **9** en 43, 44 y 46; **11** en 42; y **12** en 45. T=9 tiene la mejor media de vehículos sin detenerse y de tiempo E1→E5; T=12, la menor media de waiting E1. Las diferencias medias de waiting secundario entre T son pequeñas frente a su dispersión entre demandas. Por tanto, entre estos cuatro valores no aparece un óptimo universal: el resultado depende de la semilla. Para una referencia única de esta pequeña muestra, T=9 tiene el mejor promedio de progresión, pero no demuestra una superioridad robusta frente a T=12 en todas las demandas.

La comparación mantiene fijo el modelo PPO y la semilla interna de SUMO, pero J0 responde al tráfico y puede producir distintas liberaciones al variar T. Cinco demandas permiten detectar la falta de robustez de la ventaja observada para T=11; no son suficientes para afirmar un óptimo definitivo fuera de estas semillas.
