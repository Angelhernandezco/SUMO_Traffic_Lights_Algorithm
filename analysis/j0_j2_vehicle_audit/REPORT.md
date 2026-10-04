# Auditoría de los 42 vehículos rectos J0 → J2, V1

## Reproducción y definiciones

Se reprodujo sin cambiar el controlador V1: mismo checkpoint PPO determinista, demanda, semilla 42, horizonte 3600 s, `T_J0_J2=9 s`, J2 dinámico V1 y J10/J16 fijos. La instrumentación fue sólo de lectura. Se obtuvieron exactamente los resultados validados: 42 vehículos rectos `-E0→E1→E5`, 18 con alguna detención en E1 y 35 vehículo-s detenidos.

Los tiempos están en segundos SUMO, muestreados a 1 s. **Cruce de J0** es la primera muestra en el enlace interno `:J0_6/7/8`; la salida de J0 a E1 figura por separado en `per_vehicle.csv`. **Llegada a la línea de J2** es la primera muestra a ≤5 m de la señal en E1; si el vehículo atravesó esos 5 m entre dos muestras, se usa la primera muestra en el enlace interno de J2. Nueve de 42 llegadas usan esa segunda regla y están marcadas en el CSV. El estado `r/G` de la tabla es el del enlace recto E1→E5 en ese instante; la clasificación de una parada usa además el estado **al primer segundo detenido**, porque un vehículo puede detenerse detrás de una cola y llegar físicamente a la línea cuando ya hay verde. Se consideró detención `velocidad <0,1 m/s` en E1.

La cola previa se identificó cuando, al primer segundo detenido, había un líder a ≤5 m de separación con velocidad <3 m/s. `R` = detención directa ante rojo sin líder lento cercano; `CR` = cola con rojo; `CV` = cola que persiste cuando el enlace ya está verde; `—` = ninguna detención. La categoría «fin del verde» requería una primera detención tras haber observado verde durante la aproximación seguida de amarillo/rojo; no apareció ningún caso.

## Cada vehículo

| Vehículo | Cruza J0 | Llega línea J2 | Estado | Cruza J2 | Causa |
|---:|---:|---:|:---:|---:|:---:|
| 50 | 139 | 153 | G | 154 | — |
| 63 | 142 | 157 | G | 157 | — |
| 112 | 268 | 280 | r | 283 | R |
| 162 | 338 | 351 | G | 352 | — |
| 184 | 406 | 418 | G | 418 | — |
| 254 | 534 | 551 | G | 552 | — |
| 259 | 537 | 553 | G | 554 | — |
| 299 | 609 | 617 | G | 617 | — |
| 342 | 737 | 747 | G | 748 | — |
| 358 | 737 | 750 | G | 750 | — |
| 413 | 866 | 883 | G | 884 | CR |
| 454 | 935 | 947 | G | 947 | — |
| 478 | 991 | 1008 | G | 1010 | CV |
| 483 | 991 | 1000 | r | 1006 | R |
| 587 | 1195 | 1207 | G | 1208 | — |
| 620 | 1256 | 1272 | G | 1273 | — |
| 687 | 1395 | 1407 | G | 1408 | — |
| 704 | 1464 | 1476 | G | 1477 | — |
| 726 | 1464 | 1478 | G | 1479 | — |
| 778 | 1595 | 1612 | G | 1612 | — |
| 901 | 1849 | 1866 | G | 1867 | CR |
| 918 | 1849 | 1859 | r | 1864 | R |
| 954 | 1923 | 1935 | G | 1936 | — |
| 962 | 1981 | 1998 | G | 1999 | CR |
| 992 | 1995 | 2004 | G | 2004 | — |
| 995 | 2044 | 2064 | G | 2065 | CR |
| 1002 | 2044 | 2056 | r | 2059 | — |
| 1025 | 2101 | 2118 | G | 2119 | CR |
| 1056 | 2156 | 2168 | r | 2172 | R |
| 1060 | 2156 | 2168 | r | 2172 | R |
| 1061 | 2160 | 2174 | G | 2175 | CV |
| 1086 | 2227 | 2249 | G | 2249 | CV |
| 1284 | 2613 | 2625 | G | 2626 | — |
| 1309 | 2679 | 2693 | G | 2693 | — |
| 1374 | 2803 | 2816 | r | 2818 | R |
| 1422 | 2866 | 2878 | r | 2881 | R |
| 1448 | 2931 | 2943 | r | 2946 | R |
| 1462 | 2935 | 2948 | G | 2949 | CR |
| 1466 | 2997 | 3009 | r | 3012 | — |
| 1495 | 3002 | 3011 | r | 3012 | — |
| 1567 | 3179 | 3192 | r | 3195 | R |
| 1628 | 3311 | 3325 | G | 3326 | — |

## Por qué se detuvieron los 18

| Causa inmediata | Vehículos | Segundos detenidos en E1 | Evidencia al primer alto |
|---|---:|---:|---|
| Llegada directa antes del verde (`R`) | 9 | 16 | Enlace rojo, detención a ~1 m de la señal, sin líder lento cercano. |
| Cola previa aún con rojo (`CR`) | 6 | 15 | Enlace rojo, líder detenido a ≤0,03 m de separación; el vehículo se detuvo a 8,5–16,0 m de la señal. |
| Cola previa durante el verde (`CV`) | 3 | 4 | Enlace verde, líder lento a ≤2,4 m; la cola se había formado antes o durante la transición al verde. |
| Fin del verde | 0 | 0 | Ninguna primera detención al pasar de verde a amarillo/rojo. |
| Otra causa | 0 | 0 | No se observó una primera detención sin señal o líder que la explique. |

En el instante de **primera detención**, 15/18 veían rojo y 3/18 veían verde detrás de un líder lento. Cuando alcanzaron físicamente la proximidad de la línea, 9 detenidos veían rojo y 9 verde: estos últimos ya habían esperado antes en una cola. En los 42 vehículos, el estado al llegar a la línea fue `r` para 12 y `G` para 30; tres de los que vieron `r` allí no llegaron a detenerse porque el verde comenzó antes de su cruce. Los 42 cruzaron finalmente J2 durante fase 2 con señal `G`.

**Conclusión:** los 18 altos restantes no se deben a un verde de J2 demasiado corto ni a su final. Nueve son alcance directo del rojo antes del próximo verde y nueve son propagación de una cola previa; seis de esos nueve frenan mientras sigue el rojo y tres cuando ya hay verde. Las detenciones suman 35 s, unos 1,94 s por vehículo detenido. Esta clasificación describe el mecanismo observado a resolución de 1 s; no atribuye cada cola a una única causa más allá de su señal y líder medidos.

Los datos verificables están en `per_vehicle.csv` (42 registros, tiempos y causas) y `trajectory_1s.csv` (2108 muestras de posición, velocidad, señal y líder).
