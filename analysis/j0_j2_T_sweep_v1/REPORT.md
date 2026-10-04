# Barrido de T en V1, J0 → J2

Se ejecutó la misma V1 sin modificar su lógica para `T_J0_J2 = 9, 10, 11, 12, 13, 14, 15 s`. Cada corrida usó demanda `maps/master_slave.rou.xml`, semilla 42, 3600 s, el mismo checkpoint PPO determinista de J0 y los mismos offsets fijos de J10/J16. Se verificaron 42 vehículos rectos `-E0→E1→E5` en las siete corridas, con los mismos IDs. La simulación valida programa y duraciones de fases y comprueba que J10/J16 sigan fijos.

El **tiempo J0→J2** mantiene la definición de las comparaciones V1 previas: primera muestra en E1 hasta primera muestra en E5. La **cobertura** es el porcentaje de segundos de todas las ventanas previstas por la acción PPO desplazadas `+T` que coinciden con verde recto de J2; excluye ventanas incompletas al horizonte. El waiting secundario suma vehículos detenidos por segundo en los accesos secundarios de J2.

| T (s) | Sin parada en J2 | Waiting E1 (vehículo-s) | Tiempo medio E1→E5 (s) | Cobertura ventana | Ventanas completas | Waiting secundario J2 (vehículo-s) |
|---:|---:|---:|---:|---:|---:|---:|
| 9 | 24/42 (57,14 %) | 35 | 14,12 | 64,98 % | 16/56 | 10167 |
| 10 | 27/42 (64,29 %) | 24 | 14,43 | 74,45 % | 21/55 | 10246 |
| **11** | **31/42 (73,81 %)** | **19** | 14,52 | 79,74 % | 22/55 | **9999** |
| 12 | 29/42 (69,05 %) | 23 | 14,17 | 84,66 % | 24/55 | 10145 |
| 13 | 22/42 (52,38 %) | 35 | 15,24 | 89,48 % | 27/55 | 10256 |
| 14 | 25/42 (59,52 %) | 32 | 14,81 | 92,84 % | 28/54 | 10199 |
| 15 | 11/42 (26,19 %) | 64 | 16,60 | 97,79 % | 51/55 | 10477 |

En esta semilla, **T=11 s es el mejor equilibrio**: frente a T=9, siete vehículos adicionales cruzaron sin detenerse (+16,67 puntos), el waiting de E1 bajó de 35 a 19 vehículo-s (−45,7 %) y el secundario de J2 bajó de 10167 a 9999 (−1,65 %). Su tiempo medio E1→E5 fue 0,40 s mayor. T=12 s es la alternativa cercana: dos vehículos menos sin parada y cuatro vehículo-s más de waiting de E1 que T=11, pero 0,35 s menos de tiempo medio y 4,92 puntos más de cobertura.

La cobertura crece casi monótonamente hasta T=15, pero la progresión real no: T=15 produjo 51/55 ventanas teóricas completas y aun así sólo 11/42 cruces sin parada. Por tanto, maximizar la superposición con toda la ventana de acción PPO no selecciona el mejor T para los vehículos de esta demanda. Es compatible con que el verde de J2 se desplace demasiado tarde para parte del pelotón; esta comparación por sí sola no mide la llegada a la línea en cada T.

Los resultados son deterministas para una sola semilla. Aunque demanda, checkpoint, red y IDs fueron idénticos, la realimentación del tráfico puede cambiar las acciones de J0 entre corridas (56 liberaciones rectas con T=9 y 55 con los demás). No se infiere robustez entre semillas. Los datos están en `results.csv`, `results.json` y `per_vehicle.csv`.
