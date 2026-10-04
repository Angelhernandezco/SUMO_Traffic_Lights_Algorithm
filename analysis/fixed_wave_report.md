# Sincronización fija J0→J2→J10→J16

> **Contexto histórico de red.** Al ejecutar este experimento, J10 tenía un programa de 90 s en la red original y se normalizó en una copia temporal. La red **actual** de `master-slave-clean-v2` ya tiene sólo el programa `0` de 76 s en J0/J2/J10/J16. Los resultados de la onda fija se conservan como evidencia experimental.

## Red y fase recta

En la copia de trabajo actual de `maps/master_slave.net.xml`, los programas `0` de J0, J2, J10 y J16 son idénticos como secuencia de estados y duraciones: cuatro verdes de 15 s, cada uno seguido de 4 s de amarillo; ciclo de 76 s. El recto norte usa los índices de señal 6–8 y la fase de índice 2 en J0 (`-E0→E1`), J2 (`E1→E5`) y J16 (`E10→E13`). En J10 (`E5→E10`) usa los índices 11–13 y la fase de índice 4, porque allí el orden de conexiones está rotado.

En la red original de aquella corrida, SUMO activaba por defecto el programa `1` en J0 y J10. El de J0 duplicaba el programa `0`, pero el de J10 tenía dos verdes de 41 s y dos amarillos de 4 s (90 s por ciclo). Por eso la ejecución directa de `configuration.sumocfg` **en aquel estado de red** no era una referencia de cuatro ciclos iguales. El script generó una copia temporal que conservaba exclusivamente los cuatro programas `0`; tanto baseline como variantes usaron exactamente esa copia. No se editó la red original para esta prueba.

## Método

- SUMO 1.26.0, demanda fija `maps/master_slave.rou.xml`, paso de 1 s, semilla 42. Sin PPO, entrenamiento, predictor ni control durante la simulación. TraCI solo observa. Los únicos parámetros ensayados son tres atributos XML `offset` estáticos.
- Se buscaron 92 combinaciones únicas en simulaciones de 2 000 s, primero alrededor de los tiempos de recorrido y después con ajustes finos y comprobaciones de borde. El ganador por porcentaje de vehículos que pasan J2, J10 y J16 sin parar (desempate por menor tiempo medio J0→J16) y una alternativa con menor perjuicio secundario se volvieron a medir a 3 600 s con la demanda completa. La primera malla exploratoria usó el signo contrario del offset; se corrigió tras comprobar los cambios de fase observados.
- Vehículo del corredor: ruta que contiene `E1 E5 E10` consecutivamente. Para tasas y tiempos se usan los 100 vehículos con salida antes de 3 420 s; 88 cruzan J16, y los 12 restantes terminan en E10. Para espera se cuentan todos los vehículos de esa ruta observados (104), frente a 1 021 vehículos de las demás rutas.
- Cruce sin parada: velocidad menor de 0,1 m/s **nunca** observada en el enlace de llegada (`E1`, `E5` o `E10`). Cada tramo empieza al entrar en su primer enlace y termina al entrar en el siguiente; J10→J16 termina al entrar en el enlace posterior a E10. Las paradas por vehículo cuentan episodios de velocidad menor de 0,1 m/s en la aproximación a J0 y en los tres enlaces del corredor, entre los 88 que cruzan J16.
- Waiting: vehículo-segundos con velocidad menor de 0,1 m/s sobre las aproximaciones controladas por estos cuatro TLS. Se separa por ruta del corredor/resto de rutas; no es el waiting global de toda la red.

## Tiempos reales medidos

Baseline fijo sin desfase, 3 600 s. La dispersión es la desviación estándar muestral; P10–P90 ayuda a ver la cola.

| Tramo | n | Mediana | Media | Desv. estándar | P10–P90 | Sin parada |
|---|---:|---:|---:|---:|---:|---:|
| J0→J2 | 100 | 22 s | 28,97 s | 19,03 s | 8–57 s | 26/100; mediana 9 s, media 9,23 s, DE 2,37 s |
| J2→J10 | 100 | 19 s | 18,30 s | 3,60 s | 12–22 s | 20/100; mediana 12 s, media 12,60 s, DE 3,07 s |
| J10→J16 | 88 | 56 s | 55,35 s | 2,16 s | 53–58 s | 0/88; no existe estimación libre de parada en el baseline |

Con la onda elegida, los recorridos reales J2→J10 y J10→J16 tienen medianas de 10 s; en el último tramo 87/88 circulan sin parar (media 10,17 s, DE 1,31 s para esos 87). Esto confirma que el tiempo libre del último enlace es del orden de 10 s; los 56 s del baseline eran casi totalmente espera semafórica.

## Offsets y comparación final

Mejor progresión de las 92 configuraciones probadas: **J2=8 s, J10=74 s (equivalente a −2 s), J16=26 s**, respecto de J0=0 y módulo 76 s. Una traza real de fases da inicios del verde recto en **t=20, 28, 37 y 46 s**, repetidos cada 76 s. La alternativa más equilibrada comprobada usa **J2=0, J10=72, J16=24 s**.

| Métrica, 3 600 s | Sin desfase | Mejor progresión | Alternativa equilibrada |
|---|---:|---:|---:|
| Sin parar en J2 | 26/100 (26%) | 25/100 (25%) | 25/100 (25%) |
| Sin parar en J10 | 20/100 (20%) | 100/100 (100%) | 60/100 (60%) |
| Sin parar en J16 | 0/88 (0%) | 87/88 (98,86%) | 86/88 (97,73%) |
| Sin parar en los tres | 0/88 (0%) | 19/88 (21,59%) | 17/88 (19,32%) |
| Paradas por vehículo, media | 4,22 | 2,38 (−43,6%) | 2,82 (−33,2%) |
| J0→J16, mediana | 97 s | 50 s | 49 s |
| J0→J16, media | 103,94 s | 54,31 s (−47,8%) | 56,93 s (−45,2%) |
| Waiting de rutas del corredor | 7 298 veh·s | 3 542 veh·s (−51,5%) | 3 463 veh·s (−52,5%) |
| Waiting del resto de rutas | 43 097 veh·s | 52 333 veh·s (+21,4%) | 44 399 veh·s (+3,0%) |

En el subgrupo recto que llega a J0 desde `-E0` y después recorre el corredor (19 vehículos), los tres cruces sin parada pasan de 0/19 a **18/19 (94,74%)** con la mejor progresión, y a 16/19 (84,21%) con la alternativa equilibrada. La mediana J0→J16 de ese subgrupo cae de 79 a 34 s con la mejor progresión.

El primer tramo del conjunto amplio **no mejora** con la mejor progresión: J0→J2 pasa de mediana 22 a 29 s, y de media 28,97 a 32,11 s. Muchos vehículos de ese conjunto llegan a J0 desde otras fases o empiezan en E1. El porcentaje de progresión completa del 21,59% no representa por sí solo el éxito para el recto norte, ni implica que todo el tráfico se beneficie.

## Conclusión y alcance

La onda fija **sí sirve como referencia experimental para la progresión recta**: con la misma demanda y ciclo, casi todos los vehículos del subgrupo recto avanzan sin parar de J0 a J16. La combinación de máxima progresión desplaza espera al resto de rutas (+21,4%), por lo que no conviene tratarla como óptimo de la red. La alternativa `0/72/24` conserva casi toda la reducción de tiempo y waiting del corredor con +3,0% de waiting secundario; es una referencia más razonable para evaluar más adelante si PPO puede mejorar el balance. Estos son óptimos **entre las combinaciones probadas**, con una demanda y una semilla, no una garantía general para otros patrones de tráfico.

Los valores históricos `OFF` de los documentos de contexto no se comparan numéricamente aquí: esta prueba aseguró explícitamente cuatro programas `0` de 76 s, mientras la red anterior a la corrección permanente activaba el programa `1` de 90 s de J10.

Reproducción: `python analysis/fixed_wave.py --audit` y `python analysis/fixed_wave.py --offsets 8 74 26 --end 3600`. Los resultados completos están en los JSON `fixed_wave_*_3600.json`; las 92 configuraciones buscadas están en `fixed_wave_search.csv`.
