# PPO en J0 frente a la onda fija equilibrada

> **Contexto histórico.** Esta corrida precede a la corrección permanente de la red y usa el checkpoint `test1`, distinto del oficial `test36_3`. Hoy J0/J2/J10/J16 tienen sólo el programa `0` de 76 s en `maps/master_slave.net.xml`; las cifras siguientes describen la corrida original, no una nueva evaluación del checkpoint oficial.

## Configuración verificable

Rama `master-slave-clean-v2`, SUMO 1.26.0, rutas `maps/master_slave.rou.xml`, semilla 42, paso de 1 s. La comparación principal duró 3 600 s (demanda completa); se repitió también con horizonte de 2 000 s. La copia temporal de la red mantuvo J2/J10/J16 en programa `0`, ciclo de 76 s y offsets **0/72/24**. En la red original usada entonces, J10 programa `0` aún tenía 90 s: sólo en la copia temporal se sustituyó su secuencia por cuatro verdes de 15 s y cuatro amarillos de 4 s. Se eliminaron los programas `1` de esa copia para asegurar que SUMO seleccionara `0`. La red original no cambió durante esa prueba.

El checkpoint local elegido es `policy/models/model_future_v39_yellow_test1.pth` (SHA-256 `2ae989372805342d39eb80d53576669ab1080f9a927d7858929131780276a657`). Sus metadatos declaran J0, fases `[0,2,4,6]`, observación de 32 valores, `min_green=5`, `max_green=45`, amarillo separado, y mejor waiting determinista histórico 10 926. La inferencia usa la media determinista de la Beta del `PPOAgent` original, la misma normalización almacenada y el mismo mapeo de acción a duración. El script actualiza **solo J0**: verde decidido por PPO, seguido de 4 s de amarillo; conserva el cursor cíclico de cuatro fases. No hay entrenamiento, predictor ni control sobre los slaves. Una segunda corrida con horizonte de 2 000 s produjo exactamente el mismo prefijo de liberaciones y duraciones de J0.

TraCI confirmó que los verdes rectos de los slaves siguieron sin deriva durante los 3 600 s: J2 empezó en `20, 96, 172…`, J10 en `35, 111, 187…` y J16 en `44, 120, 196…`, todos separados exactamente 76 s. J0 PPO inició su recto **63 veces**; la referencia fija de J0 tiene 48 inicios dentro del horizonte.

## Registro de liberaciones y llegadas

`ppo_releases_3600.csv` contiene una fila por cada uno de los 63 inicios rectos de J0: instante real, duración de verde, acción continua, desplazamiento respecto del inicio fijo más cercano, ventana fija esperada de 15 s en cada slave, cantidad de vehículos asociados, mediana de llegadas y paradas. `ppo_vehicles_3600.csv` enlaza por `release_index` cada vehículo que viene del recto `-E0→E1` con las tres ventanas, su llegada observada, cruce, parada y error temporal.

La **llegada observada** es el primer instante en que el vehículo se detiene sobre el enlace de aproximación o entra en los últimos 30 m antes del TLS, lo que ocurra primero. El error firmado respecto de la ventana es 0 si llega dentro, negativo si llega antes del inicio y positivo si llega después del final. Para cada liberación se elige la ventana fija más cercana a su tiempo de llegada libre aproximado (9, 19 y 29 s después de J0 en J2, J10 y J16); esos tiempos solo sirven para el análisis offline, nunca para controlar señales. Los vehículos cuya ruta termina en E10 no se cuentan como cruces ni llegadas a J16.

| Error temporal, 3 600 s | n | Media | Mediana | P90 | Mínimo | Máximo |
|---|---:|---:|---:|---:|---:|---:|
| Inicio recto PPO frente al inicio fijo más cercano, módulo 76 s | 63 | −1,17 s | +1 s | +30,8 s | −37 s | +37 s |
| Llegada a J2 fuera de la ventana de su liberación | 24 | +2,46 s | 0 s | +22,4 s | −27 s | +27 s |
| Llegada a J10 fuera de la ventana de su liberación | 24 | +27,17 s | +27,5 s | +56,7 s | −6 s | +60 s |
| Llegada a J16 fuera de la ventana de su liberación | 19 | +27,42 s | 0 s | +65,6 s | 0 s | +70 s |

El error **absoluto** de inicio PPO tiene mediana 23 s, P90 **35 s** y máximo 37 s. Entre liberaciones consecutivas, el cambio absoluto de fase tiene mediana 19 s, P90 **27,9 s** y máximo 28 s. La separación entre liberaciones PPO tiene media 57,74 s, mediana 57 s y rango 48–73 s; el ciclo fijo es 76 s. Comparar la liberación PPO número *i* con la ventana fija número *i* acumula hasta −1 139 s al final porque PPO libera 15 veces más en el horizonte. Ese valor acumulado expresa una **diferencia de cadencia**, no un desfase que deba corregirse con un salto de 1 139 s.

Los errores de J10 y J16 respecto de la ventana original pueden incluir una vuelta completa del ciclo tras esperar en J2. Como diagnóstico **local**, respecto del verde fijo más cercano a la llegada real, el error absoluto P90 es 25,8 s en J2, 6 s en J10 y 0 s en J16. Esto separa la desalineación inicial de la espera propagada aguas abajo.

## Efecto sobre el tráfico

Vehículo del corredor: ruta que contiene `E1 E5 E10` consecutivamente. Las tasas y tiempos usan los 100 vehículos que salen antes de 3 420 s; 88 atraviesan J16. Las paradas son episodios de velocidad menor de 0,1 m/s en la aproximación a J0 y los tres enlaces del corredor. Waiting es la suma de vehículo-segundos detenidos en las aproximaciones controladas de J0/J2/J10/J16; 104 vehículos del corredor y 1 021 de las otras rutas se observaron en ambas corridas.

| Métrica, 3 600 s | Onda fija equilibrada | J0 PPO + slaves fijos | Cambio |
|---|---:|---:|---:|
| Sin parada en J2 | 25/100 (25%) | 13/100 (13%) | −12 puntos |
| Sin parada en J10 | 60/100 (60%) | 62/100 (62%) | +2 puntos |
| Sin parada en J16 | 86/88 (97,73%) | 88/88 (100%) | +2,27 puntos |
| Sin parada en los tres | 17/88 (19,32%) | 9/88 (10,23%) | −9,09 puntos |
| Paradas por vehículo, media | 2,82 | 2,89 | +0,07 |
| Tiempo J0→J16, media | 56,93 s | 64,81 s | +7,88 s (+13,8%) |
| Tiempo J0→J16, mediana | 49 s | 69 s | +20 s |
| Waiting del corredor | 3 463 veh·s | 3 743 veh·s | +8,1% |
| Waiting del resto de rutas | 44 399 veh·s | 43 210 veh·s | −2,7% |

Para el recto que entra a J0 desde `-E0` y atraviesa los tres slaves, la progresión completa cae de **16/19 a 1/19** (84,21% a 5,26%). Su tiempo medio J0→J16 sube de 40,63 a 60,47 s. El efecto se concentra en J2: la media del tramo J0→J2 para todo el corredor pasa de 29,01 a 39,59 s; después de esperar allí, muchos vehículos alcanzan ventanas posteriores de J10 y J16. En 2 000 s se observa el mismo sentido: 18,75%→10,42% de progresión completa y 58,62→66,48 s de tiempo medio.

## Magnitud de corrección y un Δ común

Para seguir **la fase** de una liberación de J0 respecto de una onda de 76 s, un futuro sincronizador tendría que cubrir aproximadamente **±35 s para el P90** y hasta **±37–38 s** en esta corrida. Además, la consigna cambiaría hasta unos **28 s entre liberaciones consecutivas** y J0 produce liberaciones cada ~58 s frente a ciclos esclavos de 76 s. Esa diferencia de frecuencia no se resuelve con un offset constante ni con un único traslado de toda la onda.

Como prueba geométrica offline, se buscó un mismo Δ módulo 76 que colocara las **llegadas observadas** de cada vehículo dentro de las tres ventanas desplazadas de 15 s. Existe para **8/19 vehículos** que cruzan J16 y para **5/16 liberaciones** con esos vehículos (exigiendo que un solo Δ cubra todos los vehículos de la liberación). Este cálculo no simula los efectos causales de mover un slave: al cambiar J2 también cambiarían las llegadas a J10 y J16. Sí indica que **un mismo Δ aplicado a los tres slaves no basta como estrategia general** para las trayectorias medidas. Puede servir como ajuste inicial de fase del primer receptor, pero la cadencia distinta y la espera propagada exigen evaluar cada ventana aguas abajo por separado antes de diseñar control dinámico.

La comparación es determinista para esta demanda y semilla. No se ha evaluado robustez ante otras rutas, semillas o checkpoints.
