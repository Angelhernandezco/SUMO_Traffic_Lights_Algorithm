# Contexto actual de sincronización J0→J2→J10→J16

## Red y control

El objetivo es sincronizar la progresión recta **J0→J2→J10→J16** sin cambiar el PPO de J0. En `maps/master_slave.net.xml`, los cuatro TLS tienen **únicamente el programa `0`**, con los mismos ocho estados y el mismo orden: cuatro verdes de 15 s alternados con cuatro amarillos de 4 s; **ciclo de 76 s** y offset XML `0` en cada uno. No existe ahora un programa de 90 s en J10. Los offsets citados abajo pertenecen a configuraciones experimentales, no a la red base.

J0 usa evaluación determinista del checkpoint oficial `policy/models/model_future_v39_yellow_test36_3.pth`. Su única acción PPO decide la duración del verde actual; las fases avanzan en orden fijo. Cada verde respeta **5–45 s** y cada amarillo dura **4 s**. **V1, V2 y V3 coordinan los tres receptores J2/J10/J16** mediante `run_sync.py` y los controladores de `sync/`. El ejecutor admite únicamente el corredor completo. Las pruebas históricas que coordinaban sólo J2 se resumen en [CORRIDOR_HISTORY.md](../analysis/CORRIDOR_HISTORY.md). El corredor avanza **sur→norte**, por `-E0→E1→E5→E10→E13`; fase recta XML 2 en J0/J2/J16 y 4 en J10. No se coordina el sentido contrario.

## Evidencia y límite actual

- **Onda fija:** entre los offsets estáticos probados respecto a J0, la mejor progresión fue **J2/J10/J16 = 8/74/26 s**; la referencia más equilibrada fue **0/72/24 s**. En la cohorte recta que atraviesa el corredor, pasaron los tres TLS sin parar **18/19** y **16/19**, respectivamente, frente a **0/19** sin desfase. La onda fija sí produce progresión, aunque 8/74/26 elevó el waiting secundario.
- **PPO en J0 con slaves fijos:** sus liberaciones cambian de instante y cadencia respecto al ciclo fijo. En la prueba histórica con **otro checkpoint** (`test1`), la progresión completa de la cohorte recta bajó de **16/19 a 1/19** frente a 0/72/24; esta cifra muestra la ruptura de la onda, pero no es una comparación directa del checkpoint oficial.
- **V1, J0→J2:** desde el **inicio del verde recto de J0**, J2 intenta abrir en `inicio_J0 + T`, acortando sólo verdes no-corredor pendientes. Con **T≈11 s** como referencia de comparación en cinco demandas, **154/232 (66,38 %)** vehículos rectos `-E0→E1→E5` pasaron J2 sin detenerse. El mejor T varió por semilla: 11 s no es un óptimo general demostrado.
- **V2 causal histórica, retirada de la comparación activa:** el primer cruce real de J0 fijaba el frente en `cruce + 12 s`; cruces posteriores extendían la cola, sin suponer un tamaño de pelotón. Logró **114/232 (49,14 %)** sin parada, **17,24 puntos menos** que V1. El `+12 s` aproxima razonablemente el viaje libre (mediana de 12 s para quienes no pararon), pero **153/205** frentes eran legalmente inalcanzables al conocerlos. V3 conserva las operaciones causales que necesita dentro de `sync/v3.py`.

Los antecedentes, sus límites y las decisiones derivadas se conservan en [CORRIDOR_HISTORY.md](../analysis/CORRIDOR_HISTORY.md).

## V2 actual: copiar el verde PPO en los tres receptores

En cada inicio recto de J0, se comunica la duración **ya decidida** por PPO a J2/J10/J16. Cada receptor intenta abrir en `inicio_J0 + tiempo_acumulado` y ejecuta exactamente esa duración en su verde corredor; no añade el amarillo a la copia. Tiempos iniciales por tramo **11/10/10 s**, parametrizables: objetivos acumulados **11/21/31 s**. Son aproximaciones iniciales, no una calibración nueva del corredor completo.

La preparación ajusta sólo verdes no-corredor pendientes, dentro de 5–45 s, para acercarse al inicio solicitado. Se mantienen amarillos de 4 s y orden cíclico. Si el inicio exacto no es alcanzable, se registra la apertura real; una fase ya iniciada sólo se aprovecha si aún puede terminar con la duración copiada exacta. Copiar duración no garantiza cubrir toda la ventana desplazada. No hay preaviso desde la fase anterior ni correcciones por cruces reales en esta V2.

Ejecutor activo: `python run_sync.py --mode v2 --seed 42 --gui`. También permite `--mode v1` y `--mode v3`, siempre con alcance completo. Las tres devuelven `waiting_time`, `effective_flow` y `avg_queue_length`, usando `get_lane_metrics` sobre los carriles controlados sin duplicados, igual que `plain.py`. El waiting global de toda la red se registra aparte. Los resúmenes se guardan en `results/`; `--traces` habilita los CSV detallados.

En la validación inicial con demanda 42, se verificaron **170/170 copias completas exactas** entre los tres receptores; una copia final de J16 quedó censurada por el horizonte. J2 logró **34/42 (80,95%)** sin parada. El waiting global fue **44223 veh·s**, frente a 45560 de V1 local, aunque waiting E1 y secundario J2 subieron. Fue una validación de una sola demanda; la comparación completa de cinco demandas aparece abajo.

## V3 histórica: preaviso aplicado sólo a J2

Al decidir PPO el verde anterior al recto (fase XML 0), V3 anuncia `inicio_anterior + duración_decidida + 4 s` y prepara J2 para ese inicio `+12 s`. No anticipa la inferencia del próximo verde ni conoce aún su duración. El primer cruce real corrige el frente y los posteriores amplían la cola del mismo evento; se mantienen los límites y J10/J16 fijos.

En demandas 42–46, **220/232 (94,83%)** pasaron J2 sin parada y las **202 ventanas objetivo derivadas de cruces reales** quedaron cubiertas. El preaviso resolvió la factibilidad observada, pero frente a V1 el waiting secundario subió **62,55%** y el total de red **14,10%**. Hubo **81 preparaciones vacías**. Ese ensayo local validó el mecanismo y mostró un coste secundario elevado; no medía una coordinación completa.

## Extensión actual al corredor completo

V1 aplica el mismo aviso desde el inicio recto de J0 a J2/J10/J16, con objetivos acumulados **11/21/31 s**. Acorta únicamente verdes no-corredor pendientes; el verde receptor permanece en 15 s. V3 aplica su preaviso a los tres, con objetivos **12/22/32 s**; el primer cruce real en J0 corrige el frente y los posteriores amplían causalmente la cola. No reserva la duración PPO ni copia el verde como V2.

Los tres receptores siguen tomando J0 como referencia; aún no se corrige J10/J16 con los retrasos reales observados en los tramos intermedios. Los 10 s de cada tramo posterior son aproximaciones iniciales. Las corridas actuales usan demandas 42–46, 3600 s y SUMO seed 42, con resultados separados del historial. [Comparación completa y validaciones](../analysis/corridor_full/REPORT.md).

De **93 vehículos que atraviesan los tres receptores**, V1/V2/V3 logran **51/61/84 sin parada (54,84/65,59/90,32%)**. Entre los 27 que continúan recto por E13: **17/18/27**. Waiting global medio: **44072/44480/49560 veh·s**; tiempo medio J0→J16 entre demandas: **37,96/42,54/36,09 s**. V3 da la mejor progresión agregada con mayor waiting global; V1 tiene el menor waiting medio y V2 un compromiso intermedio. Cero violaciones de duración/orden y cero teleports; todos los vehículos de la cohorte completa terminaron. Se conservan la tabla por demanda, agregados, referencia congelada y validación en `analysis/corridor_full/`.

**Pregunta abierta:** ¿qué compromiso entre progresión completa y waiting secundario ofrece cada versión, y cuánto ayudaría calibrar los tramos posteriores o corregir los retrasos reales intermedios?
