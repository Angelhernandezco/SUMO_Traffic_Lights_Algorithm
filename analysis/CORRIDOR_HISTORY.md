# Antecedentes útiles del corredor

Estos resultados explican las decisiones actuales. Son experimentos históricos;
la comparación vigente de V1/V2/V3 completos está en
[corridor_full/REPORT.md](corridor_full/REPORT.md).
Los reportes originales permanecen en Git, en el commit `edba86a`.
Las métricas actuales y las referencias para comprobar su reproducción se
conservan en `corridor_full/`.

## Onda fija

Se probaron 92 combinaciones de offsets y se confirmaron las seleccionadas a
3600 s, demanda original y SUMO seed 42. En una copia temporal se aseguraron
cuatro programas de 76 s. La red original de aquella etapa todavía podía cargar
90 s en J10; la red actual ya contiene únicamente programas `0` de 76 s.

| Configuración fija | Offsets J2/J10/J16 | Rectos completos sin parada |
|---|---|---:|
| Sin desfase | 0/0/0 | 0/19 |
| Mejor progresión entre las probadas | 8/74/26 | 18/19 |
| Referencia equilibrada | 0/72/24 | 16/19 |

En el conjunto amplio de rutas del corredor, la mejor configuración redujo
el tiempo medio de 103,94 a 54,31 s y las paradas por vehículo de 4,22 a 2,38,
pero aumentó el waiting de las demás rutas un 21,4%. La equilibrada redujo el
tiempo a 56,93 s, con un aumento secundario del 3,0%. Ese waiting se medía por
rutas en aproximaciones controladas; no equivale al waiting global actual.

## PPO con receptores fijos

Con los offsets equilibrados, una prueba histórica del checkpoint `test1`
redujo la progresión recta completa de 16/19 a 1/19. El desplazamiento absoluto
de los inicios PPO respecto al ciclo fijo tuvo mediana 23 s y P90 35 s.
Las liberaciones aparecieron cada 57,74 s en promedio, frente a ciclos fijos de
76 s. Un desplazamiento estático no resuelve esa diferencia de cadencia.
Estas cifras no son una evaluación del checkpoint oficial `test36_3`.

## Calibración y llegadas J0 a J2

En la V1 histórica, que sólo coordinaba J2, T=11 s mejoró la demanda 42:
31/42 sin parada frente a 24/42 con T=9 s. En cinco demandas, el mejor T
dependió de la semilla: 11 s en 42, 9 s en 43/44/46 y 12 s en 45.
Por tanto, 11 s quedó como referencia de comparación, sin demostrar un óptimo
universal.

La auditoría de 42 vehículos con T=9 s clasificó sus 18 detenciones en:
9 ante rojo sin líder lento cercano, 6 en cola con rojo y 3 en cola con verde.
No se observaron primeras detenciones causadas por el final del verde.
La señal verde al llegar a la línea no descarta una parada previa en la cola.

La auditoría de pelotones reutilizó 20 corridas V1 (cinco demandas y cuatro T).
El viaje cruce real de J0 a proximidad de J2 tuvo mediana 12 s en trayectorias
sin parada, con P10–P90 de 8–16 s. El retraso del primer cruce de J0 explicaba
mejor el frente de llegada que la duración PPO. Hubo liberaciones vacías y
tamaños variables; no se debe fijar un número de vehículos por pelotón.

## Ventanas históricas y preaviso

Todas las filas siguientes coordinaban únicamente J2, con J10/J16 fijos,
checkpoint oficial, demandas 42–46 y SUMO seed 42.

| Mecanismo histórico | Sin parada en J2 |
|---|---:|
| V1: inicio de verde J0 +11 s | 154/232 (66,38%) |
| Antigua V2 causal: primer cruce +12 s | 114/232 (49,14%) |
| V3 local: preaviso y corrección causal | 220/232 (94,83%) |

La antigua V2 causal conocía demasiado tarde el frente: 153/205 eran
inalcanzables al recibir el primer cruce. El preaviso de V3 resolvió la
factibilidad observada, pero aumentó el waiting secundario de J2 un 62,55%
y el global un 14,10% frente a V1 local. Hubo 81 preparaciones vacías.
La V2 vigente es **copia de duración PPO en J2/J10/J16**; no es esta variante
causal retirada ni la primera prueba de ventana PPO con T=9 s.

La comparación inicial mezclaba V1/V3 locales con V2 completo. Sus conclusiones
no sustituyen la evaluación actual: ahora las tres coordinan J2/J10/J16 y se
comparan con iguales demandas, cohortes y definiciones de tiempo.
