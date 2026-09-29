# Contexto para rediseño desde primeros principios

## A. Qué tenemos funcionando

El workspace previsto es C:\Users\luuis\GitHub\SUMO_Traffic_Lights_Algorithm en master-slave-clean-v2. Esta rama parte de 39e862f y conserva el PPO v39 single-agent de policy/train.py + policy/agent.py, junto con la red SUMO master/slave de J0/J2/J10/J16. El PPO usa una acción continua escalar para duración de verde y fase cíclica fija. La red corregida tiene J2 con ciclo 76 s. El código PPO está diseñado para un TLS, pero toma el primer ID devuelto por TraCI. J0 aparece primero en las declaraciones tlLogic del XML, lo cual no garantiza el orden de getIDList(); Python/TraCI no pudo ejecutarse en este equipo, así que la correspondencia efectiva sigue **no verificada**. No se cambió la lógica.

## B. Qué queremos conseguir

Favorecer progresión temporal real desde la liberación de J0 por J2, J10 y J16, manteniendo J0 como master PPO y tratando las otras señales como slaves. La progresión debe evaluarse junto con waiting, throughput, estabilidad temporal y efectos por origen/movimiento.

## C. Qué se intentó

Infraestructura multi-TLS, tracking de vehículos/platoones, ETA libre Phase B, C0 shadow, C1 con adelantos limitados, predictor encadenado offline, auditorías de autoridad, filtro downstream/admission-aware, descomposición por origen/movimiento y pruebas multi-demanda. La rama histórica contiene código, resultados y detalle cronológico.

La evaluación pareada D0–D3 de C1/FILTERED cerró como **CASO C — sensible a la demanda**: Δ waiting total/cohorte fue D0 −376 s/−6.95 s por vehículo, D1 −30,899 s/−5.62 s (con teleports y 52 inserciones menos en FILTERED), D2 seed101 +873 s/+7.97 s y D3 seed202 +127 s/−10.00 s. Los alimentadores laterales E empeoraron waiting por vehículo en 4/4 pares. La línea queda cerrada; no continuar su refinamiento aquí.

## D. Qué aprendimos

El amarillo accidental de J2=24 s distorsionaba la red; esta rama lo corrige a 4 s. C1 y FILTERED tuvieron mejoras parciales en métricas/demandas específicas, pero no validaron progresión robusta. Forecasts con MAE bajo aún pueden tener errores de cola/ventana y reanclajes de decenas de segundos. La autoridad local de 5 s es pequeña frente a parte de la necesidad geométrica; los cambios pueden desplazar espera o desalinear el siguiente TLS. Mejorar una métrica o intersección local no equivale a sincronizar el corredor.

## E. Caminos que no repetir sin nueva justificación

- Acumular ajustes locales pequeños independientes como si no alteraran calendarios/offsets downstream.
- Convertir ETA de llegada libre, acceso a edge o presencia de vehículos delante en hora de cruce/servicio sin modelar cola y señal.
- Tratar un pronóstico shadow o una envolvente geométrica como efecto causal.
- Juzgar FILTERED por D0 únicamente o por waiting agregado sin analizar origen/movimiento y varias demandas.
- Usar resultados de J2=96 s para diseñar la red corregida de 76 s.
- Optimizar reward como sustituto de métricas deterministas de tráfico.

No son prohibiciones permanentes; cualquier reutilización necesita hipótesis nueva y validación apropiada.

## F. Qué queda deliberadamente fuera del código

C0/C1, FILTERED/admission-aware, predictores, reservas, cooldowns, autoridad temporal, reanchoring experimental y sus scripts/logs no forman parte de este árbol. Solo se conservan como conocimiento en EXPERIMENT_HISTORY.md. No se implementa C2 ni un nuevo controlador slave. Los artefactos y reportes siguen en la rama/workspace histórico.

Los resultados de demanda solo sobreviven aquí como síntesis documental; los archivos experimentales sin seguimiento están resguardados en el stash histórico local.

## G. Pregunta arquitectónica abierta

> Partiendo del PPO original estable en J0 y una red de cuatro TLS, ¿cuál es la arquitectura master/slave adecuada para producir verdadera progresión J0→J2→J10→J16 sin depender de pequeñas correcciones locales independientes que acumulen deriva o desplacen el problema downstream?

Este documento plantea la pregunta; no la responde.
