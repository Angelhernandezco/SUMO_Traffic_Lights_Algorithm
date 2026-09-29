# PPO original: baseline v39 con transición amarilla

Este documento describe el código en el commit base 39e862f, salvo la corrección de duración de J2 documentada en la guía de red. No atribuye al baseline componentes que aparecieron después.

## Implementación y alcance real

Los archivos centrales son main.py, policy/train.py, policy/agent.py, sumo_utils.py, configuration.sumocfg y la red/rutas bajo maps/. main.py --policy-train y --policy-test llaman a run_policy. Este inicia configuration.sumocfg y opera sobre junctions[0] según el orden devuelto por TraCI, toma hasta cuatro fases con verde y requiere exactamente cuatro.

La intención del proyecto es que PPO controle J0. En el XML, la primera declaración tlLogic es de J0, pero eso no establece el orden de respuesta de TraCI. El código candidato no valida que junctions[0] == J0. No se pudo consultar el orden en runtime: Python no está disponible en PATH y el launcher de .venv apunta a una instalación Python 3.11 que no existe en este equipo. Por tanto, que junctions[0] corresponda efectivamente a J0 queda **no verificado**; no se cambió esa lógica. Los TLS slaves no tienen agentes PPO ni reglas de coordinación en este commit. SUMO deja correr sus programas estáticos cuando no se intervienen.

## Observación

La observación se calcula con conteo de vehículos y vehículos detenidos por carril, agregados por fase verde. Para cada una de cuatro fases usa presión = cola detenida + 0.35 × vehículos. El vector incluye, en orden cíclico desde la fase actual, cuatro valores por fase: log(1+vehículos), log(1+cola), log(1+presión) y participación de presión. Añade 16 resúmenes relativos: presión total y futura ponderada, máximo/suma futuros, brechas actual-vs-futuras, posición de pico futuro, presión de otros, razones de dominancia, brecha vs mejor otro, proporción top2/top1 y rango actual. Dimensión total: 4 × 4 + 16 = 32 valores.

La normalización de observaciones usa RunningMeanStd y recorta valores normalizados a ±5. En entrenamiento actualiza durante los primeros cuatro episodios; después queda congelada. Evaluación usa las estadísticas guardadas y no las actualiza.

## Acción y fases

El actor emite una acción continua escalar en [0,1], modelada con una distribución Beta. En entrenamiento muestrea de esa distribución. El mapeo primero recorta a [0,1], luego redondea min_green + acción × (max_green−min_green) al segundo entero más cercano. La CLI proporciona min_green=5 y max_green=45 tanto en entrenamiento como en prueba. El default interno de run_policy es max_green=60; por eso los comandos documentados pasan por main.py y la evaluación oficial debe indicar explícitamente 45. La función no rechaza rangos fuera de 5–45 si se llama directamente.

La política elige cuánto dura el verde actual. phase_cursor avanza (actual + 1) módulo número de fases: PPO no selecciona la fase siguiente ni cambia el orden. El entorno extrae hasta las cuatro fases verdes, ignorando fases de transición/all-red para construir la lista de fases del agente.

## Verde, amarillo y horizonte

Cada acción fija la fase verde del programa y avanza SUMO segundo a segundo por la duración solicitada, limitada por el tiempo restante del episodio. Captura after_green justo al terminar ese verde. Si hay verde ejecutado, construye estado amarillo reemplazando G/g por y y ejecuta hasta 4 segundos, también limitado por el horizonte.

El código mantiene separados green_executed_duration, yellow_executed_duration, green_waiting_sum y yellow_waiting_sum. La recompensa usa promedios/estados del tramo verde: la espera durante amarillo no se atribuye a la decisión de verde. El total waiting del episodio acumula la espera observada durante verde y amarillo. La siguiente observación se obtiene tras completar amarillo, por lo que corresponde al estado real posterior a la transición. Cerca del final, el amarillo puede truncarse para respetar el horizonte.

## Recompensa

La señal es una recompensa moldeada por transición, no la métrica final. El costo incluye la espera detenida media solo durante el verde, 0.25 × presión total restante después del verde y una penalización por crecimiento de presión. Se combina con alivio de presión de la fase actual y global, contexto de paso rápido, extensión de fase dominante, extensiones malas, demora, extensión en tráfico mixto, verde desperdiciado en fase débil/sin cola e infraverde. Los coeficientes están definidos en SumoTrafficEnv.step en policy/train.py.

La selección del checkpoint se hace por menor espera total en evaluación determinista, no por reward. La evaluación es una corrida completa según el horizonte configurado. Reward se reporta como diagnóstico.

## PPO y entrenamiento

policy/agent.py implementa actor-crítico feed-forward con backbone de dos capas ocultas de 256 unidades, cabezas alpha/beta para la Beta de acción y una cabeza de valor. PPO calcula retornos/ventajas con gamma=0.99 y GAE lambda=0.95; normaliza ventajas y usa objetivo clipped, pérdida MSE del valor, entropía, Adam y recorte del gradiente.

Los valores concretos que run_policy pasa al agente son: learning rate 2e-4, clip epsilon 0.15, entropía inicial 0.07, cuatro épocas PPO por update, minibatch 64, max grad norm 0.35, normalización de observación activada, concentración mínima Beta 0.2 e inicialización de media de acción 0.095 / concentración total 3.0. Entropía decrece a 0.015 durante el 70% inicial del entrenamiento. Se reúnen dos episodios estocásticos por update (el último grupo puede ser parcial). Después de cada update corre una evaluación determinista; la mejor espera total guarda el modelo y sus metadatos.

El protocolo práctico actual indicado por el contexto de proyecto es 32 episodios de 2000 s. El CLI por defecto ofrece 50 episodios, así que para reproducir el protocolo práctico se deben pasar -e 32 -s 2000.

## Evaluación y checkpoint

--policy-test carga policy/models/<nombre>.pth, valida contra los límites min/max almacenados cuando existen y llama al actor con deterministic=True, que utiliza la media alpha/(alpha+beta), no una muestra. El comando documentado abre sumo-gui. Evaluación en entrenamiento también usa la media y SUMO sin GUI.

El código base no contiene pesos .pth versionados y no permite identificar desde Git un checkpoint histórico específico ni su resultado asociado: **no verificado**. Los resultados v39 descritos en archivos de contexto posteriores son contexto histórico, no evidencia de que ese artefacto esté dentro de este commit.

## Limitaciones y verificaciones pendientes

- Selección del primer ID TLS de TraCI; el archivo de red declara J0 primero, pero no hay evidencia local de que el orden runtime coincida ni guardia que confirme J0.
- run_policy tiene max_green interno 60, aunque el CLI usa 45; scripts que llamen directamente deben pasar 45.
- Los valores/rendimientos específicos de un checkpoint local no se pueden recuperar desde este commit.
- Esta descripción es sobre la implementación single-TLS del PPO. No implica que el actor PPO controle la coordinación J0→J2→J10→J16.
