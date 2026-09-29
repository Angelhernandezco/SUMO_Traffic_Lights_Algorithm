# SUMO Traffic Lights Algorithm

Proyecto de control semafórico con Python, SUMO, TraCI y PPO.

El objetivo general es entrenar un PPO para decidir la duración del verde actual y estudiar progresión entre intersecciones. Esta rama conserva el PPO v39 original para un solo TLS y la red master/slave de cuatro TLS. J0 es el TLS que el proyecto reserva para el PPO; J2, J10 y J16 son infraestructura slave estática, sin coordinador nuevo.

La configuración de red es maps/master_slave.net.xml, con rutas maps/master_slave.rou.xml, cargadas por configuration.sumocfg. En esta rama, J2 queda con cuatro verdes de 15 s y cuatro amarillos de 4 s: ciclo de 76 s. J10 conserva 41/4/41/4 (90 s), y J0/J16 conservan 15/4 repetido cuatro veces (76 s).

## Ejecutar el PPO original

Con Python, dependencias del proyecto y SUMO disponibles:

- Entrenar: python main.py --policy-train -e 32 -s 2000 -m master_slave_clean
- Evaluar con política determinista y GUI: python main.py --policy-test -s 2000 -m master_slave_clean

La interfaz CLI pasa min_green=5 y max_green=45 en ambos modos. Los archivos de modelo se guardan bajo policy/models/; no hay checkpoint versionado en esta base. Consulta las notas sobre selección de TLS en PPO_BASELINE.

## Estructura

- main.py: opciones de ejecución.
- policy/train.py, policy/agent.py: entorno, recompensa, entrenamiento y agente PPO.
- sumo_utils.py: extracción de fases y utilidades TraCI.
- configuration.sumocfg, maps/: SUMO, red y rutas.
- docs/: baseline técnico, objetivo, historial y contexto de reinicio.

## Estado

**Baseline limpio para rediseño master/slave.** No contiene coordinación C0/C1 ni predictor experimental. El siguiente trabajo puede diseñar la arquitectura desde primeros principios, usando el historial como conocimiento.

El workspace de esta rama es la carpeta habitual del repositorio: C:\Users\luuis\GitHub\SUMO_Traffic_Lights_Algorithm, en la rama master-slave-clean-v2.

- [PPO_BASELINE.md](docs/PPO_BASELINE.md)
- [MASTER_SLAVE_GOAL.md](docs/MASTER_SLAVE_GOAL.md)
- [EXPERIMENT_HISTORY.md](docs/EXPERIMENT_HISTORY.md)
- [RESTART_CONTEXT.md](docs/RESTART_CONTEXT.md)
