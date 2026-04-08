# README.md

# SUMO Traffic Lights Algorithm — Estado actual del proyecto

## Resumen

Este proyecto busca entrenar un **controlador semafórico con PPO** para un cruce en SUMO bajo un enfoque deliberadamente simple:

- **single agent**
- **single scalar action**
- **orden fijo de fases** mediante `phase_cursor` cíclico
- la policy **no elige la siguiente fase**
- la policy **solo ajusta la duración del verde de la fase actual**
- **test determinista**
- **mismos valores de `min_green` y `max_green` en train y test**

La meta no es cambiar la secuencia de fases ni meter reglas externas, sino lograr que la policy aprenda por sí sola a:

- dar **más verde a la fase dominante**
- dejar las fases débiles **cerca del mínimo**
- y **adaptarse** cuando el tráfico se vuelve más balanceado

---

## Estado actual

### Versión actual
**v36**  
Modelo de referencia actual: `model_future_v36`

### Mejor resultado observado hasta ahora
- **Best deterministic waiting:** `6438`
- obtenido en training y reproducido en test determinista
- comando usado:
  - train: `python main.py --policy-train -m model_future_v36 -e 15 -s 2000 --min-green 5 --max-green 45`
  - test: `python main.py --policy-test -m model_future_v36 -s 2000 --min-green 5 --max-green 45`

### Lectura general del comportamiento de v36

La v36 **sí mejoró** respecto a versiones claramente colapsadas, porque ya muestra algo de estructura útil:

- las fases débiles tienden a quedarse en valores bajos
- la fase dominante recibe más tiempo que las demás
- el controlador ya distingue parcialmente entre presión baja y presión alta

Pero todavía **no llega al comportamiento deseado**.

Lo que sigue faltando es:

- que las fases débiles se acerquen más a **`min_green` real**
- que la fase dominante se acerque más a **`max_green` real**
- que el comportamiento sea más claramente **dependiente del estado**
- que no se quede en una política de compromiso tipo:
  - fases débiles: `7–8 s`
  - fase dominante: `17–19 s`

Es decir, la v36 ya encontró la **dirección correcta**, pero todavía con **amplitud insuficiente**.

---

## Diagnóstico actual

## Qué está pasando realmente

El archivo de tráfico tiene una estructura no uniforme:

1. **inicio:** domina fuertemente una aproximación
2. **transición:** empiezan a entrar otras direcciones
3. **más adelante:** el flujo se vuelve más mixto

Eso significa que el comportamiento óptimo no es dar tiempos parecidos a todas las fases.

Lo esperable sería algo como:

- al inicio:
  - fases débiles cerca de `5`
  - fase dominante cerca de `45`
- más adelante:
  - aumentar otras fases cuando acumulen presión real
- al final:
  - reparto más balanceado

## Problema actual del policy

Aunque la v36 ya reacciona algo al estado, todavía cae en este patrón:

- “pasar relativamente rápido” por las fases vacías
- “dar algo más de tiempo” a la dominante
- pero **sin explotar los extremos**

Eso produce mejoras, pero no el máximo ahorro posible.

## Conclusión del diagnóstico

El sistema **no está roto**, pero sigue atrapado en un **óptimo local intermedio**:

- ya aprendió a no tratar todas las fases igual
- pero todavía no aprendió a usar el rango completo `min_green` ↔ `max_green`

En particular:

- el lado bajo del rango está **casi** resuelto
- el lado alto del rango todavía no está suficientemente aprendido

---

## Aprendizajes acumulados de versiones anteriores

## Base sólida que se conserva
La base que sigue teniendo más sentido es la de la familia **v3**, porque:

- mantenía la filosofía PPO pura
- no dependía de heurísticas duras
- ya mostraba señales de que el agente sí podía aprender algo útil del patrón del tráfico

## Qué aprendimos y no queremos repetir

### De v3.5
Se vio que premiar demasiado el **ciclado rápido** puede inducir una solución degenerada:

- pasar casi todo en mínimo
- mejorar waiting solo por “dar vueltas rápido”
- pero **sin servir correctamente la fase dominante**

Eso no es lo que queremos.

### De v4 y variantes más agresivas
Cuando se endureció demasiado la reward o se empujó demasiado una idea concreta, el sistema tendió a:

- desestabilizarse
- degradar el test determinista
- o romper el balance entre “fast pass” y “servir bien”

### Lección importante
El `fast_pass` **sí debe existir**, pero con esta jerarquía:

1. **servir bien la fase dominante** cuando realmente domina
2. **fast pass inteligente** cuando la fase actual no merece tiempo
3. castigar extensiones innecesarias

O sea:

- pasar rápido es bueno
- pasar rápido cuando otra fase domina es mejor
- pero **servir correctamente una fase realmente dominante es todavía más importante**

---

## Filosofía vigente del proyecto

Se mantiene sin cambios:

- PPO puro
- single agent
- single scalar action
- orden fijo de fases
- sin early cutoff
- sin recortes heurísticos del action
- sin forzar reglas del tipo:
  - “si pasa X, entonces duración = Y”

Sí se aceptan:

- mejores features de estado
- mejor reward shaping
- mejoras de entrenamiento
- mejor evaluación
- test determinista

---

## Estado del controlador actual

## Lo que ya logra
- distinguir parcialmente la fase dominante
- reducir bastante el waiting frente a versiones claramente planas
- reproducir en test el mejor checkpoint guardado
- mantener comportamiento consistente entre train y test

## Lo que todavía no logra
- no se va lo suficiente a los extremos
- no da a la fase dominante un verde suficientemente largo al inicio
- no deja a las otras fases suficientemente pegadas al mínimo
- cuando aparece una buena política, no siempre queda como una regla robusta y estable

---

## Objetivo inmediato

El objetivo de la siguiente iteración no es cambiar de paradigma.

Es **seguir sobre la base v36 / familia v3**, pero afinarla para que el controlador aprenda mejor esta regla implícita:

- si una fase domina claramente, **estírala mucho**
- si una fase no tiene presión, **déjala casi en mínimo**
- si todavía no es su momento, **pásala rápido**
- cuando el tráfico se balancee, **balancea también el verde**

---

## Cambios a futuro cercano (próxima iteración)

Estos son los cambios planeados para continuar mañana.

## 1. Mejorar el estado con demanda relativa más clara

La prioridad principal es que el estado deje más explícito cuándo la fase actual es dominante y cuándo no.

### Features a reforzar o agregar
- demanda/presión de la fase actual
- suma de demanda de las otras fases
- máximo de demanda entre las otras fases
- razón entre fase actual y resto
- share de la fase actual respecto al total
- posición relativa de la fase actual:
  - dominante
  - intermedia
  - débil
- gap entre:
  - presión actual
  - mejor fase alternativa

### Intención
Que la red no solo vea “hay vehículos”, sino que vea claramente:

- “esta fase manda”
- “esta fase no importa todavía”
- “todavía conviene pasar rápido para llegar a otra fase más fuerte”

---

## 2. Ajustar la reward para empujar más amplitud sin volverla rígida

La reward no debe forzar manualmente mínimos y máximos, pero sí debe alinear mejor el incentivo.

### Dirección de ajuste
- mantener el objetivo principal:
  - **reducir waiting global**
- mantener un término de:
  - **servir bien una fase dominante**
- mantener un término de:
  - **fast pass**
- pero cambiar su balance para que:
  - el `fast_pass` no gane por sí solo
  - la dominante bien servida tenga más valor
  - las extensiones sin utilidad sigan penalizadas

### Regla conceptual deseada
- fase dominante real:
  - premiar más cuando se le da tiempo suficiente
- fase débil:
  - premiar pasar rápido
- fase intermedia:
  - dejar que el PPO decida de forma continua

### Qué NO se quiere hacer
- no meter hard rules
- no truncar la acción por fuera
- no imponer manualmente “si share > X entonces 45”

---

## 3. Empujar mejor el uso del rango completo de acción

Hoy el policy ya baja bastante, pero no sube lo suficiente.

Entonces la siguiente iteración debe ayudar a que el agente descubra más fácilmente que:

- `7` no siempre basta para una fase vacía, si `5` sería mejor
- `18–19` no siempre basta para una dominante, si `35–45` sería mejor

La idea no es forzar eso desde fuera, sino hacer que:

- el estado lo vuelva evidente
- la reward lo vuelva rentable

---

## 4. Mantener test determinista como criterio final

Se seguirá usando:

- train con exploración PPO
- test con política determinista

Porque el criterio real del proyecto es:

- que el comportamiento útil aparezca en **test determinista**
- no solo como episodios buenos aislados durante train

---

## Hipótesis de trabajo para la siguiente iteración

La hipótesis actual del proyecto es esta:

> El agente ya aprendió parcialmente a distinguir fases fuertes y débiles, pero todavía no tiene una representación suficientemente clara de la dominancia relativa ni un incentivo suficientemente bien alineado para explotar el rango completo de duración.

Si esta hipótesis es correcta, entonces la siguiente mejora debería producir:

- menos acciones medias constantes
- más diferencias claras entre fases
- verdes muy cortos en fases débiles
- verdes mucho más largos en la dominante
- y adaptación gradual cuando el flujo se equilibre

---

## Criterio de éxito de la siguiente versión

La siguiente versión irá en la dirección correcta si en **test determinista** se observa:

- la acción deja de verse casi constante
- la duración depende más claramente del estado
- al inicio:
  - fase dominante claramente más larga
  - demás fases cerca del mínimo
- después:
  - las fases secundarias suben solo cuando su presión lo justifica
- más adelante:
  - reparto más balanceado cuando el tráfico realmente se balancea

Idealmente, además, se buscará mejorar el mejor waiting actual de referencia:

- **objetivo a superar:** `6438`

---

## Restricciones que siguen vigentes

No se quiere cambiar:

- PPO
- una sola acción
- orden fijo de fases
- test determinista
- sin early cutoff
- sin reglas duras externas

No se quiere que el sistema gane únicamente por:

- ciclar rápido a ciegas
- colapsar a mínimo para casi todo
- producir una política casi constante

---

## Resumen ejecutivo

### Punto actual
La **v36** es una versión funcional y útil, pero todavía conservadora.

### Mejor hallazgo
Ya mostró que el agente puede aprender una estructura razonable y bajar el waiting hasta **6438**.

### Principal limitación
Todavía no llega a los extremos `min_green` / `max_green` cuando la situación del tráfico lo justificaría.

### Próximo paso
La siguiente iteración se enfocará en:

- **estado más expresivo en demanda relativa**
- **reward mejor alineada con dominancia real**
- **mantener PPO puro y acción escalar única**
- **sin heurísticas duras**
- **buscando que el agente use mejor todo el rango de duración**

---

## Comandos base de trabajo

### Entrenamiento
```bash
python main.py --policy-train -m model_future_v36 -e 15 -s 2000 --min-green 5 --max-green 45