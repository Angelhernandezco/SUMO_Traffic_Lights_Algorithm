# Objetivo master/slave

## Objetivo conceptual

J0 sigue siendo el único agente PPO y el master. Cuando J0 libera tráfico hacia el corredor J0 → J2 → J10 → J16, buscamos que las intersecciones slave utilicen esa información para favorecer progresión temporal coordinada.

La imagen conceptual es:

J0 libera → ventana objetivo J2 → ventana objetivo J10 → ventana objetivo J16

Es una descripción del objetivo, no una arquitectura decidida. Aún no se elige entre offsets, green wave, planificación por pelotón, calendario master, solución híbrida u otra alternativa. El próximo rediseño debe comparar opciones desde primeros principios.

## Restricciones conceptuales

- J0 conserva un único PPO salvo decisión futura explícita.
- No se busca multi-agent PPO por defecto.
- J2, J10 y J16 son slaves deterministas/reactivas al master, salvo nueva justificación arquitectónica.
- El PPO conserva su acción escalar de duración verde y su orden fijo de fases.
- La evaluación debe juzgar progresión de corredor y efectos en tráfico lateral/directo, además de waiting determinista.

## Estado de esta rama

Se conserva la red SUMO master/slave y PPO v39. Los programas estáticos de J2/J10/J16 son infraestructura de base; no se incorpora un coordinador nuevo. J2 tiene cuatro pares de verde 15 s/amarillo 4 s, ciclo 76 s. J10 permanece 41/4/41/4, ciclo 90 s; J16 15/4 repetido, ciclo 76 s. La corrección de J2 no cambia el estado de fase, el orden, las rutas ni las otras intersecciones.

No se implementan C0, C1, FILTERED, predictor, reservas, cooldowns, reanchoring, admission-aware ni lógica de autoridad temporal. Este documento no prescribe cómo coordinar.
