# Protocolo de anotación visual (A / I / N)

**Unidad:** el frente objetivo en la fotografía, no todos los comercios visibles en la escena.

- **A — Actividad comercial visible:** acceso abierto dispuesto para ventas/servicios, productos exhibidos,
  puesto instalado, mesas preparadas; no exige observar una compra real.
- **I — Indeterminada:** hay indicios de establecimiento, pero su operación actual no es verificable;
  cortina cerrada, lona que impide ver, atención no visible. No significa negocio inactivo.
- **N — Sin comercio visible:** el frente objetivo no tiene señales visuales de establecimiento comercial.
  No demuestra que no exista negocio dentro ni fuera de la foto.

**Uso mixto:** planta baja comercial visible + pisos superiores residenciales = A.
**Múltiples frentes:** se anota únicamente el inmueble seleccionado. F11 usa edificio central y
excluye comercios parciales de los extremos.
**Frentes repetidos:** F02 y F13 podrían ser el mismo negocio: conservar ambas imágenes y reportar dependencia.
**Evidencia:** registrar un hecho observable y no inferir actividad por el nombre, el tipo o la zona.

**Procedimiento recomendado para nuevas pruebas:**
1. Delimitar frente objetivo antes de ver puntuaciones; anotar A/I/N por humanos sin acceder a predicciones.
2. Conservar justificación y ROI en CSV. Una segunda persona debería revisar dudas cuando sea posible.
3. Congelar etiquetas y prompts antes de ejecutar la evaluación independiente.
4. Documentar ambigüedades y discrepancias sin inventar horarios ni estados de operación.

`data/annotations.csv` contiene etiquetas del análisis anterior confirmado de manera iterativa, no un
nuevo experimento de anotación ciega. Revisa las etiquetas antes de entregar los resultados.
