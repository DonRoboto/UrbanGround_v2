# Checklist para nuevas ejecuciones (Parte B)

Antes de ejecutar SigLIP 2:

- [ ] Verificar que `data/images/fachada_01.jpg` a `fachada_13.jpg` sean las originales de **640×640**.
- [ ] Revisar visualmente `runs/roi_review.png` y justificar qué frente/planta baja se recorta.
- [ ] Confirmar y versionar las etiquetas A/I/N antes de ejecutar las variantes experimentales.
- [ ] Mantener `F11` restringida al frente central; documentar `F02/F13` como vistas del mismo negocio.
- [ ] Ejecutar tests sin GPU y `urban-vlm run --limit 2` para comprobar dependencias.

Durante la ejecución:

- [ ] E0: 13 × 2 × 2 = **52 predicciones**, **780 logits**.
- [ ] E1: ablación offline contra los mismos 780 logits, sin cambiar el checkpoint.
- [ ] E2: 13 × 2 × 5 = **130 decisiones**, **1,950 logits** (cinco intervenciones espaciales).
- [ ] E3: dos etapas, 13 × 2 = **26 decisiones**, **520 logits** si `--region both`.
- [ ] Guardar nuevos resultados bajo un directorio `runs/` único, nunca sobrescribir E0.
- [ ] Guardar hashes, checkpoint/revisión, versiones de Python, Transformers, Torch y hardware.

Después:

- [ ] Revisar las **13 filas por banco** del reporte y explicar todos los desacuerdos.
- [ ] Describir el efecto del *recorte* como combinación de cambio de contexto y escala.
- [ ] Describir Top-2 y clasificación jerárquica como exploración **post hoc**, no pruebas independientes.
- [ ] No interpretar el score `I` como evidencia de cierre permanente.
- [ ] No afirmar que se demostró sesgo regional; el conjunto no permite ese contraste.
- [ ] Para GitHub, confirmar licencias de las fotografías; evitar incluir imagen personal/identificable sin permiso.
