# Plan para repetir las pruebas antes de publicar

**Antes de inferencia:** comprobar los 13 archivos 640×640; revisar las 13 ROI en `runs/roi_review.png`;
reconfirmar etiquetas 05,07,09 y el edificio central 11.

**Ensayo de humo:** `--limit 2`, validar 8 decisiones y 120 logits; no comparar exactitud con n=13.

**Ensayo completo:** 52 decisiones, 780 logits, cuatro filas de métricas y anotaciones adjuntas.
Comparar las predicciones con `reference/prior_exploratory_predictions.csv`, sin exigir igualdad
bit a bit si cambió versión de paquete/checkpoint.

**Diagnóstico de errores:** 05 (I→A), 07 (I→N), 09 (I→A) bajo roi/generic según análisis previo.
Ejecutar perturbaciones sobre estos casos y guardar margen antes/después.

**Ablación textual:** no cambiar prompts para elegir el mejor sobre las mismas 13 imágenes y luego
presentar ese resultado como test independiente. Si cambias prompts, marca el resultado *exploratorio*.

**Publicación:** ejecutar `pytest -q`; incluir manifest, requisitos, comando y protocolo de anotación;
no subir las 13 capturas a GitHub sin permiso o verificación de sus términos.
