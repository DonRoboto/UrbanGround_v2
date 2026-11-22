# Checklist de entrega y publicación (Parte B)

- [ ] Revisar **personalmente** las 13 etiquetas A/I/N y evidencia; documentar decisiones ambiguas.
- [ ] Documentar que F02 y F13 probablemente no son negocios independientes.
- [ ] Conservar imágenes originales de 640×640 y comprobar cada ROI.
- [ ] Ejecutar `python -m pytest -q` y `bash scripts/run_offline_audit.sh ...`.
- [ ] Ejecutar `validate`, `preview-rois` y la corrida E0–E4 en el equipo con SigLIP 2.
- [ ] Archivar `manifest.json`, logits y prompts para cada experimento.
- [ ] Comparar las métricas de E0 con el histórico 2/13, 1/13, 10/13, 8/13; explicar diferencias.
- [ ] Comparar E3 y E4, reportar **errores nuevos además de los corregidos**.
- [ ] Analizar todos los desacuerdos con la anotación, incluyendo los de la imagen completa.
- [ ] Atribuir correctamente el efecto de ROI a contexto **y** cambio de escala relativa.
- [ ] No escribir que las puntuaciones son probabilidades de actividad.
- [ ] No afirmar que los 13 casos prueban sesgo latinoamericano o generalización.
- [ ] Verificar derechos de redistribución de las fotos de Google Street View; mantenerlas fuera del Git público.
- [ ] Ejecutar la instalación del paquete desde un ZIP limpio, sin utilizar el CLI global viejo.
- [ ] Enviar informe, CSV de decisiones por foto, código y documentación, con procedencia.
