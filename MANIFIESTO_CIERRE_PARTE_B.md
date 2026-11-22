# Parte B — Manifiesto de cierre

**Corte:** 21 de noviembre de 2026. **Release:** candidato listo para auditoría y publicación cuando se resuelva la distribución de las fotografías.

## Archivos centrales

- `nota_tecnica.pdf` (3 páginas) / `nota_tecnica.tex` — documento principal con respuestas directas y análisis F01–F13. `nota_tecnica.md` — versión ampliada de apoyo.
- `README.md` — instalación, datos, evaluación, comandos y limitaciones.
- `requirements.txt`, `pyproject.toml` — dependencias.
- `src/urban_vlm` y `scripts` — implementación E0–E5 y auditoría.
- `anotaciones.csv` / `data/annotations.csv` — 13 etiquetas con evidencia.
- `data/rois.csv` — ROI manuales; archivos auto ROI preservados bajo `reference/.../e5/`.
- `configs/` — formulaciones genérica/contextual/E3/E4 y checkpoint congelado.
- `reference/` — logits, predicciones y comparación histórica auditables sin GPU.
- `docs/diagnostico_13_fachadas.csv` — etiquetas, 7 formulaciones y explicación por fotografía.

## Verificación efectuada en el paquete

- 48 pruebas de CPU de `pytest`: **aprobadas**.
- Auditoría offline E0–E4 desde registros archivados: **ejecutada**.
- E0: 52 decisiones, 780 logits, 13 etiquetas consistentes; E5: 6 condiciones y 26 comparaciones pareadas.
- Nota técnica PDF: **3 páginas**, renderizada e inspeccionada visualmente.
- Buscador de rutas privadas de usuario en código/docs/JSON: **sin coincidencias**.

## Puntos no concluidos

1. **Faltan las 13 imágenes originales** en `data/images/`; deben copiarse sin transformación y revisar el permiso de redistribución de Google Street View y el Centro. No dar por cumplida la exigencia literal de entregar imágenes en el repositorio público.
2. No se ha ejecutado en este entorno la inferencia real sobre esas imágenes con GPU y pesos del modelo; se verificaron registros históricos y pruebas simuladas.
3. El repositorio público de GitHub aún debe crearse/subirse desde la cuenta del usuario.
4. Las métricas corresponden a un conjunto de desarrollo de 13 imágenes, no a un held-out.

## Comandos de salida

```bash
python -m pip install -r requirements.txt
python scripts/check_entrega.py
python -m pytest -q
bash scripts/run_offline_audit.sh runs/auditoria_publicacion --plots

# Para verificar y copiar el conjunto original SIN cambios (si se cuenta con autorización):
python scripts/import_originals.py --source /RUTA/AL/DIRECTORIO/DE/FOTOS
python scripts/check_entrega.py --require-images

# Solo con originales y pesos disponibles:
python -m urban_vlm run --config configs/experiment_pinned.yaml \
    --device auto --output runs/replica_e0
```

No modificar los prompts, etiquetas ni ROI históricas antes de la entrega; los métodos E3/E4/E5 se presentan como diagnósticos complementarios y no se seleccionan como ganadores por ajuste post hoc.
