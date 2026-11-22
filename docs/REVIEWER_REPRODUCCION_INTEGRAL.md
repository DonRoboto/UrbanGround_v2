# Reproducir y verificar las cifras de la Parte B (E0–E5)

## Meta de reproducción

Que una tercera persona **ejecute la inferencia desde las 13 imágenes originales** con los bancos y checkpoints archivados y contraste **predicción por fotografía, coordenadas de ROI, métricas y valores logit** con los resultados usados en `nota_tecnica.pdf`. El programa **no entrena ningún modelo**.

## Requisitos y limitaciones comprobables

- Python 3.10 recomendado; Linux/Ubuntu. Para GPU, instala PyTorch compatible antes de instalar requisitos. Los resultados históricos se obtuvieron con Python 3.10.22, PyTorch 2.6.0+cu124, Transformers 4.57.6 y NVIDIA RTX 3090; pueden existir discrepancias pequeñas entre GPU/CPU.
- Los **13 originales exactos**, sin recodificar, deben estar en `data/images/fachada_01.jpg` … `fachada_13.jpg`. `check_entrega.py --require-images` exige coincidencia SHA256 respecto a los archivos con los que se obtuvieron los resultados E5. **Sin esas imágenes no es una reproducción desde cero**, aunque los CSV históricos sí sean auditables.
- `configs/experiment_pinned.yaml` fija SigLIP 2 al commit `75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2`; E5 fija OWLv2 al commit `cfd3195ba4ea9592eec887ded089f4c08eff231d`.
- Para mayor cercanía al entorno original utiliza `requirements-reproduction.txt`. Las bibliotecas, la implementación CUDA y la detección de OWLv2/OpenCV aún pueden producir variaciones. La auditoría **señala** discrepancias en vez de ocultarlas.

## Instalación limpia

```bash
# git clone URL_DEL_REPOSITORIO && cd NOMBRE_DEL_REPOSITORIO
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# En NVIDIA, instalar primero la compilación de PyTorch 2.6.0 adecuada a CUDA y driver.
python -m pip install -r requirements-reproduction.txt
python -m pip check
python -m pytest -q
```

## Modo A (recomendado): regenerar TODOS los experimentos y auditar

```bash
python scripts/check_entrega.py --require-images
CONFIG=configs/experiment_pinned.yaml DEVICE=auto OWL_DEVICE=auto \
ROI_SOURCE=regenerate \
bash scripts/reproducir_todos_los_experimentos.sh runs/revisor_01
```

El comando ejecuta:

1. **E0**: fotografía completa y ROI manual × dos bancos (52 predicciones, 780 logits).
2. **E1**: cinco reglas de agregación de E0, sin reinferencia (20 combinaciones agregadas).
3. **E2**: intervenciones espaciales (130 predicciones); no existe archivo histórico completo de E2 para cotejar sus logits, por lo que se auditan cardinalidad y coherencia frente a E0.
4. **E3**: clasificación jerárquica, presencia y evidencia de operación (26 predicciones).
5. **E4**: refinamiento de los prompts de operación conservando el primer nivel de E3 (26 pares).
6. **E5**: regeneración de ROI con OpenCV y OWLv2, luego clasificación SigLIP 2 con imagen completa y tres ROI para dos bancos (8 condiciones).
7. **Auditoría**: compara automáticamente los resultados con las referencias históricas, caso por caso.

**Archivo que indica si se obtuvieron nuestros resultados:**

`runs/revisor_01/VERIFICATION/reproduction_report.md`

El informe enumera cada comprobación `PASS`/`FAIL`; los desacuerdos precisos quedan en `VERIFICATION/differences.csv` y las condiciones de la corrida en `environment.freeze.txt`.

El proceso se detiene con código de error si la verificación falla; **un error no se declara reproducido**.

## Modo B: aislar variaciones de los detectores automáticos

En equipos distintos, OWLv2 u OpenCV pueden devolver coordenadas ligeramente diferentes. Para estudiar si el clasificador sí reproduce nuestras decisiones utilizando los **mismos recortes automáticos de la ejecución histórica**, ejecute:

```bash
CONFIG=configs/experiment_pinned.yaml DEVICE=auto \
ROI_SOURCE=historical \
bash scripts/reproducir_todos_los_experimentos.sh runs/revisor_roi_fijas
```

Este modo **vuelve a ejecutar SigLIP 2 sobre las tres ROI y las imágenes completas**, pero utiliza las propuestas ROI históricas almacenadas en `reference/` en lugar de volver a ejecutar los localizadores de E5. **No sustituye al Modo A**, ya que no prueba la regeneración de ROI automáticas. Permite identificar por qué difiere el resultado (localización frente a clasificación).

## Cifras esperadas (13 fotografías; conjunto de desarrollo)

| Entrada | Banco genérico | Banco contextual |
|---|---:|---:|
| Imagen completa, sin ROI | 2/13 (15.4 %) | 1/13 (7.7 %) |
| ROI manual | 10/13 (76.9 %) | 8/13 (61.5 %) |
| ROI automática OpenCV | 7/13 (53.8 %) | 8/13 (61.5 %) |
| ROI automática OWLv2 | 9/13 (69.2 %) | 7/13 (53.8 %) |

Además: ROI E5 media IoU OpenCV ~0.285 y OWLv2 ~0.360; E3 ROI 9/13 (macro-F1 ~0.778); E4 ROI 9/13 (macro-F1 ~0.771). El comparador verifica predicciones individuales, no solamente estos totales.

## Revisión histórica sin imágenes ni GPU (NO equivale a reinferencia)

```bash
python scripts/verificar_reproduccion.py --reference-only
bash scripts/run_offline_audit.sh runs/audit_archived --plots
```

Esto valida consistencia de los archivos históricos y reconstruye tablas, pero **no ejecuta nuevamente SigLIP 2**. Debe describirse explícitamente como auditoría de resultados, no como reproducción completa.

## Interpretación de diferencias

- `E5-ROI FAIL`: el método de localización produjo coordenadas distintas. Puede afectar la clasificación. Comprueba versión de OpenCV, OWLv2 y precisión del dispositivo. Verifica si el Modo B coincide.
- `E0/E3/E4 FAIL` en predicciones: comprobar SHA256 de imágenes, textos, checkpoint, tokenizer y entorno. No modificar prompts o umbrales para forzar coincidencia.
- `Logit FAIL` con mismas predicciones: tolerancia absoluta predeterminada `0.01`; se puede reportar la variación sin sostener identidad bit a bit.
- `E2`: solo comprobaciones de integridad y consistencia con E0; **no se asegura reproducción numérica exacta histórica** porque no están archivadas todas las puntuaciones originales de E2.
- Las 13 fotografías son el **conjunto de desarrollo**, sin held-out. Las predicciones comparables no implican generalización.

**No redistribuir capturas de Google Street View si no se tienen los permisos necesarios.** Para una entrega reproducible, el evaluador debe disponer legalmente de los mismos archivos originales, identificados por sus hashes.
