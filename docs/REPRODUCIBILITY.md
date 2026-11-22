# Reproducibilidad de UrbanGroundFloor

## Artefactos
`configs/experiment.yaml`, los cuatro bancos de prompts JSON, `data/rois.csv` y `data/annotations.csv` definen el experimento. Las fotografías originales (640×640) se añaden localmente y no se suben automáticamente a GitHub. Los `runs/` se mantienen fuera de Git: conserva el directorio exacto para reproducir resultados individuales.

Para **archivos completos** se espera: E0=52 decisiones/780 logits; E2=130/1950; E3=26/520; E4=26/520 (260 de E3 inalterados y 260 nuevos). Son conteos de estructura, no garantías de validez de las predicciones.

## Entorno histórico registrado, no forzosamente requerido
En una ejecución anterior (E3, 2026-11-21) se registró:
- Python 3.10.22, Linux 6.8 x86_64.
- PyTorch 2.6.0+cu124, Transformers 4.57.6, NumPy 2.2.6, Pillow 12.3.0.
- GPU NVIDIA GeForce RTX 3090, runtime CUDA 12.4.
- Checkpoint `google/siglip2-base-patch16-224`, commit detectado: `75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2`.

`model_revision: null` permite descargar la revisión actual, pero para un nuevo conjunto cerrado se recomienda fijar `model_revision` al commit anterior **antes** de ejecutar de nuevo todas las etapas; E4 solo admite un checkpoint compatible con E3. Las imágenes y ROI deben seguir siendo las mismas. Los dispositivos y librerías diferentes pueden producir pequeñas variaciones numéricas incluso con semilla fija.

## Trazabilidad
Cada ejecución crea un manifiesto; el informe consolidado genera `study_inputs.json` con SHA256 de los CSV utilizados. La versión pública del manifiesto histórico de E3 oculta las rutas absolutas del equipo del usuario; se conserva el commit y los hashes. El historial del desarrollo no equivale a una nueva ejecución.

## Prueba local
```bash
python -m pip install -e ".[report,test]"
python -m pytest -q
bash scripts/run_offline_audit.sh runs/offline_01 --plots
```
Estos comandos no necesitan fotografías ni checkpoint. **Para verificar la inferencia**, añade las 13 fotos y las dependencias de GPU/CPU de `.[inference]`, valida ROI y ejecuta `scripts/run_full_study.sh`.

## Integridad y versiones
Si una ejecución cambia las ROI, etiquetas o prompts, conserva el archivo original, registra la modificación y ejecuta todo el conjunto bajo un **nuevo ID de experimento**. No reutilices el reporte E3 de otra revisión cuando ejecutas E4: el validador E4 bloquea las mezclas de hashes y logits.

## Límites
El valor numérico de accuracy no prueba desempeño generalizable ni sesgo cultural o geográfico. SigLIP 2 calcula compatibilidades imagen–texto, no probabilidades calibradas de apertura del negocio.
