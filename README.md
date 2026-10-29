# Herramienta para la Detección Rápida de Neumonía

Deep Learning aplicado al procesamiento de imágenes radiográficas de tórax en formato DICOM y JPEG con el fin de clasificarlas en 3 categorías diferentes:

1. Neumonía Bacteriana
2. Neumonía Viral
3. Sin Neumonía

Incluye la aplicación de la técnica de explicabilidad **Grad-CAM** para resaltar con un mapa de calor las regiones relevantes de la imagen analizada.

---

## Requisitos Previos

- **Python 3.13** (gestionado automáticamente por `uv`).
- **[uv](https://docs.astral.sh/uv/)** instalado en el sistema.
- **Make** (opcional, para usar los comandos del `Makefile`).

---

## Instalación y Ejecución Rápida

Este proyecto utiliza `uv` para garantizar un entorno determinista y reproducible.

### 1. Clonar el repositorio
```bash
git clone [https://github.com/dalquinones/UAO-Neumonia.git](https://github.com/dalquinones/UAO-Neumonia.git)
cd UAO-Neumonia