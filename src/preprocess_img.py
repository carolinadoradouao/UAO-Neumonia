import cv2
import numpy as np


def preprocess_image(image: np.ndarray) -> np.ndarray:
    """Preprocesa un arreglo de imagen para inferencia con el modelo CNN.

    Pasos:
    1. Conversión a escala de grises si tiene 3 canales.
    2. Redimensionamiento a 512x512.
    3. Ecualización adaptativa de histograma (CLAHE).
    4. Normalización de intensidad en rango [0.0, 1.0].
    5. Conversión a formato tensor batch (1, 512, 512, 1).
    """
    # 1. Asegurar escala de grises
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()

    # Asegurar tipo de dato uint8 [0, 255] antes de aplicar CLAHE
    if gray.dtype != np.uint8:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # 2. Resize a 512x512
    resized = cv2.resize(gray, (512, 512), interpolation=cv2.INTER_AREA)

    # 3. Ecualización del histograma con CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized = clahe.apply(resized)

    # 4. Normalización entre 0 y 1 en punto flotante
    normalized = equalized.astype(np.float32) / 255.0

    # 5. Agregar dimensiones de batch y canal: (1, 512, 512, 1)
    tensor = np.expand_dims(normalized, axis=(0, -1))
    return tensor