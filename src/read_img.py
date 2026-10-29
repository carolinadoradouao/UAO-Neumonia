import cv2
import numpy as np
import pydicom


def load_image(filepath: str) -> np.ndarray:
    """Lee una imagen en formato DICOM (.dcm) o estándar (.jpeg, .png).

    Retorna un arreglo NumPy con la imagen en formato RGB o escala de grises.
    """
    filepath_lower = filepath.lower()

    if filepath_lower.endswith(".dcm"):
        # Lectura de metadatos y pixeles DICOM
        dicom_data = pydicom.dcmread(filepath)
        img_array = dicom_data.pixel_array.astype(np.float32)

        # Escalar el rango dinámico de la radiografía a [0, 255]
        max_val = img_array.max()
        if max_val > 0:
            img_array = (np.maximum(img_array, 0) / max_val) * 255.0
        return np.uint8(img_array)
    else:
        # Lectura de imágenes estándar con OpenCV
        img_bgr = cv2.imread(filepath)
        if img_bgr is None:
            raise ValueError(f"No se pudo cargar o encontrar la imagen en: {filepath}")
        # Convertir BGR a RGB estándar
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)