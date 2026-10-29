import numpy as np

from src.grad_cam import generate_gradcam
from src.load_model import get_model
from src.preprocess_img import preprocess_image
from src.read_img import load_image

# Categorías diagnósticas del modelo
CLASSES = ["Bacteriana", "Viral", "Normal"]


def predict_and_explain(filepath: str) -> dict:
    """Orquesta el flujo: lectura -> preprocesamiento -> predicción -> explicación."""
    # 1. Leer imagen DICOM o convencional
    raw_img = load_image(filepath)

    # 2. Preprocesar la imagen al tamaño y formato de tensor requerido
    tensor = preprocess_image(raw_img)

    # 3. Inferencia con el modelo previamente cargado
    model = get_model()
    predictions = model.predict(tensor, verbose=0)[0]
    best_idx = int(np.argmax(predictions))
    predicted_class = (
        CLASSES[best_idx] if best_idx < len(CLASSES) else f"Clase_{best_idx}"
    )
    probability = float(predictions[best_idx])

    # 4. Generación del mapa de calor con Grad-CAM
    heatmap = generate_gradcam(model, tensor)

    return {
        "class": predicted_class,
        "probability": probability,
        "raw_shape": raw_img.shape,
        "heatmap": heatmap,
    }