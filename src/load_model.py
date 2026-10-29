import os

from tensorflow.keras.models import load_model as keras_load_model

# Variable global para almacenar la instancia del modelo en memoria (Singleton)
_MODEL = None

def get_model(model_path: str = "Model/conv_MLP_84.h5"):
    """
    Carga el modelo de red neuronal convolucional (.h5) una sola vez en memoria.
    Si ya está cargado, retorna la instancia existente.
    """
    global _MODEL
    if _MODEL is None:
        # Busca el modelo en Model/ o en models/
        target_path = model_path
        if not os.path.exists(target_path):
            alt_path = "models/conv_MLP_84.h5"
            if os.path.exists(alt_path):
                target_path = alt_path

        if os.path.exists(target_path):
            # compile=False para inferencia rápida sin requerir optimizadores de entrenamiento
            _MODEL = keras_load_model(target_path, compile=False)
        else:
            raise FileNotFoundError(
                f"No se encontró el archivo del modelo en '{target_path}'. "
                "Recuerda descargarlo del aula virtual y colocarlo en la carpeta Model/."
            )
    return _MODEL