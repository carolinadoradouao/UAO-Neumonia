import numpy as np
import pytest

from src.load_model import get_model
from src.preprocess_img import preprocess_image
from src.read_img import load_image


# PRUEBAS DEL MÓDULO: src/read_img.py
def test_read_img_nonexistent_file_raises_error():
    """Prueba 1: Intento de lectura de archivo inexistente lanza error."""
    # Arrange
    fake_path = "non_existent_image_path.jpg"

    # Act & Assert
    with pytest.raises(ValueError, match="No se pudo cargar o encontrar"):
        load_image(fake_path)


def test_read_img_jpeg_valid(tmp_path):
    """Prueba 2: Lectura de una imagen estándar devuelve ndarray uint8."""
    # Arrange
    import cv2

    img_file = tmp_path / "test_xray.png"
    dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.imwrite(str(img_file), dummy_img)

    # Act
    loaded = load_image(str(img_file))

    # Assert
    assert isinstance(loaded, np.ndarray)
    assert loaded.shape == (100, 100, 3)
    assert loaded.dtype == np.uint8



# PRUEBAS DEL MÓDULO: src/preprocess_img.py
def test_preprocess_output_shape_rgb():
    """Prueba 3: Imagen RGB se transforma a la forma exacta (1, 512, 512, 1)."""
    # Arrange
    dummy_rgb = np.random.randint(0, 256, (300, 400, 3), dtype=np.uint8)

    # Act
    tensor = preprocess_image(dummy_rgb)

    # Assert
    assert tensor.shape == (1, 512, 512, 1)


def test_preprocess_output_shape_grayscale():
    """Prueba 4: Imagen en escala de grises conserva forma de tensor batch."""
    # Arrange
    dummy_gray = np.random.randint(0, 256, (200, 200), dtype=np.uint8)

    # Act
    tensor = preprocess_image(dummy_gray)

    # Assert
    assert tensor.shape == (1, 512, 512, 1)


def test_preprocess_normalization_range():
    """Prueba 5: Los valores del tensor resultante están en el rango [0.0, 1.0]."""
    # Arrange
    dummy_img = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)

    # Act
    tensor = preprocess_image(dummy_img)

    # Assert
    assert np.min(tensor) >= 0.0
    assert np.max(tensor) <= 1.0
    assert tensor.dtype == np.float32


def test_preprocess_dtype_float32():
    """Prueba 6: El tensor debe ser obligatoriamente float32 para TensorFlow."""
    # Arrange
    dummy_img = np.ones((512, 512), dtype=np.uint8) * 128

    # Act
    tensor = preprocess_image(dummy_img)

    # Assert
    assert tensor.dtype == np.float32


# PRUEBAS DEL MÓDULO: src/load_model.py
def test_load_model_missing_file_raises_error():
    """Prueba 7: Intento de cargar modelo desde ruta inexistente lanza FileNotFoundError."""
    # Arrange
    invalid_model_path = "non_existent_folder/fake_model.h5"

    # Act & Assert
    with pytest.raises(FileNotFoundError, match="No se encontró el archivo del modelo"):
        get_model(invalid_model_path)


# PRUEBAS DEL MÓDULO: src/integrator.py
def test_integrator_classes_definition():
    """Prueba 8: Verificar que las clases diagnósticas coincidan con las 3 requeridas."""
    # Arrange
    from src.integrator import CLASSES

    # Act
    total_classes = len(CLASSES)

    # Assert
    assert total_classes == 3
    assert "Bacteriana" in CLASSES
    assert "Viral" in CLASSES
    assert "Normal" in CLASSES


def test_integrator_predict_raises_on_invalid_file():
    """Prueba 9: El flujo orquestador falla de forma controlada si el archivo no existe."""
    # Arrange
    from src.integrator import predict_and_explain

    # Act & Assert
    with pytest.raises(ValueError):
        predict_and_explain("invalid_path.jpg")



# PRUEBAS DEL MÓDULO: src/ui.py (Regla de Desacoplamiento)

def test_ui_does_not_import_tensorflow():
    """Prueba 10 (Verificación Dura): src/ui.py no debe importar TensorFlow ni Keras."""
    # Arrange
    import inspect

    import src.ui as ui_module

    # Act
    source_code = inspect.getsource(ui_module)

    # Assert
    assert "import tensorflow" not in source_code
    assert "from tensorflow" not in source_code
    assert "import keras" not in source_code


def test_ui_does_not_import_load_model():
    """Prueba 11 (Verificación Dura): src/ui.py no debe importar directamente load_model."""
    # Arrange
    import inspect

    import src.ui as ui_module

    # Act
    source_code = inspect.getsource(ui_module)

    # Assert
    assert "load_model" not in source_code
    assert "src.load_model" not in source_code