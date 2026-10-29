import cv2
import numpy as np
import tensorflow as tf


def generate_gradcam(
    model: tf.keras.Model,
    tensor_input: np.ndarray,
    layer_name: str | None = None,
) -> np.ndarray:
    """Genera el mapa de calor Grad-CAM para la predicción de mayor activación.

    Parámetros:
        model: Instancia del modelo Keras previamente cargado.
        tensor_input: Tensor preprocesado de entrada con forma (1, 512, 512, 1).
        layer_name: Nombre de la capa convolucional de interés. Si es None,
                    se selecciona la última capa convolucional disponible.

    Retorna:
        np.ndarray: Mapa de calor normalizado y redimensionado a 512x512.
    """
    if layer_name is None:
        # Detectar automáticamente la última capa convolucional
        for layer in reversed(model.layers):
            if "conv" in layer.name.lower():
                layer_name = layer.name
                break

    if layer_name is None:
        raise ValueError("No se encontró una capa convolucional en el modelo.")

    # Submodelo que extrae las salidas de la capa convolucional y la predicción final
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(layer_name).output, model.output]
    )

    # Cálculo de gradientes con GradientTape
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(tensor_input)
        pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Ponderación de mapas de activación por su importancia de gradiente
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Normalizar entre 0 y 1 aplicando ReLU (solo contribuciones positivas)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
    heatmap_np = heatmap.numpy()

    # Redimensionar el mapa de calor a 512x512 para empatar con la imagen preprocesada
    heatmap_resized = cv2.resize(heatmap_np, (512, 512))
    return heatmap_resized