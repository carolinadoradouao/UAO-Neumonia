import tkinter as tk
from tkinter import filedialog, messagebox

# REGLA DURA DEL CURSO: src/ui.py solo puede importar desde integrator
from src.integrator import predict_and_explain


def create_app() -> tk.Tk:
    """Construye y configura la ventana principal de la interfaz gráfica."""
    root = tk.Tk()
    root.title("Detección Rápida de Neumonía — UAO")
    root.geometry("480x280")
    root.resizable(False, False)

    label_title = tk.Label(
        root,
        text="Clasificador de Radiografías de Tórax",
        font=("Arial", 14, "bold"),
    )
    label_title.pack(pady=12)

    label_desc = tk.Label(
        root,
        text="Formatos compatibles: DICOM (.dcm), JPEG y PNG",
        font=("Arial", 9),
        fg="gray",
    )
    label_desc.pack()

    label_status = tk.Label(
        root,
        text="Seleccione una imagen radiográfica para comenzar",
        font=("Arial", 11),
        wraplength=420,
    )
    label_status.pack(pady=15)

    def on_select_file():
        file_path = filedialog.askopenfilename(
            title="Seleccionar Radiografía",
            filetypes=[
                ("Archivos médicos e imágenes", "*.dcm *.jpeg *.jpg *.png"),
                ("Todos los archivos", "*.*"),
            ],
        )
        if not file_path:
            return

        label_status.config(
            text="Procesando imagen y calculando inferencia...", fg="black"
        )
        root.update_idletasks()

        try:
            result = predict_and_explain(file_path)
            prob_percent = result["probability"] * 100
            label_status.config(
                text=(
                    f"Diagnóstico: {result['class']}\n"
                    f"Confianza: {prob_percent:.2f}%\n"
                    f"Mapa Grad-CAM generado: {result['heatmap'].shape}"
                ),
                fg="green",
            )
        except FileNotFoundError as fnf_err:
            messagebox.showwarning("Modelo no encontrado", str(fnf_err))
            label_status.config(text="Falta el archivo del modelo .h5.", fg="red")
        except Exception as err:  # noqa: BLE001
            messagebox.showerror("Error de Inferencia", f"Ocurrió un error: {err}")
            label_status.config(text="Error al procesar la radiografía.", fg="red")

    btn_load = tk.Button(
        root,
        text="Cargar Radiografía y Predecir",
        command=on_select_file,
        bg="#0066cc",
        fg="white",
        font=("Arial", 10, "bold"),
        padx=12,
        pady=6,
        cursor="hand2",
    )
    btn_load.pack(pady=15)

    return root


def run():
    """Punto de entrada ejecutable para el módulo ui."""
    app = create_app()
    app.mainloop()


if __name__ == "__main__":
    run()