from src.integrator import app

if __name__ == "__main__":
    import uvicorn
    print("Iniciando servicio desde ui.py...")
    uvicorn.run("src.integrator:app", host="127.0.0.1", port=8000, reload=True)