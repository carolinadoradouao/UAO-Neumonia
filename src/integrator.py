from fastapi import FastAPI

app = FastAPI(title="Detector Neumonia API")

@app.get("/")
def read_root():
    return {"status": "ok", "message": "Pipeline de deteccion listo para integrar"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("integrator:app", host="127.0.0.1", port=8000, reload=True)