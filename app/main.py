from fastapi import FastAPI

from controller.hello_controller import router as hello_router

app = FastAPI(title="surgical-planning-ai", version="0.1.0")
app.include_router(hello_router)
