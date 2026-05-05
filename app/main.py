from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

from fastapi import FastAPI

from api.controller.hello_controller import router as hello_router
from api.controller.surgical_planning_controller import router as surgical_planning_router

app = FastAPI(title="surgical-planning-ai", version="0.1.0")
app.include_router(hello_router)
app.include_router(surgical_planning_router)
