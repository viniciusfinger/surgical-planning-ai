from fastapi import APIRouter

router = APIRouter()


@router.get("/")
def hello() -> dict[str, str]:
    return {"message": "Hello World"}
