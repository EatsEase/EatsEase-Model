from fastapi import APIRouter

router = APIRouter(
    prefix="/api/recommendation",
    tags=["recommendation"],
    responses={404: {"description": "Not found"}},
)