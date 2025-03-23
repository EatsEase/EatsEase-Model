from fastapi import APIRouter
from db import menu_connection

router = APIRouter(
    prefix="/api/recommendation",
    tags=["recommendation"],
    responses={404: {"description": "Not found"}},
)

@router.get('/menu')
def recommend_menu():
    all_menu = list(menu_connection.find({}, {"_id": 0}))
    return {"data": all_menu}