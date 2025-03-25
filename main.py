from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
from db import user_profile_connection, menu_connection

app = FastAPI()

# Load precompute data
df = pd.read_excel("unique_menu_model.xlsx")
menu_embeddings = np.load("menu_embeddings.npy")
like_weight = 1.0
dislike_weight = 0.8

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/recommendation/menu/{username}")
def recommend(username: str):
    user = user_profile_connection.find_one({"user_name": username})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    liked_menus = user.get("liked_menu")
    disliked_menus = user.get("disliked_menu")

    liked_indices = [i for i, name in enumerate(
        df["menu_name"]) if name in liked_menus]
    disliked_indices = [i for i, name in enumerate(
        df["menu_name"]) if name in disliked_menus]

    liked_vectors = menu_embeddings[liked_indices] if liked_indices else np.zeros_like(
        menu_embeddings[0])
    disliked_vectors = menu_embeddings[disliked_indices] if disliked_indices else np.zeros_like(
        menu_embeddings[0])

    user_vector = np.zeros_like(menu_embeddings[0])

    if len(liked_vectors) > 0:
        user_vector += like_weight * np.mean(liked_vectors, axis=0)

    if len(disliked_vectors) > 0:
        user_vector -= dislike_weight * np.mean(disliked_vectors, axis=0)

    user_vector = user_vector.reshape(1, -1)
    scores = cosine_similarity(user_vector, menu_embeddings).flatten()

    top_n = 10
    recommendations = sorted(
        [
            (
                i,
                df.iloc[i]["menu_name"],
                df.iloc[i]["ingredients"],
                df.iloc[i]["characteristics"],
                df.iloc[i]["menu_category"],
                scores[i]
            )
            for i in range(len(scores))
            if df.iloc[i]["menu_name"] not in liked_menus + disliked_menus
        ],
        key=lambda x: x[-1],
        reverse=True
    )[:top_n]

    results = []
    for i, name, ingredients, characteristics, category, score in recommendations:
        results.append({
            "menu_name": name,
            "score": float(score),
            "ingredients": ingredients,
            "characteristics": characteristics,
            "menu_category": category
        })

    return {"results": results}
