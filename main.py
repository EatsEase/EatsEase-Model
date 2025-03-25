from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
from db import user_profile_connection, menu_connection
from utils import log_memory_usage

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
    log_memory_usage("🔄 Start of /recommend route")

    user = user_profile_connection.find_one({"user_name": username})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    liked_menus = user.get("liked_menu", [])
    disliked_menus = user.get("disliked_menu", [])

    log_memory_usage("📋 Retrieved liked/disliked menus from DB")

    liked_indices = [i for i, name in enumerate(df["menu_name"]) if name in liked_menus]
    disliked_indices = [i for i, name in enumerate(df["menu_name"]) if name in disliked_menus]

    liked_vectors = menu_embeddings[liked_indices] if liked_indices else np.zeros_like(menu_embeddings[0])
    disliked_vectors = menu_embeddings[disliked_indices] if disliked_indices else np.zeros_like(menu_embeddings[0])

    log_memory_usage("📊 Vectors loaded for liked/disliked menus")

    user_vector = np.zeros_like(menu_embeddings[0])
    if len(liked_vectors) > 0:
        user_vector += like_weight * np.mean(liked_vectors, axis=0)
    if len(disliked_vectors) > 0:
        user_vector -= dislike_weight * np.mean(disliked_vectors, axis=0)
    user_vector = user_vector.reshape(1, -1)

    log_memory_usage("🧠 User vector computed")

    scores = cosine_similarity(user_vector, menu_embeddings).flatten()

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
    )[:10]

    log_memory_usage("✅ Recommendations generated")

    results = [
        {
            "menu_name": name,
            "score": float(score),
            "ingredients": ingredients,
            "characteristics": characteristics,
            "menu_category": category
        }
        for i, name, ingredients, characteristics, category, score in recommendations
    ]

    log_memory_usage("🚀 Returning results")
    return {"results": results}

