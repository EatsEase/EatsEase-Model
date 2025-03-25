from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
from db import user_profile_collection, menu_collection
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
async def recommend(username: str):
    log_memory_usage("🔄 Start of /recommend route")

    user = await user_profile_collection.find_one({"user_name": username})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    liked_menus = user.get("liked_menu", [])
    disliked_menus = user.get("disliked_menu", [])
    allergies = set(user.get("allergies", []))

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

    # ----- Diversity Enforcement with Score Thresholding + Allergy Filter -----
    category_seen = set()
    recommendations = []
    sorted_indices = scores.argsort()[::-1]

    def has_allergy(menu_allergens):
        menu_allergy_set = set(str(menu_allergens).split(", "))
        return not allergies.isdisjoint(menu_allergy_set)

    # Get top 5 exact matches first (regardless of category)
    for i in sorted_indices:
        name = df.iloc[i]["menu_name"]
        if name in liked_menus or name in disliked_menus:
            continue
        if has_allergy(df.iloc[i].get("matched_allergies", "")):
            continue

        recommendations.append(
            (
                i,
                name,
                df.iloc[i]["ingredients"],
                df.iloc[i]["characteristics"],
                df.iloc[i]["menu_category"],
                scores[i]
            )
        )
        if len(recommendations) >= 5:
            break

    # Use the score of the 5th item as threshold (or lower by margin)
    similarity_threshold_1 = recommendations[-1][-1] - 0.01 if len(recommendations) >= 5 else 1.0
    similarity_threshold_2 = similarity_threshold_1 - 0.02

    # Continue adding more recommendations with diversity enforcement
    for i in sorted_indices:
        name = df.iloc[i]["menu_name"]
        if name in liked_menus or name in disliked_menus:
            continue
        if name in [r[1] for r in recommendations]:
            continue
        if has_allergy(df.iloc[i].get("matched_allergies", "")):
            continue

        category = df.iloc[i]["menu_category"]
        score = scores[i]

        if len(recommendations) < 10:
            if category in category_seen:
                continue
            if score >= similarity_threshold_1:
                continue
        elif len(recommendations) < 15:
            if score >= similarity_threshold_2:
                continue

        recommendations.append(
            (
                i,
                name,
                df.iloc[i]["ingredients"],
                df.iloc[i]["characteristics"],
                category,
                score
            )
        )
        category_seen.add(category)
        if len(recommendations) >= 15:
            break

    # Fallback to fill up to 15 if needed
    if len(recommendations) < 15:
        for i in sorted_indices:
            name = df.iloc[i]["menu_name"]
            if name in liked_menus or name in disliked_menus:
                continue
            if name in [r[1] for r in recommendations]:
                continue
            if has_allergy(df.iloc[i].get("matched_allergies", "")):
                continue

            recommendations.append(
                (
                    i,
                    name,
                    df.iloc[i]["ingredients"],
                    df.iloc[i]["characteristics"],
                    df.iloc[i]["menu_category"],
                    scores[i]
                )
            )
            if len(recommendations) >= 15:
                break

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