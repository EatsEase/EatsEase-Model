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
    food_preferences = set(user.get("food_preferences", []))

    log_memory_usage("📋 Retrieved liked/disliked menus from DB")
    
    def is_safe_menu(menu_allergens):
        menu_allergy_set = set(str(menu_allergens).split(", "))
        return allergies.isdisjoint(menu_allergy_set)
    
    if liked_menus == [] and disliked_menus == []:
        candidates = df[
            df["menu_category"].apply(
                lambda x: bool(set(str(x).split(",")).intersection(food_preferences))
            ) &  # menu must match at least one preferred category
            df["matched_allergies"].apply(is_safe_menu)  # menu must not conflict with allergies
        ]

        sampled = candidates.sample(n=min(5, len(candidates)))
        return {
            "results": sampled[["menu_name", "menu_category"]].to_dict(orient="records")
        }

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
    sorted_indices = scores.argsort()[::-1]

    def has_allergy(menu_allergens):
        menu_allergy_set = set(str(menu_allergens).split(", "))
        return not allergies.isdisjoint(menu_allergy_set)

    def matches_food_preference(category_string):
        category_list = set(str(category_string).split(","))
        return not food_preferences.isdisjoint(category_list)

    recommendations = []

    # Top 10 from food_preferences only
    for i in sorted_indices:
        name = df.iloc[i]["menu_name"]
        category = df.iloc[i]["menu_category"]
        if name in liked_menus or name in disliked_menus:
            continue
        if has_allergy(df.iloc[i].get("matched_allergies", "")):
            continue
        if not matches_food_preference(category):
            continue

        recommendations.append((
            i,
            name,
            df.iloc[i]["ingredients"],
            df.iloc[i]["characteristics"],
            category,
            scores[i]
        ))
        if len(recommendations) >= 10:
            break

    similarity_threshold_2 = recommendations[-1][-1] - 0.03 if len(recommendations) >= 10 else 1.0

    # Final 5: diverse menus not in food_preferences and lower score
    for i in sorted_indices:
        name = df.iloc[i]["menu_name"]
        category = df.iloc[i]["menu_category"]
        score = scores[i]
        if name in liked_menus or name in disliked_menus:
            continue
        if name in [r[1] for r in recommendations]:
            continue
        if has_allergy(df.iloc[i].get("matched_allergies", "")):
            continue
        if matches_food_preference(category):
            continue
        if score >= similarity_threshold_2:
            continue

        recommendations.append((
            i,
            name,
            df.iloc[i]["ingredients"],
            df.iloc[i]["characteristics"],
            category,
            score
        ))
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


@app.get("/api/recommendation/next_meal/{username}")
async def next_meal(username: str):
    log_memory_usage("🔄 Start of /recommend route")

    user = await user_profile_collection.find_one({"user_name": username})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    liked_menus = user.get("liked_menu", [])
    disliked_menus = user.get("disliked_menu", [])
    allergies = set(user.get("allergies", []))
    temp_recommend = set(user.get("temp_recommend", []))

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

    sorted_indices = scores.argsort()[::-1]

    def has_allergy(menu_allergens):
        menu_allergy_set = set(str(menu_allergens).split(", "))
        return not allergies.isdisjoint(menu_allergy_set)

    for i in sorted_indices:
        name = df.iloc[i]["menu_name"]
        if name in liked_menus or name in disliked_menus or name in temp_recommend:
            continue
        if has_allergy(df.iloc[i].get("matched_allergies", "")):
            continue

        # Update temp_recommend
        temp_recommend.add(name)
        await user_profile_collection.update_one(
            {"user_name": username},
            {"$set": {"temp_recommend": list(temp_recommend)}}
        )
        log_memory_usage("🚀 Returning results")
        return {"menu_name": name}

    raise HTTPException(status_code=404, detail="No suitable menu found")