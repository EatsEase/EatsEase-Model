from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
from db import user_profile_collection, menu_collection
from utils import log_memory_usage

app = FastAPI()

# Load precompute data
df = pd.read_excel("unique_with_cloud_image.xlsx")
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
        cold_start = sampled[["menu_name", "menu_category"]].to_dict(orient="records")
        await user_profile_collection.update_one(
        {"_id": user["_id"]},  # Use user's ObjectId to update
        {"$set": {"cold_start": cold_start}}  # Set cold_start field
    )
        return {
            "results": cold_start
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

    # Top 3 from food_preferences only
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
        if len(recommendations) >= 3:
            break
        
    # Final 5: diverse menus not in food_preferences
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

        recommendations.append((
            i,
            name,
            df.iloc[i]["ingredients"],
            df.iloc[i]["characteristics"],
            category,
            score
        ))
        if len(recommendations) >= 5:
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

@app.get("/api/evaluate/loo2/{username}")
async def leave_one_out_evaluation2(username: str, top_n: int = 20):
    threshold = 0.5
    log_memory_usage("🔄 Start of LOO Evaluation")

    # Fetch user profile from DB
    user = await user_profile_collection.find_one({"user_name": username})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    liked_menus = user.get("liked_menu", [])
    disliked_menus = user.get("disliked_menu", [])
    allergies = set(user.get("allergies", []))  # ✅ Already converted to set


    # Check if the user has enough liked menus to perform LOO
    if len(liked_menus) < 3:
        raise HTTPException(
            status_code=400, detail="Not enough liked menus for evaluation"
        )

    hit_count = 0
    dislike_hits = 0
    total_tests = 0
    y_true = []
    y_pred = []
    all_results = []
    
    def is_safe_menu(menu_allergens):
        menu_allergy_set = set(str(menu_allergens).split(", "))
        return allergies.isdisjoint(menu_allergy_set)

    # Iterate through each liked menu (Leave-One-Out)
    for target_menu in liked_menus:
        # Create a new list excluding the target menu
        reduced_likes = [menu for menu in liked_menus if menu != target_menu]

        # Generate user vector based on reduced likes
        liked_indices = [
            i for i, name in enumerate(df["menu_name"]) if name in reduced_likes
        ]
        liked_vectors = (
            menu_embeddings[liked_indices]
            if liked_indices
            else np.zeros_like(menu_embeddings[0])
        )

        disliked_indices = [
            i for i, name in enumerate(df["menu_name"]) if name in disliked_menus
        ]
        disliked_vectors = (
            menu_embeddings[disliked_indices]
            if disliked_indices
            else np.zeros_like(menu_embeddings[0])
        )

        # Calculate user vector with weighted penalization
        user_vector = np.zeros_like(menu_embeddings[0])
        if len(liked_vectors) > 0:
            user_vector += like_weight * np.mean(liked_vectors, axis=0)
        if len(disliked_vectors) > 0:
            user_vector -= dislike_weight * np.mean(disliked_vectors, axis=0)

        user_vector = user_vector.reshape(1, -1)

        # Calculate cosine similarity
        scores = cosine_similarity(user_vector, menu_embeddings).flatten()
        sorted_indices = scores.argsort()[::-1][:top_n]
        recommended_menus = df.iloc[sorted_indices]["menu_name"].tolist()
        for i in sorted_indices:
            menu_name = df.iloc[i]["menu_name"]
            if is_safe_menu(df.iloc[i]["matched_allergies"]):
                recommended_menus.append(menu_name)
            if len(recommended_menus) >= top_n:
                break

        # Check if target_menu is in top-N recommendations
        is_hit = target_menu in recommended_menus
        hit_count += int(is_hit)
        total_tests += 1

        # Check if any disliked menu is in top-N recommendations
        disliked_hits_count = sum(
            1 for menu in disliked_menus if menu in recommended_menus
        )
        dislike_hits += disliked_hits_count

        # Calculate similarity of target menu
        target_index = df[df["menu_name"] == target_menu].index
        target_similarity = (
            scores[target_index[0]] if len(target_index) > 0 else 0.0
        )

        # Classify based on threshold
        predicted_label = "like" if target_similarity >= threshold else "dislike"
        actual_label = "like"  # Ground truth since target_menu is from liked_menus

        # Append to evaluation lists
        y_true.append(actual_label)
        y_pred.append(predicted_label)

        # Track results for analysis
        # Correct this section where you append the results
        all_results.append(
            {
                "target_menu": target_menu,
                "recommended_menus": recommended_menus,
                "hit": is_hit,
                "disliked_hit_count": int(disliked_hits_count),  # Convert to int
                "predicted_label": predicted_label,
                "actual_label": actual_label,
                "similarity_score": round(float(target_similarity), 4),  # ✅ Fix here
            }
        )


    # Calculate classification metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, pos_label="like")
    recall = recall_score(y_true, y_pred, pos_label="like")
    f1 = f1_score(y_true, y_pred, pos_label="like")

    # Calculate FPR (False Positive Rate) for disliked menus
    fpr = (
        dislike_hits / (len(disliked_menus) * total_tests)
        if len(disliked_menus) > 0
        else 0
    )

    # Calculate hit rate
    hit_rate = hit_count / total_tests if total_tests > 0 else 0

    log_memory_usage("✅ LOO Evaluation completed")

    return {
        "hit_rate": round(float(hit_rate), 2),  # ✅ Convert to float
        "precision": round(float(precision), 2),
        "recall": round(float(recall), 2),
        "f1_score": round(float(f1), 2),
        "fpr": round(float(fpr), 2),
        "total_tests": int(total_tests),  # ✅ Convert to int
        "hits": int(hit_count),
        "disliked_hits": int(dislike_hits),
        "detailed_results": all_results
    }
@app.get("/api/evaluate/loo/{username}")
async def leave_one_out_evaluation(username: str, top_n: int = 10):
    threshold = 0.5  # Define similarity threshold
    log_memory_usage("🔄 Start of LOO Evaluation")

    # Fetch user profile from DB
    user = await user_profile_collection.find_one({"user_name": username})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    liked_menus = user.get("liked_menu", [])
    disliked_menus = user.get("disliked_menu", [])
    allergies = set(user.get("allergies", []))
    food_preferences = set(user.get("food_preferences", []))

    # Check if the user has enough liked/disliked menus to perform LOO
    if len(liked_menus) < 6 or len(disliked_menus) < 6:
        raise HTTPException(
            status_code=400,
            detail="Not enough interactions to exclude cold start and evaluate.",
        )

    # 🎯 Step 1: Remove the first liked and first 2 disliked menus (Cold Start Removal)
    post_cold_start_liked_menus = liked_menus[1:]  # Remove first liked (cold start)
    post_cold_start_disliked_menus = disliked_menus[2:]  # Remove first 2 disliked

    hits_liked = 0
    hits_disliked = 0
    dislike_hits = 0
    total_tests_liked = 0
    total_tests_disliked = 0
    y_true = []
    y_pred = []
    all_results = []

    # 🎯 Step 2: Iterate through both liked and disliked menus for evaluation
    for target_menu in post_cold_start_liked_menus + post_cold_start_disliked_menus:
        # Define ground truth label
        actual_label = "like" if target_menu in post_cold_start_liked_menus else "dislike"

        # Create a new list excluding the target menu
        reduced_likes = [
            menu for menu in post_cold_start_liked_menus if menu != target_menu
        ]

        # Generate user vector based on reduced likes (after cold start removed)
        liked_indices = [
            i for i, name in enumerate(df["menu_name"]) if name in reduced_likes
        ]
        liked_vectors = (
            menu_embeddings[liked_indices]
            if liked_indices
            else np.zeros_like(menu_embeddings[0])
        )

        disliked_indices = [
            i
            for i, name in enumerate(df["menu_name"])
            if name in post_cold_start_disliked_menus
        ]
        disliked_vectors = (
            menu_embeddings[disliked_indices]
            if disliked_indices
            else np.zeros_like(menu_embeddings[0])
        )

        # ✅ Step 3: Calculate user vector with penalization
        user_vector = np.zeros_like(menu_embeddings[0])
        if len(liked_vectors) > 0:
            user_vector += like_weight * np.mean(liked_vectors, axis=0)
        if len(disliked_vectors) > 0:
            user_vector -= dislike_weight * np.mean(disliked_vectors, axis=0)

        user_vector = user_vector.reshape(1, -1)

        # ✅ Step 4: Calculate cosine similarity
        scores = cosine_similarity(user_vector, menu_embeddings).flatten()
        sorted_indices = scores.argsort()[::-1]

        # ✅ Step 5: Generate Top 10 + Diverse 5 Recommendations
        top_10_recommendations = []
        diverse_5_recommendations = []

        def has_allergy(menu_allergens):
            menu_allergy_set = set(str(menu_allergens).split(", "))
            return not allergies.isdisjoint(menu_allergy_set)

        def matches_food_preference(category_string):
            category_list = set(str(category_string).split(","))
            return not food_preferences.isdisjoint(category_list)

        # Generate Top 10 matching food preferences
        for i in sorted_indices:
            name = df.iloc[i]["menu_name"]
            category = df.iloc[i]["menu_category"]
            if name in reduced_likes:
                continue
            if has_allergy(df.iloc[i].get("matched_allergies", "")):
                continue
            if not matches_food_preference(category):
                continue
            if len(top_10_recommendations) < 10:
                top_10_recommendations.append(name)
            else:
                break

        # Set threshold for diverse recommendations
        similarity_threshold_2 = (
            scores[sorted_indices[9]] - 0.03
            if len(top_10_recommendations) >= 10
            else 1.0
        )

        # Generate Final 5 diverse recommendations
        for i in sorted_indices:
            name = df.iloc[i]["menu_name"]
            category = df.iloc[i]["menu_category"]
            score = scores[i]

            if (
                name in reduced_likes
                or name in top_10_recommendations
            ):
                continue
            if has_allergy(df.iloc[i].get("matched_allergies", "")):
                continue
            if matches_food_preference(category):
                continue
            if score >= similarity_threshold_2:
                continue
            if len(diverse_5_recommendations) < 5:
                diverse_5_recommendations.append(name)
            else:
                break

        # ✅ Step 6: Merge top 10 and diverse 5 for final evaluation
        recommended_menus = top_10_recommendations + diverse_5_recommendations

        # ✅ Step 7: Check Hits for Liked and Disliked Menus
        is_hit_top10 = target_menu in top_10_recommendations
        is_hit_diverse5 = target_menu in diverse_5_recommendations

        if target_menu in post_cold_start_liked_menus:
            hits_liked += int(is_hit_top10 or is_hit_diverse5)
            total_tests_liked += 1
        else:
            hits_disliked += int(is_hit_top10 or is_hit_diverse5)
            total_tests_disliked += 1

        # ✅ Step 8: Check if any disliked menu is in recommendations (False Positives)
        disliked_hits_count = sum(
            1 for menu in post_cold_start_disliked_menus if menu in recommended_menus
        )
        dislike_hits += disliked_hits_count

        # ✅ Step 9: Calculate similarity of target menu
        target_index = df[df["menu_name"] == target_menu].index
        target_similarity = (
            scores[target_index[0]] if len(target_index) > 0 else 0.0
        )

        # ✅ Step 10: Classify based on threshold
        predicted_label = "like" if target_similarity >= threshold else "dislike"

        # ✅ Step 11: Track results for analysis
        y_true.append(actual_label)
        y_pred.append(predicted_label)

        all_results.append(
            {
                "target_menu": target_menu,
                "recommended_menus": recommended_menus,
                "hit_top10": is_hit_top10,
                "hit_diverse5": is_hit_diverse5,
                "disliked_hit_count": int(disliked_hits_count),
                "predicted_label": predicted_label,
                "actual_label": actual_label,
                "similarity_score": round(float(target_similarity), 4),
            }
        )

    # ✅ Step 12: Calculate classification metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, pos_label="like")
    recall = recall_score(y_true, y_pred, pos_label="like")
    f1 = f1_score(y_true, y_pred, pos_label="like")

    # ✅ Step 13: Calculate FPR (False Positive Rate) for disliked menus
    fpr = (
        dislike_hits / (len(post_cold_start_disliked_menus) * (total_tests_liked + total_tests_disliked))
        if len(post_cold_start_disliked_menus) > 0
        else 0
    )

    # ✅ Step 14: Calculate hit rate for Top 10 and Diverse 5
    hit_rate_liked_top10 = hits_liked / total_tests_liked if total_tests_liked > 0 else 0
    hit_rate_disliked_top10 = hits_disliked / total_tests_disliked if total_tests_disliked > 0 else 0
    overall_hit_rate = (hits_liked + hits_disliked) / (total_tests_liked + total_tests_disliked)

    log_memory_usage("✅ LOO Evaluation completed (Post-Cold Start)")

    return {
        "hit_rate_liked_top10": round(float(hit_rate_liked_top10), 2),
        "hit_rate_disliked_top10": round(float(hit_rate_disliked_top10), 2),
        "overall_hit_rate": round(float(overall_hit_rate), 2),
        "precision": round(float(precision), 2),
        "recall": round(float(recall), 2),
        "f1_score": round(float(f1), 2),
        "fpr": round(float(fpr), 2),
        "total_tests_liked": int(total_tests_liked),
        "total_tests_disliked": int(total_tests_disliked),
        "hits_liked_top10": int(hits_liked),
        "hits_disliked_top10": int(hits_disliked),
        "disliked_hits": int(dislike_hits),
        "detailed_results": all_results,
    }