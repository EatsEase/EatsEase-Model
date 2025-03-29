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
    cold_start_menus = [menu["menu_name"] for menu in user.get("cold_start", [])]
    
        # ✅ Step 1: Remove Cold Start Menus from liked and disliked lists
    liked_menus = [menu for menu in liked_menus if menu not in cold_start_menus]
    disliked_menus = [menu for menu in disliked_menus if menu not in cold_start_menus]

    # 🎯 Step 1: Remove the first liked and first 2 disliked menus (Cold Start Removal)
    post_cold_start_liked_menus = liked_menus
    post_cold_start_disliked_menus = disliked_menus

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

        # Generate Final 5 diverse recommendations
        for i in sorted_indices:
            name = df.iloc[i]["menu_name"]
            category = df.iloc[i]["menu_category"]

            if (
                name in reduced_likes
                or name in top_10_recommendations
            ):
                continue
            if has_allergy(df.iloc[i].get("matched_allergies", "")):
                continue
            if matches_food_preference(category):
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
    
# @app.get("/api/evaluate/loo3/{username}")
# async def leave_one_out_evaluation_loo3(username: str):
#     threshold = 0.5  # Define similarity threshold
#     log_memory_usage("🔄 Start of LOO3 Evaluation")

#     # Fetch user profile from DB
#     user = await user_profile_collection.find_one({"user_name": username})
#     if not user:
#         raise HTTPException(status_code=404, detail="User not found")

#     liked_menus = user.get("liked_menu", [])
#     disliked_menus = user.get("disliked_menu", [])
#     allergies = set(user.get("allergies", []))
#     food_preferences = set(user.get("food_preferences", []))
#     cold_start_menus = [menu["menu_name"] for menu in user.get("cold_start", [])]

#     # ✅ Step 1: Remove Cold Start Menus from liked and disliked lists
#     liked_menus = [menu for menu in liked_menus if menu not in cold_start_menus]
#     disliked_menus = [menu for menu in disliked_menus if menu not in cold_start_menus]

#     # 🎯 Step 2: Prepare menus for evaluation
#     post_cold_start_liked_menus = liked_menus
#     post_cold_start_disliked_menus = disliked_menus

#     hits_liked = 0
#     hits_disliked = 0
#     dislike_hits = 0
#     total_tests_liked = 0
#     total_tests_disliked = 0
#     y_true = []
#     y_pred = []
#     all_results = []

#     # 🎯 Step 3: Iterate through both liked and disliked menus for evaluation
#     for target_menu in post_cold_start_liked_menus + post_cold_start_disliked_menus:
#         # Define ground truth label
#         actual_label = "like" if target_menu in post_cold_start_liked_menus else "dislike"

#         # Create a new list excluding the target menu
#         reduced_likes = [
#             menu for menu in post_cold_start_liked_menus if menu != target_menu
#         ]

#         # Generate user vector based on reduced likes (after cold start removed)
#         liked_indices = [
#             i for i, name in enumerate(df["menu_name"]) if name in reduced_likes
#         ]
#         liked_vectors = (
#             menu_embeddings[liked_indices]
#             if liked_indices
#             else np.zeros_like(menu_embeddings[0])
#         )

#         disliked_indices = [
#             i
#             for i, name in enumerate(df["menu_name"])
#             if name in post_cold_start_disliked_menus
#         ]
#         disliked_vectors = (
#             menu_embeddings[disliked_indices]
#             if disliked_indices
#             else np.zeros_like(menu_embeddings[0])
#         )

#         # ✅ Step 4: Calculate user vector with penalization
#         user_vector = np.zeros_like(menu_embeddings[0])
#         if len(liked_vectors) > 0:
#             user_vector += like_weight * np.mean(liked_vectors, axis=0)
#         if len(disliked_vectors) > 0:
#             user_vector -= dislike_weight * np.mean(disliked_vectors, axis=0)

#         user_vector = user_vector.reshape(1, -1)

#         # ✅ Step 5: Calculate cosine similarity
#         scores = cosine_similarity(user_vector, menu_embeddings).flatten()
#         sorted_indices = scores.argsort()[::-1]

#         # ✅ Step 6: Generate Top 10 Recommendations (No Diversity)
#         recommended_menus = []

#         def has_allergy(menu_allergens):
#             menu_allergy_set = set(str(menu_allergens).split(", "))
#             return not allergies.isdisjoint(menu_allergy_set)

#         def matches_food_preference(category_string):
#             category_list = set(str(category_string).split(","))
#             return not food_preferences.isdisjoint(category_list)

#         # Generate Top 10 matching food preferences
#         for i in sorted_indices:
#             name = df.iloc[i]["menu_name"]
#             category = df.iloc[i]["menu_category"]
#             if name in reduced_likes:
#                 continue
#             if has_allergy(df.iloc[i].get("matched_allergies", "")):
#                 continue
#             if not matches_food_preference(category):
#                 continue
#             if len(recommended_menus) < 10:
#                 recommended_menus.append(name)
#             else:
#                 break

#         # ✅ Step 7: Check Hits for Liked and Disliked Menus
#         is_hit_top10 = target_menu in recommended_menus

#         if target_menu in post_cold_start_liked_menus:
#             hits_liked += int(is_hit_top10)
#             total_tests_liked += 1
#         else:
#             hits_disliked += int(is_hit_top10)
#             total_tests_disliked += 1

#         # ✅ Step 8: Check if any disliked menu is in recommendations (False Positives)
#         disliked_hits_count = sum(
#             1 for menu in post_cold_start_disliked_menus if menu in recommended_menus
#         )
#         dislike_hits += disliked_hits_count

#         # ✅ Step 9: Calculate similarity of target menu
#         target_index = df[df["menu_name"] == target_menu].index
#         target_similarity = (
#             scores[target_index[0]] if len(target_index) > 0 else 0.0
#         )

#         # ✅ Step 10: Classify based on threshold
#         predicted_label = "like" if target_similarity >= threshold else "dislike"

#         # ✅ Step 11: Track results for analysis
#         y_true.append(actual_label)
#         y_pred.append(predicted_label)

#         all_results.append(
#             {
#                 "target_menu": target_menu,
#                 "recommended_menus": recommended_menus,
#                 "hit_top10": is_hit_top10,
#                 "disliked_hit_count": int(disliked_hits_count),
#                 "predicted_label": predicted_label,
#                 "actual_label": actual_label,
#                 "similarity_score": round(float(target_similarity), 4),
#             }
#         )

#     # ✅ Step 12: Calculate classification metrics
#     accuracy = accuracy_score(y_true, y_pred)
#     precision = precision_score(y_true, y_pred, pos_label="like")
#     recall = recall_score(y_true, y_pred, pos_label="like")
#     f1 = f1_score(y_true, y_pred, pos_label="like")

#     # ✅ Step 13: Calculate FPR (False Positive Rate) for disliked menus
#     fpr = (
#         dislike_hits / (len(post_cold_start_disliked_menus) * (total_tests_liked + total_tests_disliked))
#         if len(post_cold_start_disliked_menus) > 0
#         else 0
#     )

#     # ✅ Step 14: Calculate hit rate for Top 10
#     hit_rate_liked_top10 = hits_liked / total_tests_liked if total_tests_liked > 0 else 0
#     hit_rate_disliked_top10 = hits_disliked / total_tests_disliked if total_tests_disliked > 0 else 0
#     overall_hit_rate = (hits_liked + hits_disliked) / (total_tests_liked + total_tests_disliked)

#     log_memory_usage("✅ LOO3 Evaluation completed (No Diverse)")

#     return {
#         "hit_rate_liked_top10": round(float(hit_rate_liked_top10), 2),
#         "hit_rate_disliked_top10": round(float(hit_rate_disliked_top10), 2),
#         "overall_hit_rate": round(float(overall_hit_rate), 2),
#         "precision": round(float(precision), 2),
#         "recall": round(float(recall), 2),
#         "f1_score": round(float(f1), 2),
#         "fpr": round(float(fpr), 2),
#         "total_tests_liked": int(total_tests_liked),
#         "total_tests_disliked": int(total_tests_disliked),
#         "hits_liked_top10": int(hits_liked),
#         "hits_disliked_top10": int(hits_disliked),
#         "disliked_hits": int(dislike_hits),
#         "detailed_results": all_results,
#     }


# @app.get("/api/evaluate/loo3_multiple_users")
# async def leave_one_out_evaluation_loo3_multiple_users():
#     threshold = 0.5  # Define similarity threshold
#     log_memory_usage("🔄 Start of Multi-User LOO3 Evaluation")

#     # List of target users
#     user_list = ["guest2085265849", "guest0338470649", "guest6685179857"]

#     # Aggregate per-user metrics
#     user_metrics = []

#     all_results = []  # Store detailed results for all users

#     # 🎯 Loop through all users in user_list
#     for username in user_list:
#         log_memory_usage(f"🔍 Evaluating user: {username}")

#         # Fetch user profile from DB
#         user = await user_profile_collection.find_one({"user_name": username})
#         if not user:
#             log_memory_usage(f"⚠️ User {username} not found, skipping...")
#             continue

#         liked_menus = user.get("liked_menu", [])
#         disliked_menus = user.get("disliked_menu", [])
#         allergies = set(user.get("allergies", []))
#         food_preferences = set(user.get("food_preferences", []))
#         cold_start_menus = [menu["menu_name"] for menu in user.get("cold_start", [])]

#         # ✅ Step 1: Remove Cold Start Menus from liked and disliked lists
#         liked_menus = [menu for menu in liked_menus if menu not in cold_start_menus]
#         disliked_menus = [menu for menu in disliked_menus if menu not in cold_start_menus]

#         # 🎯 Step 2: Prepare menus for evaluation
#         post_cold_start_liked_menus = liked_menus
#         post_cold_start_disliked_menus = disliked_menus

#         hits_liked = 0
#         hits_disliked = 0
#         dislike_hits = 0
#         total_tests_liked = 0
#         total_tests_disliked = 0

#         y_true_user = []
#         y_pred_user = []

#         # 🎯 Step 3: Iterate through both liked and disliked menus for evaluation
#         for target_menu in post_cold_start_liked_menus + post_cold_start_disliked_menus:
#             # Define ground truth label
#             actual_label = "like" if target_menu in post_cold_start_liked_menus else "dislike"

#             # Create a new list excluding the target menu
#             reduced_likes = [
#                 menu for menu in post_cold_start_liked_menus if menu != target_menu
#             ]

#             # Generate user vector based on reduced likes
#             liked_indices = [
#                 i for i, name in enumerate(df["menu_name"]) if name in reduced_likes
#             ]
#             liked_vectors = (
#                 menu_embeddings[liked_indices]
#                 if liked_indices
#                 else np.zeros_like(menu_embeddings[0])
#             )

#             disliked_indices = [
#                 i
#                 for i, name in enumerate(df["menu_name"])
#                 if name in post_cold_start_disliked_menus
#             ]
#             disliked_vectors = (
#                 menu_embeddings[disliked_indices]
#                 if disliked_indices
#                 else np.zeros_like(menu_embeddings[0])
#             )

#             # ✅ Step 4: Calculate user vector with penalization
#             user_vector = np.zeros_like(menu_embeddings[0])
#             if len(liked_vectors) > 0:
#                 user_vector += like_weight * np.mean(liked_vectors, axis=0)
#             if len(disliked_vectors) > 0:
#                 user_vector -= dislike_weight * np.mean(disliked_vectors, axis=0)

#             user_vector = user_vector.reshape(1, -1)

#             # ✅ Step 5: Calculate cosine similarity
#             scores = cosine_similarity(user_vector, menu_embeddings).flatten()
#             sorted_indices = scores.argsort()[::-1]

#             # ✅ Step 6: Generate Top 10 Recommendations (No Diversity)
#             recommended_menus = []
#             def has_allergy(menu_allergens):
#                 menu_allergy_set = set(str(menu_allergens).split(", "))
#                 return not allergies.isdisjoint(menu_allergy_set)

#             def matches_food_preference(category_string):
#                 category_list = set(str(category_string).split(","))
#                 return not food_preferences.isdisjoint(category_list)

#             # Generate Top 10 matching food preferences
#             for i in sorted_indices:
#                 name = df.iloc[i]["menu_name"]
#                 category = df.iloc[i]["menu_category"]
#                 if name in reduced_likes:
#                     continue
#                 if has_allergy(df.iloc[i].get("matched_allergies", "")):
#                     continue
#                 if not matches_food_preference(category):
#                     continue
#                 if len(recommended_menus) < 10:
#                     recommended_menus.append(name)
#                 else:
#                     break

#             # ✅ Step 7: Check Hits for Liked and Disliked Menus
#             is_hit_top10 = target_menu in recommended_menus

#             if target_menu in post_cold_start_liked_menus:
#                 hits_liked += int(is_hit_top10)
#                 total_tests_liked += 1
#             else:
#                 hits_disliked += int(is_hit_top10)
#                 total_tests_disliked += 1

#             # ✅ Step 8: Check if any disliked menu is in recommendations (False Positives)
#             disliked_hits_count = sum(
#                 1 for menu in post_cold_start_disliked_menus if menu in recommended_menus
#             )
#             dislike_hits += disliked_hits_count

#             # ✅ Step 9: Calculate similarity of target menu
#             target_index = df[df["menu_name"] == target_menu].index
#             target_similarity = (
#                 scores[target_index[0]] if len(target_index) > 0 else 0.0
#             )

#             # ✅ Step 10: Classify based on threshold
#             predicted_label = "like" if target_similarity >= threshold else "dislike"

#             # ✅ Step 11: Track results for analysis
#             y_true_user.append(actual_label)
#             y_pred_user.append(predicted_label)

#             all_results.append(
#                 {
#                     "user": username,
#                     "target_menu": target_menu,
#                     "recommended_menus": recommended_menus,
#                     "hit_top10": is_hit_top10,
#                     "disliked_hit_count": int(disliked_hits_count),
#                     "predicted_label": predicted_label,
#                     "actual_label": actual_label,
#                     "similarity_score": round(float(target_similarity), 4),
#                 }
#             )

#         # ✅ Step 12: Calculate per-user classification metrics
#         user_precision = precision_score(y_true_user, y_pred_user, pos_label="like", zero_division=0)
#         user_recall = recall_score(y_true_user, y_pred_user, pos_label="like", zero_division=0)
#         user_f1 = f1_score(y_true_user, y_pred_user, pos_label="like", zero_division=0)

#         user_hit_rate_liked_top10 = hits_liked / total_tests_liked if total_tests_liked > 0 else 0
#         user_hit_rate_disliked_top10 = hits_disliked / total_tests_disliked if total_tests_disliked > 0 else 0
#         user_fpr = (
#             dislike_hits / (len(post_cold_start_disliked_menus) * (total_tests_liked + total_tests_disliked))
#             if len(post_cold_start_disliked_menus) > 0
#             else 0
#         )
#         user_overall_hit_rate = (hits_liked + hits_disliked) / (total_tests_liked + total_tests_disliked)
#         print(user_metrics)
#         # ✅ Step 13: Store per-user metrics
#         user_metrics.append({
#             "hit_rate_liked_top10": user_hit_rate_liked_top10,
#             "hit_rate_disliked_top10": user_hit_rate_disliked_top10,
#             "overall_hit_rate": user_overall_hit_rate,
#             "precision": user_precision,
#             "recall": user_recall,
#             "f1_score": user_f1,
#             "fpr": user_fpr,
#         })

#     # ✅ Step 14: Calculate Average Across All Users
#     avg_metrics = {
#         "hit_rate_liked_top10": np.mean([m["hit_rate_liked_top10"] for m in user_metrics]),
#         "hit_rate_disliked_top10": np.mean([m["hit_rate_disliked_top10"] for m in user_metrics]),
#         "overall_hit_rate": np.mean([m["overall_hit_rate"] for m in user_metrics]),
#         "precision": np.mean([m["precision"] for m in user_metrics]),
#         "recall": np.mean([m["recall"] for m in user_metrics]),
#         "f1_score": np.mean([m["f1_score"] for m in user_metrics]),
#         "fpr": np.mean([m["fpr"] for m in user_metrics]),
#     }

#     log_memory_usage("✅ Multi-User LOO3 Evaluation completed (Averaged Per-User Metrics)")

#     return {
#         "avg_hit_rate_liked_top10": round(float(avg_metrics["hit_rate_liked_top10"]), 2),
#         "avg_hit_rate_disliked_top10": round(float(avg_metrics["hit_rate_disliked_top10"]), 2),
#         "avg_overall_hit_rate": round(float(avg_metrics["overall_hit_rate"]), 2),
#         "avg_precision": round(float(avg_metrics["precision"]), 2),
#         "avg_recall": round(float(avg_metrics["recall"]), 2),
#         "avg_f1_score": round(float(avg_metrics["f1_score"]), 2),
#         "avg_fpr": round(float(avg_metrics["fpr"]), 2),
#         "detailed_results": all_results,
#     }

@app.get("/api/evaluate/loo2")
async def leave_one_out_evaluation_for_users(top_n: int = 20, threshold: float = 0.5):
    # Define the list of usernames directly as a variable
    user_list = ["savetang", "Tanabodee"]  # 🎉 Define your list here!

    log_memory_usage("🔄 Start of LOO Evaluation for Specific Users")

    # Fetch user profiles for the given list of usernames
    users = await user_profile_collection.find({"user_name": {"$in": user_list}}).to_list(length=1000)
    all_results = []
    total_hits = 0
    total_dislike_hits = 0
    total_tests = 0
    y_true = []
    y_pred = []

    # Loop through each user for LOO evaluation
    for user in users:
        liked_menus = user.get("liked_menu", [])
        disliked_menus = user.get("disliked_menu", [])
        allergies = set(user.get("allergies", []))

        # Check if the user has enough liked menus to perform LOO
        if len(liked_menus) < 3:
            continue  # Skip user if not enough liked menus for evaluation

        hit_count = 0
        dislike_hits = 0

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
            
            def is_safe_menu(menu_allergens):
                menu_allergy_set = set(str(menu_allergens).split(", "))
                return allergies.isdisjoint(menu_allergy_set)
            
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
            all_results.append(
                {
                    "user": user["user_name"],
                    "target_menu": target_menu,
                    "recommended_menus": recommended_menus,
                    "hit": is_hit,
                    "disliked_hit_count": int(disliked_hits_count),  # Convert to int
                    "predicted_label": predicted_label,
                    "actual_label": actual_label,
                    "similarity_score": round(float(target_similarity), 4),
                }
            )

        total_hits += hit_count
        total_dislike_hits += dislike_hits

    # Calculate classification metrics for all users
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, pos_label="like")
    recall = recall_score(y_true, y_pred, pos_label="like")
    f1 = f1_score(y_true, y_pred, pos_label="like")

    # Calculate FPR (False Positive Rate) for disliked menus
    fpr = (
        total_dislike_hits / (len(disliked_menus) * total_tests)
        if len(disliked_menus) > 0
        else 0
    )

    # Calculate hit rate
    hit_rate = total_hits / total_tests if total_tests > 0 else 0

    log_memory_usage("✅ LOO Evaluation completed for all users")

    return {
        "hit_rate": round(float(hit_rate), 2),  # ✅ Convert to float
        "precision": round(float(precision), 2),
        "recall": round(float(recall), 2),
        "f1_score": round(float(f1), 2),
        "fpr": round(float(fpr), 2),
        "total_tests": int(total_tests),  # ✅ Convert to int
        "hits": int(total_hits),
        "disliked_hits": int(total_dislike_hits),
        "detailed_results": all_results,
    }



