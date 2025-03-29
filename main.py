from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
from db import user_profile_collection
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



@app.get("/api/evaluate/loo4/{username}")
async def leave_one_out_evaluation_loo4(username: str):
    threshold = 0.5  # Define similarity threshold
    log_memory_usage("🔄 Start of LOO3 Evaluation")

    # Fetch user profile from DB
    user = await user_profile_collection.find_one({"user_name": username})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    liked_menus = user.get("liked_menu", [])
    disliked_menus = user.get("disliked_menu", [])
    allergies = set(user.get("allergies", []))
    food_preferences = set(user.get("food_preferences", []))

    log_memory_usage("📋 Retrieved liked/disliked menus from DB")

    hits_liked = 0
    hits_disliked = 0
    dislike_hits = 0
    total_tests_liked = 0
    total_tests_disliked = 0
    y_true = []
    y_pred = []
    all_results = []

    # 🎯 Step 1: Iterate through both liked and disliked menus for evaluation
    for target_menu in liked_menus + disliked_menus:
        # Define ground truth label
        actual_label = "like" if target_menu in liked_menus else "dislike"

        # Create a new list excluding the target menu
        reduced_likes = [menu for menu in liked_menus if menu != target_menu]

        # ✅ Step 2: Generate user vector based on reduced likes
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
            if name in disliked_menus
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

        def has_allergy(menu_allergens):
            menu_allergy_set = set(str(menu_allergens).split(", "))
            return not allergies.isdisjoint(menu_allergy_set)

        def matches_food_preference(category_string):
            category_list = set(str(category_string).split(","))
            return not food_preferences.isdisjoint(category_list)

        # ✅ Step 5: Generate Top 10 + Diverse 5 Recommendations
        recommendations = []

        # Top 10 from food_preferences only
        for i in sorted_indices:
            name = df.iloc[i]["menu_name"]
            category = df.iloc[i]["menu_category"]
            if name in reduced_likes or name in disliked_menus:
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

        similarity_threshold_2 = (
            recommendations[-1][-1] - 0.03 if len(recommendations) >= 10 else 1.0
        )

        # Final 5: diverse menus not in food_preferences and lower score
        for i in sorted_indices:
            name = df.iloc[i]["menu_name"]
            category = df.iloc[i]["menu_category"]
            score = scores[i]
            if name in reduced_likes or name in disliked_menus:
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

        recommended_menus = [r[1] for r in recommendations]

        # ✅ Step 6: Check Hits for Liked and Disliked Menus
        is_hit_top10 = target_menu in recommended_menus

        if target_menu in liked_menus:
            hits_liked += int(is_hit_top10)
            total_tests_liked += 1
        elif target_menu in disliked_menus:
            hits_disliked += int(is_hit_top10)
            total_tests_disliked += 1

        # ✅ Step 7: Check if any disliked menu is in recommendations (False Positives)
        if target_menu in disliked_menus and target_menu in recommended_menus:
            dislike_hits += 1

        # ✅ Step 8: Calculate similarity of target menu
        target_index = df[df["menu_name"] == target_menu].index
        target_similarity = (
            scores[target_index[0]] if len(target_index) > 0 else 0.0
        )

        # ✅ Step 9: Classify based on threshold
        predicted_label = "like" if target_similarity >= threshold else "dislike"

        # ✅ Step 10: Track results for analysis
        y_true.append(actual_label)
        y_pred.append(predicted_label)

        all_results.append(
            {
                "target_menu": target_menu,
                "recommended_menus": recommended_menus,
                "hit_top10": is_hit_top10,
                "disliked_hit": int(target_menu in disliked_menus and target_menu in recommended_menus),
                "predicted_label": predicted_label,
                "actual_label": actual_label,
                "similarity_score": round(float(target_similarity), 4),
            }
        )

    # ✅ Step 11: Calculate classification metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, pos_label="like")
    recall = recall_score(y_true, y_pred, pos_label="like")
    f1 = f1_score(y_true, y_pred, pos_label="like")

    # ✅ Step 12: Calculate FPR (False Positive Rate) for disliked menus
    fpr = (
        dislike_hits / (len(disliked_menus) * (total_tests_liked + total_tests_disliked))
        if len(disliked_menus) > 0
        else 0
    )

    # ✅ Step 13: Calculate hit rate for Top 10
    hit_rate_liked_top10 = hits_liked / total_tests_liked if total_tests_liked > 0 else 0
    hit_rate_disliked_top10 = hits_disliked / total_tests_disliked if total_tests_disliked > 0 else 0
    overall_hit_rate = (hits_liked + hits_disliked) / (total_tests_liked + total_tests_disliked)

    log_memory_usage("✅ LOO3 Evaluation completed (With Disliked Check)")

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

@app.get("/api/evaluate/loo4_multiple_users")
async def leave_one_out_evaluation_loo4_multiple_users():
    threshold = 0.5  # Define similarity threshold
    log_memory_usage("🔄 Start of Multi-User LOO4 Evaluation")

    # List of users for evaluation
    user_list = ["savetang", "Tanabodee", "mhuyong", "junior1", "tunior", "bubu1234", "dudu1234", "thanidaza", "punn", "tak", "luck", "wik", "hamm", "prai", "will"]

    # Initialize aggregate variables
    total_hits_liked = 0
    total_hits_disliked = 0
    total_dislike_hits = 0
    total_tests_liked = 0
    total_tests_disliked = 0
    all_results = []
    
    y_true_global = []
    y_pred_global = []

    # 🎯 Loop through all users in user_list
    for username in user_list:
        log_memory_usage(f"🔍 Evaluating user: {username}")
        # Fetch user profile from DB
        user = await user_profile_collection.find_one({"user_name": username})
        if not user:
            log_memory_usage(f"⚠️ User {username} not found, skipping...")
            continue

        liked_menus = user.get("liked_menu", [])
        disliked_menus = user.get("disliked_menu", [])
        allergies = set(user.get("allergies", []))
        food_preferences = set(user.get("food_preferences", []))

        hits_liked = 0
        hits_disliked = 0
        dislike_hits = 0
        user_tests_liked = 0
        user_tests_disliked = 0

        y_true_user = []
        y_pred_user = []

        # 🎯 Step 1: Iterate through both liked and disliked menus for evaluation
        for target_menu in liked_menus + disliked_menus:
            # Define ground truth label
            actual_label = "like" if target_menu in liked_menus else "dislike"

            # Create a new list excluding the target menu
            reduced_likes = [menu for menu in liked_menus if menu != target_menu]

            # ✅ Step 2: Generate user vector based on reduced likes
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
                if name in disliked_menus
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

            def has_allergy(menu_allergens):
                menu_allergy_set = set(str(menu_allergens).split(", "))
                return not allergies.isdisjoint(menu_allergy_set)

            def matches_food_preference(category_string):
                category_list = set(str(category_string).split(","))
                return not food_preferences.isdisjoint(category_list)

            # ✅ Step 5: Generate Top 10 + Diverse 5 Recommendations
            recommendations = []

            # Top 10 from food_preferences only
            for i in sorted_indices:
                name = df.iloc[i]["menu_name"]
                category = df.iloc[i]["menu_category"]
                if name in reduced_likes or name in disliked_menus:
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
                    scores[i],
                ))
                if len(recommendations) >= 10:
                    break

            similarity_threshold_2 = (
                recommendations[-1][-1] - 0.03 if len(recommendations) >= 10 else 1.0
            )

            # Final 5: diverse menus not in food_preferences and lower score
            for i in sorted_indices:
                name = df.iloc[i]["menu_name"]
                category = df.iloc[i]["menu_category"]
                score = scores[i]
                if name in reduced_likes or name in disliked_menus:
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
                    score,
                ))
                if len(recommendations) >= 15:
                    break

            recommended_menus = [r[1] for r in recommendations]

            # ✅ Step 6: Check Hits for Liked and Disliked Menus
            is_hit_top10 = target_menu in recommended_menus

            if target_menu in liked_menus:
                hits_liked += int(is_hit_top10)
                user_tests_liked += 1
            elif target_menu in disliked_menus:
                hits_disliked += int(is_hit_top10)
                user_tests_disliked += 1

            # ✅ Step 7: Check if any disliked menu is in recommendations (False Positives)
            if target_menu in disliked_menus and target_menu in recommended_menus:
                dislike_hits += 1

            # ✅ Step 8: Calculate similarity of target menu
            target_index = df[df["menu_name"] == target_menu].index
            target_similarity = (
                scores[target_index[0]] if len(target_index) > 0 else 0.0
            )

            # ✅ Step 9: Classify based on threshold
            predicted_label = "like" if target_similarity >= threshold else "dislike"

            # ✅ Step 10: Track results for analysis
            y_true_user.append(actual_label)
            y_pred_user.append(predicted_label)

            all_results.append(
                {
                    "user": username,
                    "target_menu": target_menu,
                    "recommended_menus": recommended_menus,
                    "hit_top10": is_hit_top10,
                    "disliked_hit": int(target_menu in disliked_menus and target_menu in recommended_menus),
                    "predicted_label": predicted_label,
                    "actual_label": actual_label,
                    "similarity_score": round(float(target_similarity), 4),
                }
            )

        # ✅ Step 11: Aggregate results for this user
        total_hits_liked += hits_liked
        total_hits_disliked += hits_disliked
        total_dislike_hits += dislike_hits
        total_tests_liked += user_tests_liked
        total_tests_disliked += user_tests_disliked
        y_true_global.extend(y_true_user)
        y_pred_global.extend(y_pred_user)

    # ✅ Step 12: Calculate overall classification metrics
    accuracy = accuracy_score(y_true_global, y_pred_global)
    precision = precision_score(y_true_global, y_pred_global, pos_label="like")
    recall = recall_score(y_true_global, y_pred_global, pos_label="like")
    f1 = f1_score(y_true_global, y_pred_global, pos_label="like")

    # ✅ Step 13: Calculate FPR (False Positive Rate) for disliked menus
    fpr = (
        total_dislike_hits / (len(disliked_menus) * (total_tests_liked + total_tests_disliked))
        if len(disliked_menus) > 0
        else 0
    )

    # ✅ Step 14: Calculate hit rate for Top 10
    hit_rate_liked_top10 = total_hits_liked / total_tests_liked if total_tests_liked > 0 else 0
    hit_rate_disliked_top10 = total_hits_disliked / total_tests_disliked if total_tests_disliked > 0 else 0
    overall_hit_rate = (total_hits_liked + total_hits_disliked) / (total_tests_liked + total_tests_disliked)

    log_memory_usage("✅ Multi-User LOO4 Evaluation completed (With Disliked Check)")

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
        "hits_liked_top10": int(total_hits_liked),
        "hits_disliked_top10": int(total_hits_disliked),
        "disliked_hits": int(total_dislike_hits),
        "detailed_results": all_results,
    }



