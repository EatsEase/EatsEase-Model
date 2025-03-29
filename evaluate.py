# @app.get("/api/evaluate/loo2/{username}")
# async def leave_one_out_evaluation2(username: str, top_n: int = 20):
#     threshold = 0.5
#     log_memory_usage("🔄 Start of LOO Evaluation")

#     # Fetch user profile from DB
#     user = await user_profile_collection.find_one({"user_name": username})
#     if not user:
#         raise HTTPException(status_code=404, detail="User not found")

#     liked_menus = user.get("liked_menu", [])
#     disliked_menus = user.get("disliked_menu", [])
#     allergies = set(user.get("allergies", []))  # ✅ Already converted to set


#     # Check if the user has enough liked menus to perform LOO
#     if len(liked_menus) < 3:
#         raise HTTPException(
#             status_code=400, detail="Not enough liked menus for evaluation"
#         )

#     hit_count = 0
#     dislike_hits = 0
#     total_tests = 0
#     y_true = []
#     y_pred = []
#     all_results = []
    
#     def is_safe_menu(menu_allergens):
#         menu_allergy_set = set(str(menu_allergens).split(", "))
#         return allergies.isdisjoint(menu_allergy_set)

#     # Iterate through each liked menu (Leave-One-Out)
#     for target_menu in liked_menus:
#         # Create a new list excluding the target menu
#         reduced_likes = [menu for menu in liked_menus if menu != target_menu]

#         # Generate user vector based on reduced likes
#         liked_indices = [
#             i for i, name in enumerate(df["menu_name"]) if name in reduced_likes
#         ]
#         liked_vectors = (
#             menu_embeddings[liked_indices]
#             if liked_indices
#             else np.zeros_like(menu_embeddings[0])
#         )

#         disliked_indices = [
#             i for i, name in enumerate(df["menu_name"]) if name in disliked_menus
#         ]
#         disliked_vectors = (
#             menu_embeddings[disliked_indices]
#             if disliked_indices
#             else np.zeros_like(menu_embeddings[0])
#         )

#         # Calculate user vector with weighted penalization
#         user_vector = np.zeros_like(menu_embeddings[0])
#         if len(liked_vectors) > 0:
#             user_vector += like_weight * np.mean(liked_vectors, axis=0)
#         if len(disliked_vectors) > 0:
#             user_vector -= dislike_weight * np.mean(disliked_vectors, axis=0)

#         user_vector = user_vector.reshape(1, -1)

#         # Calculate cosine similarity
#         scores = cosine_similarity(user_vector, menu_embeddings).flatten()
#         sorted_indices = scores.argsort()[::-1][:top_n]
#         recommended_menus = df.iloc[sorted_indices]["menu_name"].tolist()
#         for i in sorted_indices:
#             menu_name = df.iloc[i]["menu_name"]
#             if is_safe_menu(df.iloc[i]["matched_allergies"]):
#                 recommended_menus.append(menu_name)
#             if len(recommended_menus) >= top_n:
#                 break

#         # Check if target_menu is in top-N recommendations
#         is_hit = target_menu in recommended_menus
#         hit_count += int(is_hit)
#         total_tests += 1

#         # Check if any disliked menu is in top-N recommendations
#         disliked_hits_count = sum(
#             1 for menu in disliked_menus if menu in recommended_menus
#         )
#         dislike_hits += disliked_hits_count

#         # Calculate similarity of target menu
#         target_index = df[df["menu_name"] == target_menu].index
#         target_similarity = (
#             scores[target_index[0]] if len(target_index) > 0 else 0.0
#         )

#         # Classify based on threshold
#         predicted_label = "like" if target_similarity >= threshold else "dislike"
#         actual_label = "like"  # Ground truth since target_menu is from liked_menus

#         # Append to evaluation lists
#         y_true.append(actual_label)
#         y_pred.append(predicted_label)

#         # Track results for analysis
#         # Correct this section where you append the results
#         all_results.append(
#             {
#                 "target_menu": target_menu,
#                 "recommended_menus": recommended_menus,
#                 "hit": is_hit,
#                 "disliked_hit_count": int(disliked_hits_count),  # Convert to int
#                 "predicted_label": predicted_label,
#                 "actual_label": actual_label,
#                 "similarity_score": round(float(target_similarity), 4),  # ✅ Fix here
#             }
#         )


#     # Calculate classification metrics
#     accuracy = accuracy_score(y_true, y_pred)
#     precision = precision_score(y_true, y_pred, pos_label="like")
#     recall = recall_score(y_true, y_pred, pos_label="like")
#     f1 = f1_score(y_true, y_pred, pos_label="like")

#     # Calculate FPR (False Positive Rate) for disliked menus
#     fpr = (
#         dislike_hits / (len(disliked_menus) * total_tests)
#         if len(disliked_menus) > 0
#         else 0
#     )

#     # Calculate hit rate
#     hit_rate = hit_count / total_tests if total_tests > 0 else 0

#     log_memory_usage("✅ LOO Evaluation completed")

#     return {
#         "hit_rate": round(float(hit_rate), 2),  # ✅ Convert to float
#         "precision": round(float(precision), 2),
#         "recall": round(float(recall), 2),
#         "f1_score": round(float(f1), 2),
#         "fpr": round(float(fpr), 2),
#         "total_tests": int(total_tests),  # ✅ Convert to int
#         "hits": int(hit_count),
#         "disliked_hits": int(dislike_hits),
#         "detailed_results": all_results
#     }
    
# @app.get("/api/evaluate/loo/{username}")
# async def leave_one_out_evaluation(username: str, top_n: int = 10):
#     threshold = 0.5  # Define similarity threshold
#     log_memory_usage("🔄 Start of LOO Evaluation")

#     # Fetch user profile from DB
#     user = await user_profile_collection.find_one({"user_name": username})
#     if not user:
#         raise HTTPException(status_code=404, detail="User not found")

#     liked_menus = user.get("liked_menu", [])
#     disliked_menus = user.get("disliked_menu", [])
#     allergies = set(user.get("allergies", []))
#     food_preferences = set(user.get("food_preferences", []))
#     cold_start_menus = [menu["menu_name"] for menu in user.get("cold_start", [])]
    
#         # ✅ Step 1: Remove Cold Start Menus from liked and disliked lists
#     liked_menus = [menu for menu in liked_menus if menu not in cold_start_menus]
#     disliked_menus = [menu for menu in disliked_menus if menu not in cold_start_menus]

#     # 🎯 Step 1: Remove the first liked and first 2 disliked menus (Cold Start Removal)
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

#     # 🎯 Step 2: Iterate through both liked and disliked menus for evaluation
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

#         # ✅ Step 3: Calculate user vector with penalization
#         user_vector = np.zeros_like(menu_embeddings[0])
#         if len(liked_vectors) > 0:
#             user_vector += like_weight * np.mean(liked_vectors, axis=0)
#         if len(disliked_vectors) > 0:
#             user_vector -= dislike_weight * np.mean(disliked_vectors, axis=0)

#         user_vector = user_vector.reshape(1, -1)

#         # ✅ Step 4: Calculate cosine similarity
#         scores = cosine_similarity(user_vector, menu_embeddings).flatten()
#         sorted_indices = scores.argsort()[::-1]

#         # ✅ Step 5: Generate Top 10 + Diverse 5 Recommendations
#         top_10_recommendations = []
#         diverse_5_recommendations = []

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
#             if len(top_10_recommendations) < 10:
#                 top_10_recommendations.append(name)
#             else:
#                 break

#         # Generate Final 5 diverse recommendations
#         for i in sorted_indices:
#             name = df.iloc[i]["menu_name"]
#             category = df.iloc[i]["menu_category"]

#             if (
#                 name in reduced_likes
#                 or name in top_10_recommendations
#             ):
#                 continue
#             if has_allergy(df.iloc[i].get("matched_allergies", "")):
#                 continue
#             if matches_food_preference(category):
#                 continue
#             if len(diverse_5_recommendations) < 5:
#                 diverse_5_recommendations.append(name)
#             else:
#                 break

#         # ✅ Step 6: Merge top 10 and diverse 5 for final evaluation
#         recommended_menus = top_10_recommendations + diverse_5_recommendations

#         # ✅ Step 7: Check Hits for Liked and Disliked Menus
#         is_hit_top10 = target_menu in top_10_recommendations
#         is_hit_diverse5 = target_menu in diverse_5_recommendations

#         if target_menu in post_cold_start_liked_menus:
#             hits_liked += int(is_hit_top10 or is_hit_diverse5)
#             total_tests_liked += 1
#         else:
#             hits_disliked += int(is_hit_top10 or is_hit_diverse5)
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
#                 "hit_diverse5": is_hit_diverse5,
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

#     # ✅ Step 14: Calculate hit rate for Top 10 and Diverse 5
#     hit_rate_liked_top10 = hits_liked / total_tests_liked if total_tests_liked > 0 else 0
#     hit_rate_disliked_top10 = hits_disliked / total_tests_disliked if total_tests_disliked > 0 else 0
#     overall_hit_rate = (hits_liked + hits_disliked) / (total_tests_liked + total_tests_disliked)

#     log_memory_usage("✅ LOO Evaluation completed (Post-Cold Start)")

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

# @app.get("/api/evaluate/loo2")
# async def leave_one_out_evaluation_for_users(top_n: int = 20, threshold: float = 0.5):
#     # Define the list of usernames directly as a variable
#     user_list = ["savetang", "Tanabodee", "mhuyong", "junior1", "banana", "tunior", "bubu1234", "dudu1234", "thanidaza"]  # 🎉 Define your list here!

#     log_memory_usage("🔄 Start of LOO Evaluation for Specific Users")

#     # Fetch user profiles for the given list of usernames
#     users = await user_profile_collection.find({"user_name": {"$in": user_list}}).to_list(length=1000)
#     all_results = []
#     total_hits = 0
#     total_dislike_hits = 0
#     total_tests = 0
#     y_true = []
#     y_pred = []

#     # Loop through each user for LOO evaluation
#     for user in users:
#         liked_menus = user.get("liked_menu", [])
#         disliked_menus = user.get("disliked_menu", [])
#         allergies = set(user.get("allergies", []))

#         # Check if the user has enough liked menus to perform LOO
#         if len(liked_menus) < 3:
#             continue  # Skip user if not enough liked menus for evaluation

#         hit_count = 0
#         dislike_hits = 0

#         # Iterate through each liked menu (Leave-One-Out)
#         for target_menu in liked_menus:
#             # Create a new list excluding the target menu
#             reduced_likes = [menu for menu in liked_menus if menu != target_menu]

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
#                 i for i, name in enumerate(df["menu_name"]) if name in disliked_menus
#             ]
#             disliked_vectors = (
#                 menu_embeddings[disliked_indices]
#                 if disliked_indices
#                 else np.zeros_like(menu_embeddings[0])
#             )

#             # Calculate user vector with weighted penalization
#             user_vector = np.zeros_like(menu_embeddings[0])
#             if len(liked_vectors) > 0:
#                 user_vector += like_weight * np.mean(liked_vectors, axis=0)
#             if len(disliked_vectors) > 0:
#                 user_vector -= dislike_weight * np.mean(disliked_vectors, axis=0)

#             user_vector = user_vector.reshape(1, -1)

#             # Calculate cosine similarity
#             scores = cosine_similarity(user_vector, menu_embeddings).flatten()
#             sorted_indices = scores.argsort()[::-1][:top_n]
#             recommended_menus = df.iloc[sorted_indices]["menu_name"].tolist()
            
#             def is_safe_menu(menu_allergens):
#                 menu_allergy_set = set(str(menu_allergens).split(", "))
#                 return allergies.isdisjoint(menu_allergy_set)
            
#             for i in sorted_indices:
#                 menu_name = df.iloc[i]["menu_name"]
#                 if menu_name in disliked_menus or menu_name in liked_menus:
#                     continue
#                 if is_safe_menu(df.iloc[i]["matched_allergies"]):
#                     recommended_menus.append(menu_name)
#                 if len(recommended_menus) >= top_n:
#                     break

#             # Check if target_menu is in top-N recommendations
#             is_hit = target_menu in recommended_menus
#             hit_count += int(is_hit)
#             total_tests += 1

#             # Check if any disliked menu is in top-N recommendations
#             disliked_hits_count = sum(
#                 1 for menu in disliked_menus if menu in recommended_menus
#             )
#             dislike_hits += disliked_hits_count

#             # Calculate similarity of target menu
#             target_index = df[df["menu_name"] == target_menu].index
#             target_similarity = (
#                 scores[target_index[0]] if len(target_index) > 0 else 0.0
#             )

#             # Classify based on threshold
#             predicted_label = "like" if target_similarity >= threshold else "dislike"
#             actual_label = "like"  # Ground truth since target_menu is from liked_menus

#             # Append to evaluation lists
#             y_true.append(actual_label)
#             y_pred.append(predicted_label)

#             # Track results for analysis
#             all_results.append(
#                 {
#                     "user": user["user_name"],
#                     "target_menu": target_menu,
#                     "recommended_menus": recommended_menus,
#                     "hit": is_hit,
#                     "disliked_hit_count": int(disliked_hits_count),  # Convert to int
#                     "predicted_label": predicted_label,
#                     "actual_label": actual_label,
#                     "similarity_score": round(float(target_similarity), 4),
#                 }
#             )

#         total_hits += hit_count
#         total_dislike_hits += dislike_hits

#     # Calculate classification metrics for all users
#     accuracy = accuracy_score(y_true, y_pred)
#     precision = precision_score(y_true, y_pred, pos_label="like")
#     recall = recall_score(y_true, y_pred, pos_label="like")
#     f1 = f1_score(y_true, y_pred, pos_label="like")

#     # Calculate FPR (False Positive Rate) for disliked menus
#     fpr = (
#         total_dislike_hits / (len(disliked_menus) * total_tests)
#         if len(disliked_menus) > 0
#         else 0
#     )

#     # Calculate hit rate
#     hit_rate = total_hits / total_tests if total_tests > 0 else 0

#     log_memory_usage("✅ LOO Evaluation completed for all users")

#     return {
#         "hit_rate": round(float(hit_rate), 2),  # ✅ Convert to float
#         "precision": round(float(precision), 2),
#         "recall": round(float(recall), 2),
#         "f1_score": round(float(f1), 2),
#         "fpr": round(float(fpr), 2),
#         "total_tests": int(total_tests),  # ✅ Convert to int
#         "hits": int(total_hits),
#         "disliked_hits": int(total_dislike_hits),
#         "detailed_results": all_results,
#     }

# async def train_model(train_users):
#     trained_vectors = {}
#     for username in train_users:
#         user = await user_profile_collection.find_one({"user_name": username})
#         if not user:
#             continue
        
#         liked_menus = user.get("liked_menu", [])
#         disliked_menus = user.get("disliked_menu", [])
        
#         liked_indices = [i for i, name in enumerate(df["menu_name"]) if name in liked_menus]
#         disliked_indices = [i for i, name in enumerate(df["menu_name"]) if name in disliked_menus]
        
#         liked_vectors = menu_embeddings[liked_indices] if liked_indices else np.zeros((1, menu_embeddings.shape[1]))
#         disliked_vectors = menu_embeddings[disliked_indices] if disliked_indices else np.zeros((1, menu_embeddings.shape[1]))

#         user_vector = np.zeros_like(menu_embeddings[0])
        
#         if len(liked_vectors) > 0 and np.sum(liked_vectors) > 0:
#             user_vector += like_weight * np.mean(liked_vectors, axis=0)
#         if len(disliked_vectors) > 0 and np.sum(disliked_vectors) > 0:
#             user_vector -= dislike_weight * np.mean(disliked_vectors, axis=0)
        
#         # Ensure user_vector has correct shape
#         user_vector = user_vector.reshape(1, -1)
        
#         trained_vectors[username] = user_vector
    
#     return trained_vectors

# def has_allergy(menu_allergens, user_allergies):
#     """
#     Check if the menu contains any allergens that the user is allergic to.
#     """
#     menu_allergy_set = set(str(menu_allergens).split(", "))  # Split allergens as a set
#     return not user_allergies.isdisjoint(menu_allergy_set)

# def matches_food_preference(category_string, user_preferences):
#     """
#     Check if the menu matches at least one of the user's food preferences.
#     """
#     category_list = set(str(category_string).split(","))
#     return not user_preferences.isdisjoint(category_list)


# async def evaluate_on_test_users(test_users, trained_vectors):
#     results = []
#     for username in test_users:
#         user = await user_profile_collection.find_one({"user_name": username})
#         if not user:
#             continue
        
#         liked_menus = user.get("liked_menu", [])
#         disliked_menus = user.get("disliked_menu", [])
#         allergies = set(user.get("allergies", []))
#         food_preferences = set(user.get("food_preferences", []))
        
#         # ✅ Use trained vector directly
#         user_vector = trained_vectors.get(username, np.zeros_like(menu_embeddings[0]))

#         if np.all(user_vector == 0):
#             continue
        
#         # ✅ Reshape vector if needed
#         user_vector = user_vector.reshape(1, -1)
        
#         # ✅ Calculate cosine similarity
#         scores = cosine_similarity(user_vector, menu_embeddings).flatten()
#         sorted_indices = scores.argsort()[::-1]
        
#         # ✅ Get top recommendations (after applying filters)
#         recommended_menus = []
        
#         for i in sorted_indices:
#             name = df.iloc[i]["menu_name"]
#             category = df.iloc[i]["menu_category"]
#             allergens = df.iloc[i].get("matched_allergies", "")

#             # ✅ Apply filters
#             if has_allergy(allergens, allergies):
#                 continue  # Skip menu if it has allergens
#             if not matches_food_preference(category, food_preferences):
#                 continue  # Skip menu if it doesn't match preferences
            
#             # ✅ Add filtered recommendation
#             recommended_menus.append(name)
            
#             if len(recommended_menus) >= 10:
#                 break
        
#         for target_menu in liked_menus + disliked_menus:
#             actual_label = "like" if target_menu in liked_menus else "dislike"
            
#             # ✅ Check if target_menu is in top 10 after filtering
#             is_hit_top10 = target_menu in recommended_menus

#             # ✅ Get similarity score for target_menu
#             target_indices = df[df["menu_name"] == target_menu].index.tolist()
#             target_index = target_indices[0] if target_indices else -1
#             similarity_score = scores[target_index] if target_index >= 0 else 0.0
            
#             # ✅ Store results
#             results.append({
#                 "username": username,
#                 "target_menu": target_menu,
#                 "actual_label": actual_label,
#                 "hit_top10": is_hit_top10,
#                 "similarity_score": float(similarity_score),
#             })
    
#     return results


# @app.get("/api/evaluate/train_test_split")
# async def train_test_split_evaluation():
#     # ✅ Split users for training and testing
#     train_users = ["savetang", "Tanabodee", "mhuyong", "junior1", "tunior"]
#     test_users = ["bubu1234", "dudu1234", "thanidaza"]
    
#     # ✅ Step 1: Train vectors using training users
#     trained_vectors = await train_model(train_users)
    
#     # ✅ Step 2: Evaluate test users using trained vectors
#     test_results = await evaluate_on_test_users(test_users, trained_vectors)
    
#     # ✅ Step 3: Calculate Hit Rate
#     hit_count = sum([1 for result in test_results if result["hit_top10"]])
#     total_tests = len(test_results)
#     hit_rate = hit_count / total_tests if total_tests > 0 else 0.0
#     print(f"✅ Hit Rate: {round(hit_rate * 100, 2)}%")

#     # ✅ Step 4: Prepare binary labels for actual and predicted
#     y_true = [1 if result["actual_label"] == "like" else 0 for result in test_results]
#     y_pred = [1 if result["hit_top10"] else 0 for result in test_results]

#     # ✅ Step 5: Calculate Precision, Recall, F1, and Accuracy
#     precision = precision_score(y_true, y_pred, zero_division=0)
#     recall = recall_score(y_true, y_pred, zero_division=0)
#     f1 = f1_score(y_true, y_pred, zero_division=0)
#     accuracy = accuracy_score(y_true, y_pred)

#     print(f"✅ Precision: {round(precision, 2)}")
#     print(f"✅ Recall: {round(recall, 2)}")
#     print(f"✅ F1-Score: {round(f1, 2)}")
#     print(f"✅ Accuracy: {round(accuracy, 2)}")

#     # ✅ Step 6: Return Results
#     return {
#         "hit_rate": round(hit_rate * 100, 2),
#         "precision": round(precision, 2),
#         "recall": round(recall, 2),
#         "f1_score": round(f1, 2),
#         "accuracy": round(accuracy, 2),
#         "detailed_results": test_results,
#     }