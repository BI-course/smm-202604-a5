"""
api.py

A RESTish API, built with Flask, that serves FOUR machine learning artifacts
produced in earlier labs:

    1. Regression       -- SME monthly revenue prediction
    2. Classification   -- SME credit risk category
    3. Clustering       -- Mall customer segment assignment
    4. Association Rule -- Product recommendation from a cart
       Mining (ARM)

The point of this file is to show you that a fitted scikit-learn /
imbalanced-learn Pipeline is a SINGLE object that already knows how to impute
missing values, encode categorical columns, scale numeric columns, and (for
the classifier) apply SMOTE only during training -- never during inference.
Because all of that preprocessing lives inside the Pipeline itself, this file
never manually label-encodes anything, never manually engineers date features,
and never manually reorders columns before calling .predict(). It builds a
DataFrame straight from the incoming JSON and hands it to the pipeline as-is.
This is the main lesson of the lab: "the pipeline IS the deployment artifact."

Run this file directly to start a local development server:

    python api.py
"""

import os
import numpy as np

import joblib
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS


app = Flask(__name__)

# -----------------------------------------------------------------------
# CORS (Cross-Origin Resource Sharing)
# -----------------------------------------------------------------------
# By default, a browser blocks JavaScript on one origin (e.g., a page served
# from http://127.0.0.1:5500) from calling an API on a different origin
# (e.g., http://127.0.0.1:5000). `flask_cors` relaxes that restriction for the
# specific origins we trust during local development.
#
# A few alternative configurations are shown below (commented out) so you can
# see how the restriction can be loosened or tightened depending on your
# deployment target. Only ONE of these should be active at a time.

# Option A (the most permissive. This is fine for a quick classroom demo or
# local development, *NEVER* for production):
# CORS(app)

# Option B (allow only a specific frontend origin):
# CORS(app, resources={r"/api/*": {"origins": "http://127.0.0.1:5500"}})

# Option C (the one actually in use here -- allow localhost/127.0.0.1 on the
# ports our lab's frontend files are typically served from):
CORS(app, resources={r"/api/*": {"origins": [
    "http://localhost:5000", "http://127.0.0.1:5000",
    "http://localhost:5500", "http://127.0.0.1:5500",
    "https://localhost:443", "https://127.0.0.1:443",
    "http://localhost:63342",  # VS Code Live Server extension
    "null",  # allows opening a frontend .html file directly (file://) with no web server
]}})

# -----------------------------------------------------------------------
# The best practice is to load the four persisted artifacts ONCE, at startup,
# not per-request.
# Loading inside a route function would re-read the file from disk on every
# single API call, which is slow and pointless -- the fitted model never
# changes while the server is running.
# -----------------------------------------------------------------------
MODEL_DIR = "./model"

regression_pipeline = joblib.load(
    os.path.join(MODEL_DIR, "ridge_regressor_for_monthly_rent.joblib")
)

classification_pipeline = joblib.load(
    os.path.join(MODEL_DIR, "gradientboosting_classifier_for_employee_attrition.joblib")
)

clustering_artifact = joblib.load(
    os.path.join(MODEL_DIR, "kmeans_store_segmentation.joblib")
)
clustering_scaler = clustering_artifact["scaler"]
clustering_kmeans = clustering_artifact["kmeans"]

arm_artifact = joblib.load(
    os.path.join(MODEL_DIR, "apriori_recommendation_rules.joblib")
)
recommendation_lookup = arm_artifact["recommendation_lookup"]
track_lookup = arm_artifact["track_lookup"]

print("All the four artifacts have been loaded. The API is ready.")

def build_input_row(payload, required, nullable=()):
    """Validate the JSON body and turn it into a one-row DataFrame.
    `nullable` fields may be missing or null; they are converted to a real
    NaN so the pipeline's imputer can handle them."""
    if not isinstance(payload, dict):
        return None, (jsonify({"error": "Send a JSON object with the model inputs."}), 400)
    payload = dict(payload)
    for field in nullable:
        payload.setdefault(field, None)
    missing = [f for f in required if f not in payload]
    if missing:
        return None, (jsonify({"error": "Missing inputs", "missing_fields": missing}), 400)
    row = pd.DataFrame([payload])
    for field in nullable:
        row[field] = pd.to_numeric(row[field], errors="coerce").astype(float)
    return row, None

# 1. REGRESSION -- predict monthly rent (KES) for a property
@app.route("/api/v1/models/monthly-rent-regressor/predictions", methods=["POST"])
def predict_rent():
    input_row, error = build_input_row(
        request.get_json(silent=True),
        required=["property_age_years", "num_bedrooms", "distance_to_cbd_km",
                  "garage_size_sqm", "amenities_score", "nearby_schools_count",
                  "property_type", "neighborhood", "furnishing_status"],
        nullable=["garage_size_sqm"],
    )
    if error:
        return error
    try:
        predicted_log_rent = regression_pipeline.predict(input_row)[0]
        predicted_rent_kes = float(np.expm1(predicted_log_rent))
    except Exception as e:
        return jsonify({"error": f"Could not make a prediction: {e}"}), 400
    return jsonify({
        "predicted_monthly_rent_kes": round(predicted_rent_kes, 2),
        "model": "Ridge Regressor",
    })


# 2. CLASSIFICATION -- predict employee attrition (Yes/No)
@app.route("/api/v1/models/employee-attrition-classifier/predictions", methods=["POST"])
def predict_attrition():
    input_row, error = build_input_row(
        request.get_json(silent=True),
        required=["department", "job_role_level", "education_level",
                  "performance_rating", "years_at_company",
                  "years_since_last_promotion", "monthly_salary_kes",
                  "overtime_hours_per_month", "job_satisfaction_score",
                  "work_life_balance_score", "distance_from_home_km",
                  "num_companies_worked_previously",
                  "remote_work_days_per_week", "bonus_amount_kes"],
        nullable=["bonus_amount_kes"],
    )
    if error:
        return error
    try:
        predicted_class = classification_pipeline.predict(input_row)[0]
        probabilities = classification_pipeline.predict_proba(input_row)[0]
        class_labels = classification_pipeline.named_steps["model"].classes_
    except Exception as e:
        return jsonify({"error": f"Could not make a prediction: {e}"}), 400
    return jsonify({
        "predicted_attrition": str(predicted_class),
        "class_probabilities": {
            str(label): round(float(p), 4)
            for label, p in zip(class_labels, probabilities)
        },
        "model": "Gradient Boosting Classifier",
    })


# 3. CLUSTERING -- assign a store to an existing segment
@app.route("/api/v1/models/store-segmenter/predictions", methods=["POST"])
def predict_store_segment():
    features = list(clustering_scaler.feature_names_in_)
    input_row, error = build_input_row(request.get_json(silent=True), required=features)
    if error:
        return error
    try:
        scaled_input = clustering_scaler.transform(input_row[features])
        assigned_cluster = int(clustering_kmeans.predict(scaled_input)[0])
    except Exception as e:
        return jsonify({"error": f"Could not make a prediction: {e}"}), 400
    return jsonify({
        "assigned_cluster": assigned_cluster,
        "model": "K-Means (k=5, standardized features)",
    })


# 4. ASSOCIATION RULE MINING -- recommend tracks for a cart
@app.route("/api/v1/models/track-recommender/predictions", methods=["POST"])
def recommend_tracks():
    payload = request.get_json(silent=True)
    if (not isinstance(payload, dict) or not isinstance(payload.get("cart"), list)
            or not payload["cart"]):
        return jsonify({"error": "Send a JSON body like {'cart': ['M01', 'M03']}"}), 400
    cart = set(payload["cart"])
    try:
        top_n = int(payload.get("top_n", 3))
    except (TypeError, ValueError):
        return jsonify({"error": "top_n must be a whole number"}), 400

    candidates = {}
    for antecedent, consequent_list in recommendation_lookup.items():
        if antecedent.issubset(cart):
            for track_id, confidence, lift in consequent_list:
                if track_id not in cart:
                    if track_id not in candidates or confidence > candidates[track_id][0]:
                        candidates[track_id] = (confidence, lift)

    ranked = sorted(candidates.items(), key=lambda item: item[1][0], reverse=True)[:top_n]
    recommendations = [
        {
            "track_id": str(track_id),
            "item_name": track_lookup.loc[track_id, "item_name"],
            "category": track_lookup.loc[track_id, "category"],
            "confidence": round(float(confidence), 3),
            "lift": round(float(lift), 2),
        }
        for track_id, (confidence, lift) in ranked
    ]
    return jsonify({
        "cart": sorted(cart),
        "recommendations": recommendations,
        "model": "Apriori association rules, cleaned + precomputed lookup",
    })



# 1. REGRESSION -- predict monthly revenue (KES) for an SME

# Sample request (cURL):
#   curl -X POST http://127.0.0.1:5000/api/v1/models/sme-revenue-regressor/predictions \
#        -H "Content-Type: application/json" \
#        -d '{"sector": "Technology", "county": "Nairobi", "quarter": "Q3",
#             "business_age_years": 4.5, "num_employees": 12,
#             "owner_education_level": "Secondary",
#             "monthly_marketing_spend_kes": 45000,
#             "avg_daily_foot_traffic": null,
#             "online_presence_score": 78.0, "loan_amount_kes": 250000,
#             "interest_rate_pct": 14.5, "inflation_rate_pct": 6.5,
#             "competitor_density": 5, "customer_satisfaction_score": 4.2}'
#
# Sample request (PowerShell):
#   Invoke-RestMethod -Method Post `
#     -Uri "http://127.0.0.1:5000/api/v1/models/sme-revenue-regressor/predictions" `
#     -ContentType "application/json" `
#     -Body (@{sector="Technology"; county="Nairobi"; quarter="Q3";
#              business_age_years=4.5; num_employees=12;
#              owner_education_level="Secondary";
#              monthly_marketing_spend_kes=45000; avg_daily_foot_traffic=$null;
#              online_presence_score=78.0; loan_amount_kes=250000;
#              interest_rate_pct=14.5; inflation_rate_pct=6.5;
#              competitor_density=5; customer_satisfaction_score=4.2} | ConvertTo-Json)
#
# Note: avg_daily_foot_traffic is deliberately null for a Technology-sector
# business (fully online -- foot traffic is structurally not applicable, not
# missing at random). The pipeline's own imputer handles the null value; nothing
# in this file needs to know that rule.
# @app.route("/api/v1/models/sme-revenue-regressor/predictions", methods=["POST"])
# def predict_revenue():
#     payload = request.get_json()

#     # Build a one-row DataFrame straight from the JSON body. No manual
#     # label-encoding, no manual date parsing, no manual column reordering, etc. --
#     # the pipeline's ColumnTransformer already knows which columns it expects
#     # and how to preprocess each one.
#     input_row = pd.DataFrame([payload])

#     predicted_log_revenue = regression_pipeline.predict(input_row)[0]
#     # The pipeline was trained on log1p(revenue) because revenue was
#     # right-skewed; we invert that single transformation here to report the
#     # predictions in KES.
#     import numpy as np
#     predicted_revenue_kes = float(np.expm1(predicted_log_revenue))

#     return jsonify({
#         "predicted_monthly_revenue_kes": round(predicted_revenue_kes, 2),
#         "model": "Lasso Regressor",
#     })


# # 2. CLASSIFICATION -- predict SME credit risk category (Low/Medium/High)

# # Sample request (cURL):
# #   curl -X POST http://127.0.0.1:5000/api/v1/models/sme-credit-risk-classifier/predictions \
# #        -H "Content-Type: application/json" \
# #        -d '{"sector": "Retail", "county": "Nairobi", "quarter": "Q2",
# #             "loan_purpose": "Working Capital", "owner_education_level": "Secondary",
# #             "business_age_years": 2.1, "num_employees": 4,
# #             "monthly_revenue_kes": 850000, "loan_amount_kes": 150000,
# #             "loan_term_months": 12, "interest_rate_pct": 16.5,
# #             "collateral_value_kes": null, "debt_to_income_ratio": 0.62,
# #             "credit_score": 590, "num_previous_loans": 1,
# #             "missed_payments_count": 3, "online_presence_score": 40.0,
# #             "customer_satisfaction_score": 3.4, "competitor_density": 6}'
# #
# #
# # Note: collateral_value_kes is null here because "Working Capital" loans are
# # more often unsecured (structural missingness, not missing at random) --
# # again, the pipeline's imputer + missingness indicator already handles this
# # internally; this route does not need special code for it.
# @app.route("/api/v1/models/sme-credit-risk-classifier/predictions", methods=["POST"])
# def predict_credit_risk():
#     payload = request.get_json()
#     input_row = pd.DataFrame([payload])

#     predicted_class = classification_pipeline.predict(input_row)[0]
#     predicted_probabilities = classification_pipeline.predict_proba(input_row)[0]
#     class_labels = classification_pipeline.named_steps["model"].classes_

#     return jsonify({
#         "predicted_credit_risk_category": str(predicted_class),
#         "class_probabilities": {
#             str(label): round(float(prob), 4)
#             for label, prob in zip(class_labels, predicted_probabilities)
#         },
#         "model": "Support Vector Classifier (tuned, SMOTE-resampled at training only)",
#     })


# # 3. CLUSTERING -- assign a new mall customer to an existing segment

# # Sample request (cURL):
# #   curl -X POST http://127.0.0.1:5000/api/v1/models/mall-customer-segmenter/predictions \
# #        -H "Content-Type: application/json" \
# #        -d '{"Age": 27, "Annual Income (k$)": 45, "Spending Score (1-100)": 72}'
# #
# # Note: unlike the two Pipelines above, K-Means does not bundle its own
# # scaler, so this is the one endpoint where the scaling step is applied
# # explicitly (scaler.transform(...) then kmeans.predict(...)) -- exactly the
# # two-object pattern the clustering notebook persisted together for this
# # reason. There is still no manual encoding or feature engineering: the
# # three numeric fields are used exactly as the client sends them.
# @app.route("/api/v1/models/mall-customer-segmenter/predictions", methods=["POST"])
# def predict_customer_segment():
#     payload = request.get_json()
#     input_row = pd.DataFrame([payload])

#     scaled_input = clustering_scaler.transform(input_row)
#     assigned_cluster = int(clustering_kmeans.predict(scaled_input)[0])

#     return jsonify({
#         "assigned_cluster": assigned_cluster,
#         "model": "K-Means (k=5, standardized features)",
#     })


# # 4. ASSOCIATION RULE MINING -- recommend products for a shopping cart

# # Sample request (cURL):
# #   curl -X POST http://127.0.0.1:5000/api/v1/models/ecommerce-recommender/predictions \
# #        -H "Content-Type: application/json" \
# #        -d '{"cart": ["H06", "H07"]}'
# #
# #   curl -X POST http://127.0.0.1:5000/api/v1/models/ecommerce-recommender/predictions \
# #        -H "Content-Type: application/json" \
# #        -d '{"cart": ["P04", "P01"]}'
# #
# # Note: there is no scikit-learn ".predict()" here -- the mined association
# # rules were precomputed into a lookup dictionary
# # (antecedent SKU-set -> ranked candidate consequents)
# # during the notebook's persistence stage. This
# # endpoint performs the exact same lookup+rank step a production recommender
# # microservice would perform on every incoming request; it does not mine
# # for the rules again.
# @app.route("/api/v1/models/ecommerce-recommender/predictions", methods=["POST"])
# def recommend_products():
#     payload = request.get_json()
#     cart = set(payload.get("cart", []))
#     top_n = int(payload.get("top_n", 3))

#     candidates = {}
#     for antecedent, consequent_list in recommendation_lookup.items():
#         if antecedent.issubset(cart):
#             for sku, confidence, lift in consequent_list:
#                 if sku not in cart:
#                     if sku not in candidates or confidence > candidates[sku][0]:
#                         candidates[sku] = (confidence, lift)

#     ranked = sorted(candidates.items(), key=lambda item: item[1][0], reverse=True)[:top_n]
#     recommendations = [
#         {
#             "sku": str(sku),
#             "product_name": sku_lookup.loc[sku, "product_name"],
#             "confidence": round(float(confidence), 3),
#             "lift": round(float(lift), 2),
#         }
#         for sku, (confidence, lift) in ranked
#     ]

#     return jsonify({
#         "cart": sorted(cart),
#         "recommendations": recommendations,
#         "model": "Apriori association rules, cleaned + precomputed lookup",
#     })


# -----------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------
if __name__ == "__main__":
    # Development server. debug=True gives auto-reload and an interactive
    # debugger on errors -- convenient in class, but NEVER use debug=True in
    # a deployed/public environment (it can expose a remote code execution
    # console to anyone who can reach the server).
    app.run(debug=True)

    # Production-style alternative (no debugger, no auto-reload):
    # app.run(debug=False, host="0.0.0.0", port=5000)

    # HTTPS/TLS alternative for local development (requires a self-signed
    # certificate; see app_server_reverse_proxy_server_setup.md):
    # app.run(debug=False, ssl_context=("cert.pem", "key.pem"))
