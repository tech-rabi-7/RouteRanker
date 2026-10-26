"""
RouteRanker - Model Training & Preprocessing CLI Script
Loads GTFS feeds, constructs route feature matrices, trains Random Forest & XGBoost,
fits KMeans & DBSCAN clustering, and evaluates performance metrics.
"""

import os
import json
import pandas as pd
from src.gtfs_loader import GTFSFeed
from src.feature_engineering import RouteFeatureEngineer
from src.models import RouteMLPipeline

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    sample_dir = os.path.join(base_dir, "data", "sample_gtfs")
    models_dir = os.path.join(base_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    print("=== RouteRanker AI Pipeline Training ===")

    # Aggregate profiles across all cities to train robust models
    all_profiles = []
    cities = ["delhi", "bengaluru", "mumbai", "pune", "hyderabad"]

    for city in cities:
        city_path = os.path.join(sample_dir, city)
        if not os.path.exists(city_path):
            continue
        print(f"Loading GTFS feed for {city.capitalize()}...")
        feed = GTFSFeed(city_path)
        fe = RouteFeatureEngineer(feed)
        df_p = fe.build_route_profiles()
        if not df_p.empty:
            df_p["city"] = city.capitalize()
            all_profiles.append(df_p)

    if not all_profiles:
        print("No profiles generated. Exiting.")
        return

    combined_profiles = pd.concat(all_profiles, ignore_index=True)
    print(f"Total routes across networks: {len(combined_profiles)}")

    pipeline = RouteMLPipeline(model_dir=models_dir)

    print("Fitting Clustering (KMeans & DBSCAN)...")
    clustered_df = pipeline.fit_clustering(combined_profiles, n_clusters=3)

    print("Training Classifiers (Random Forest & XGBoost)...")
    results = pipeline.train_classifiers(clustered_df)

    rf_acc = results["rf_metrics"]["accuracy"]
    rf_f1 = results["rf_metrics"]["f1_score"]
    xgb_acc = results["xgb_metrics"]["accuracy"]
    xgb_f1 = results["xgb_metrics"]["f1_score"]

    print("\n--- Model Evaluation Results ---")
    print(f"Random Forest Classifier -> Accuracy: {rf_acc*100:.1f}%, F1-Score: {rf_f1*100:.1f}%")
    print(f"XGBoost Classifier       -> Accuracy: {xgb_acc*100:.1f}%, F1-Score: {xgb_f1*100:.1f}%")

    metrics_path = os.path.join(models_dir, "training_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved evaluation metrics to {metrics_path}")

    # Save processed combined dataset
    processed_dir = os.path.join(base_dir, "data", "processed")
    os.makedirs(processed_dir, exist_ok=True)
    combined_profiles.to_csv(os.path.join(processed_dir, "all_routes_profiles.csv"), index=False)
    print("Saved aggregated route profiles to data/processed/all_routes_profiles.csv")
    print("=== Training Complete Successfully ===")

if __name__ == "__main__":
    main()
