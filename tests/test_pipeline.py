"""
Unit and Integration Tests for RouteRanker Transit Pipeline
"""

import os
import unittest
import pandas as pd
from src.gtfs_loader import GTFSFeed
from src.feature_engineering import RouteFeatureEngineer, haversine_km
from src.eda import TransitEDA
from src.models import RouteMLPipeline
from src.optimizer import RouteOptimizer

class TestRouteRankerPipeline(unittest.TestCase):

    def setUp(self):
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.feed_path = os.path.join(self.base_dir, "data", "sample_gtfs", "delhi")

    def test_gtfs_loader(self):
        feed = GTFSFeed(self.feed_path)
        self.assertTrue(feed.validation_report["valid"])
        self.assertFalse(feed.routes.empty)
        self.assertFalse(feed.trips.empty)
        self.assertFalse(feed.stops.empty)
        self.assertFalse(feed.stop_times.empty)

        # Test stop coordinates retrieval
        sample_route = feed.routes.iloc[0]["route_id"]
        coords = feed.get_route_coordinates(sample_route)
        self.assertGreater(len(coords), 0)

    def test_haversine(self):
        # Distance between Connaught Place and AIIMS in Delhi is ~7-8 km
        dist = haversine_km(28.6315, 77.2167, 28.5672, 77.2100)
        self.assertGreater(dist, 5.0)
        self.assertLess(dist, 10.0)

    def test_feature_engineering(self):
        feed = GTFSFeed(self.feed_path)
        fe = RouteFeatureEngineer(feed)
        profiles = fe.build_route_profiles()
        self.assertFalse(profiles.empty)
        self.assertIn("peak_load_factor", profiles.columns)
        self.assertIn("performance_status", profiles.columns)
        self.assertIn("route_length_km", profiles.columns)
        self.assertIn("avg_headway_min", profiles.columns)

        hourly = fe.build_hourly_ridership()
        self.assertEqual(len(hourly), 24)

    def test_eda_and_overlap(self):
        feed = GTFSFeed(self.feed_path)
        fe = RouteFeatureEngineer(feed)
        profiles = fe.build_route_profiles()

        eda = TransitEDA(feed, profiles)
        matrix, pairs = eda.compute_route_overlap_matrix()
        self.assertFalse(matrix.empty)
        self.assertIsInstance(pairs, list)

        health = eda.get_network_health_metrics()
        self.assertIn("network_inefficiency_rate_pct", health)

    def test_ml_pipeline(self):
        feed = GTFSFeed(self.feed_path)
        fe = RouteFeatureEngineer(feed)
        profiles = fe.build_route_profiles()

        pipeline = RouteMLPipeline(model_dir=os.path.join(self.base_dir, "models"))
        clustered = pipeline.fit_clustering(profiles, n_clusters=3)
        self.assertIn("kmeans_cluster", clustered.columns)
        self.assertIn("pca_1", clustered.columns)

        results = pipeline.train_classifiers(clustered)
        self.assertGreaterEqual(results["rf_metrics"]["accuracy"], 0.70)
        self.assertGreaterEqual(results["xgb_metrics"]["accuracy"], 0.70)

        preds = pipeline.predict_route_performance(clustered)
        self.assertIn("rf_predicted_status", preds.columns)

    def test_optimizer(self):
        feed = GTFSFeed(self.feed_path)
        fe = RouteFeatureEngineer(feed)
        profiles = fe.build_route_profiles()

        eda = TransitEDA(feed, profiles)
        _, pairs = eda.compute_route_overlap_matrix()

        optimizer = RouteOptimizer()
        recs = optimizer.generate_recommendations(profiles, pairs)
        self.assertIsInstance(recs, list)

        impact = optimizer.simulate_network_impact(profiles, recs)
        self.assertIn("savings", impact)
        self.assertIn("net_km_saved_daily", impact["savings"])
        self.assertGreaterEqual(impact["savings"]["net_km_saved_daily"], 0)

if __name__ == "__main__":
    unittest.main()
