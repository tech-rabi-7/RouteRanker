"""
RouteRanker - GTFS Synthetic & Realistic Data Generator for Indian Cities
Generates GTFS compliant datasets for Delhi (DTC), Bengaluru (BMTC), Mumbai (BEST), 
Pune (PMPML), and Hyderabad (TSRTC).
"""

import os
import math
import random
import pandas as pd
import numpy as np

# Coordinates & Landmarks for Indian Cities
CITY_CONFIGS = {
    "Delhi": {
        "agency_name": "Delhi Transport Corporation (DTC)",
        "center": (28.6139, 77.2090),
        "landmarks": [
            ("Kashmere Gate ISBT", 28.6675, 77.2285),
            ("Connaught Place (Central)", 28.6315, 77.2167),
            ("AIIMS Metro / Hospital", 28.5672, 77.2100),
            ("Anand Vihar ISBT", 28.6469, 77.3160),
            ("Dhaula Kuan Interchange", 28.5921, 77.1563),
            ("Nehru Place Terminal", 28.5494, 77.2526),
            ("Saket District Centre", 28.5244, 77.2177),
            ("Rohini Sector 16", 28.7328, 77.1200),
            ("Dwarka Sector 21", 28.5523, 77.0583),
            ("Karol Bagh Metro", 28.6441, 77.1888),
            ("Lajpat Nagar Ring Road", 28.5700, 77.2400),
            ("Sarai Kale Khan ISBT", 28.5900, 77.2600),
            ("Janakpuri District Centre", 28.6297, 77.0797),
            ("Badarpur Border", 28.4975, 77.3000),
            ("Azadpur Terminal", 28.7077, 77.1755),
            ("Mayur Vihar Phase 1", 28.6083, 77.2944),
            ("Vasant Kunj Promenade", 28.5398, 77.1558),
            ("Punjabi Bagh Club", 28.6681, 77.1328),
            ("IGI Airport Terminal 3", 28.5562, 77.1000),
            ("Noida Sector 62 Link", 28.6270, 77.3620),
        ],
        "route_templates": [
            ("502", "Mehrauli to Kashmere Gate", ["Saket District Centre", "AIIMS Metro / Hospital", "Connaught Place (Central)", "Kashmere Gate ISBT"]),
            ("534", "Anand Vihar to Nehru Place", ["Anand Vihar ISBT", "Mayur Vihar Phase 1", "Sarai Kale Khan ISBT", "Nehru Place Terminal"]),
            ("TMS-Ring", "Mudrika Outer Ring Road", ["Dhaula Kuan Interchange", "AIIMS Metro / Hospital", "Nehru Place Terminal", "Sarai Kale Khan ISBT", "Anand Vihar ISBT", "Azadpur Terminal", "Punjabi Bagh Club", "Dhaula Kuan Interchange"]),
            ("711", "Janakpuri to Sarai Kale Khan", ["Janakpuri District Centre", "Dhaula Kuan Interchange", "AIIMS Metro / Hospital", "Sarai Kale Khan ISBT"]),
            ("729", "Dwarka to Kashmere Gate", ["Dwarka Sector 21", "Dhaula Kuan Interchange", "Karol Bagh Metro", "Kashmere Gate ISBT"]),
            ("615", "Vasant Kunj to Connaught Place", ["Vasant Kunj Promenade", "Dhaula Kuan Interchange", "Connaught Place (Central)"]),
            ("419", "Badarpur to Kashmere Gate", ["Badarpur Border", "Nehru Place Terminal", "Lajpat Nagar Ring Road", "Kashmere Gate ISBT"]),
            ("Airport-Exp", "IGI T3 to Connaught Place", ["IGI Airport Terminal 3", "Dhaula Kuan Interchange", "Connaught Place (Central)"]),
            ("901", "Rohini to Anand Vihar", ["Rohini Sector 16", "Azadpur Terminal", "Kashmere Gate ISBT", "Anand Vihar ISBT"]),
            ("502-Dup", "Mehrauli to Old Delhi (Parallel)", ["Saket District Centre", "AIIMS Metro / Hospital", "Connaught Place (Central)", "Kashmere Gate ISBT"]), # Intentional overlapping route
            ("427", "Nehru Place to Badarpur", ["Nehru Place Terminal", "Badarpur Border"]),
            ("970", "Rohini to Janakpuri", ["Rohini Sector 16", "Punjabi Bagh Club", "Janakpuri District Centre"]),
            ("311", "Noida Link to Anand Vihar", ["Noida Sector 62 Link", "Anand Vihar ISBT"]),
            ("729-Split", "Dwarka to Karol Bagh Feeder", ["Dwarka Sector 21", "Janakpuri District Centre", "Karol Bagh Metro"]),
            ("101-Slow", "Azadpur to Sarai Kale Khan", ["Azadpur Terminal", "Karol Bagh Metro", "Connaught Place (Central)", "Sarai Kale Khan ISBT"])
        ]
    },
    "Bengaluru": {
        "agency_name": "Bangalore Metropolitan Transport Corp (BMTC)",
        "center": (12.9716, 77.5946),
        "landmarks": [
            ("Kempegowda Bus Station (Majestic)", 12.9772, 77.5708),
            ("Silk Board Junction", 12.9177, 77.6238),
            ("ITPL Whitefield", 12.9856, 77.7314),
            ("Electronic City Tollgate", 12.8452, 77.6602),
            ("Hebbal Flyover", 13.0358, 77.5970),
            ("Indiranagar 100ft Road", 12.9784, 77.6408),
            ("Marathahalli Bridge", 12.9569, 77.7011),
            ("Banashankari TTMC", 12.9155, 77.5736),
            ("Jayanagar 4th Block", 12.9304, 77.5838),
            ("Koramangala Sony World", 12.9345, 77.6257),
            ("Yeshwantpur Railway Station", 13.0238, 77.5503),
            ("Kengeri Satellite Town", 12.9176, 77.4837),
            ("Yelahanka Old Town", 13.1007, 77.5963),
            ("Shivajinagar Bus Station", 12.9863, 77.6015),
            ("Bellandur EcoSpace", 12.9260, 77.6762),
            ("Tin Factory KR Puram", 12.9972, 77.6692),
            ("Domlur Flyover", 12.9610, 77.6387),
            ("BTM Layout Udupi Garden", 12.9166, 77.6101),
        ],
        "route_templates": [
            ("500-D", "Hebbal to Silk Board (ORR)", ["Hebbal Flyover", "Tin Factory KR Puram", "Marathahalli Bridge", "Bellandur EcoSpace", "Silk Board Junction"]),
            ("335-E", "Majestic to ITPL Whitefield", ["Kempegowda Bus Station (Majestic)", "Domlur Flyover", "Marathahalli Bridge", "ITPL Whitefield"]),
            ("KIA-9", "Majestic to Kempegowda Airport", ["Kempegowda Bus Station (Majestic)", "Shivajinagar Bus Station", "Hebbal Flyover", "Yelahanka Old Town"]),
            ("356-C", "Majestic to Electronic City", ["Kempegowda Bus Station (Majestic)", "Jayanagar 4th Block", "Silk Board Junction", "Electronic City Tollgate"]),
            ("201-G", "Banashankari to Domlur", ["Banashankari TTMC", "Jayanagar 4th Block", "Koramangala Sony World", "Domlur Flyover"]),
            ("G-2", "Shivajinagar to Sarjapur Road", ["Shivajinagar Bus Station", "Domlur Flyover", "Bellandur EcoSpace"]),
            ("500-D-Dup", "Hebbal to Silk Board Redundant", ["Hebbal Flyover", "Tin Factory KR Puram", "Marathahalli Bridge", "Silk Board Junction"]),
            ("401-K", "Yeshwantpur to Yelahanka", ["Yeshwantpur Railway Station", "Hebbal Flyover", "Yelahanka Old Town"]),
            ("215-H", "Majestic to Banashankari", ["Kempegowda Bus Station (Majestic)", "Jayanagar 4th Block", "Banashankari TTMC"]),
            ("KBS-12", "Majestic to Kengeri", ["Kempegowda Bus Station (Majestic)", "Kengeri Satellite Town"]),
            ("342-F", "KBS to Domlur via Indiranagar", ["Kempegowda Bus Station (Majestic)", "Indiranagar 100ft Road", "Domlur Flyover"]),
            ("V-360B", "Shivajinagar to ITPL", ["Shivajinagar Bus Station", "Indiranagar 100ft Road", "Marathahalli Bridge", "ITPL Whitefield"]),
        ]
    },
    "Mumbai": {
        "agency_name": "Brihanmumbai Electric Supply & Transport (BEST)",
        "center": (19.0760, 72.8777),
        "landmarks": [
            ("CSMT Railway Terminus", 18.9402, 72.8356),
            ("Dadar Western Station", 19.0178, 72.8478),
            ("Bandra Kurla Complex (BKC)", 19.0662, 72.8687),
            ("Andheri West Station", 19.1197, 72.8464),
            ("Borivali West Depot", 19.2307, 72.8567),
            ("Ghatkopar Railway Station", 19.0860, 72.9090),
            ("Kurla West Station", 19.0657, 72.8794),
            ("Colaba Electric House", 18.9150, 72.8270),
            ("Thane Station West", 19.1860, 72.9759),
            ("Sion Circle", 19.0360, 72.8600),
            ("Worli Naka", 19.0020, 72.8180),
            ("Powai Hiranandani", 19.1190, 72.9080),
        ],
        "route_templates": [
            ("AS-4", "CSMT to BKC Express", ["CSMT Railway Terminus", "Dadar Western Station", "Bandra Kurla Complex (BKC)"]),
            ("332", "Andheri West to Kurla Depot", ["Andheri West Station", "Bandra Kurla Complex (BKC)", "Kurla West Station"]),
            ("C-40", "Colaba to Ghatkopar Express", ["Colaba Electric House", "CSMT Railway Terminus", "Dadar Western Station", "Sion Circle", "Ghatkopar Railway Station"]),
            ("707-LTD", "Borivali to Dadar Highway", ["Borivali West Depot", "Andheri West Station", "Bandra Kurla Complex (BKC)", "Dadar Western Station"]),
            ("83", "Colaba to Thane Central", ["Colaba Electric House", "Dadar Western Station", "Ghatkopar Railway Station", "Thane Station West"]),
            ("312", "Andheri to Powai Tech Park", ["Andheri West Station", "Powai Hiranandani"]),
            ("332-Dup", "Andheri to Kurla Parallel", ["Andheri West Station", "Bandra Kurla Complex (BKC)", "Kurla West Station"]),
            ("105", "Dadar to Worli Ferry Point", ["Dadar Western Station", "Worli Naka"]),
        ]
    },
    "Pune": {
        "agency_name": "Pune Mahanagar Parivahan Mahamandal Ltd (PMPML)",
        "center": (18.5204, 73.8567),
        "landmarks": [
            ("Swargate Bus Station", 18.5018, 73.8586),
            ("Pune Railway Station", 18.5284, 73.8744),
            ("Hinjawadi IT Park Phase 1", 18.5913, 73.7389),
            ("Katraj Snake Park / Depot", 18.4552, 73.8576),
            ("Kothrud Depot", 18.5074, 73.8077),
            ("Hadapsar Gadital", 18.5020, 73.9280),
            ("Viman Nagar Terminal", 18.5679, 73.9143),
            ("Shivajinagar Bus Stand", 18.5314, 73.8446),
            ("Nigdi Pradhikaran", 18.6540, 73.7740),
            ("Wakad Bridge", 18.5987, 73.7631),
        ],
        "route_templates": [
            ("115", "Swargate to Hinjawadi IT Park", ["Swargate Bus Station", "Shivajinagar Bus Stand", "Wakad Bridge", "Hinjawadi IT Park Phase 1"]),
            ("204", "Katraj to Pune Station", ["Katraj Snake Park / Depot", "Swargate Bus Station", "Pune Railway Station"]),
            ("100", "Kothrud to Viman Nagar", ["Kothrud Depot", "Shivajinagar Bus Stand", "Pune Railway Station", "Viman Nagar Terminal"]),
            ("115-Dup", "Swargate to Hinjawadi Parallel", ["Swargate Bus Station", "Shivajinagar Bus Stand", "Wakad Bridge", "Hinjawadi IT Park Phase 1"]),
            ("348", "Hadapsar to Nigdi BRT", ["Hadapsar Gadital", "Pune Railway Station", "Shivajinagar Bus Stand", "Nigdi Pradhikaran"]),
            ("42", "Katraj to Nigdi Corridor", ["Katraj Snake Park / Depot", "Swargate Bus Station", "Shivajinagar Bus Stand", "Nigdi Pradhikaran"]),
            ("159", "Pune Station to Hadapsar", ["Pune Railway Station", "Hadapsar Gadital"]),
        ]
    },
    "Hyderabad": {
        "agency_name": "Telangana State Road Transport Corp (TSRTC)",
        "center": (17.3850, 78.4867),
        "landmarks": [
            ("Secunderabad Railway Station", 17.4399, 78.4983),
            ("Koti Bus Station", 17.3833, 78.4842),
            ("Hitec City Cyber Towers", 17.4504, 78.3808),
            ("Mehdipatnam Depot", 17.3916, 78.4398),
            ("Gachibowli Stadium", 17.4435, 78.3489),
            ("Charminar Old City", 17.3616, 78.4747),
            ("Kukatpally Housing Board", 17.4933, 78.3995),
            ("Dilsukhnagar Bus Station", 17.3688, 78.5247),
            ("Ameerpet Metro Junction", 17.4375, 78.4482),
            ("Jubilee Hills Check Post", 17.4298, 78.4116),
        ],
        "route_templates": [
            ("10H", "Secunderabad to Hitec City", ["Secunderabad Railway Station", "Ameerpet Metro Junction", "Jubilee Hills Check Post", "Hitec City Cyber Towers"]),
            ("216", "Mehdipatnam to Gachibowli", ["Mehdipatnam Depot", "Gachibowli Stadium"]),
            ("127K", "Koti to Kukatpally KPHB", ["Koti Bus Station", "Ameerpet Metro Junction", "Kukatpally Housing Board"]),
            ("10H-Dup", "Secunderabad to Hitec City Redundant", ["Secunderabad Railway Station", "Ameerpet Metro Junction", "Hitec City Cyber Towers"]),
            ("9X", "Charminar to Secunderabad", ["Charminar Old City", "Koti Bus Station", "Secunderabad Railway Station"]),
            ("222", "Koti to Hitec City via Mehdipatnam", ["Koti Bus Station", "Mehdipatnam Depot", "Gachibowli Stadium", "Hitec City Cyber Towers"]),
            ("218", "Dilsukhnagar to Hitec City", ["Dilsukhnagar Bus Station", "Koti Bus Station", "Mehdipatnam Depot", "Hitec City Cyber Towers"]),
            ("187", "Kukatpally to Gachibowli Feeder", ["Kukatpally Housing Board", "Hitec City Cyber Towers", "Gachibowli Stadium"]),
        ]
    }
}

def haversine_dist_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2.0)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2.0)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def create_gtfs_for_city(city_name, config, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    random.seed(42)
    np.random.seed(42)

    landmarks = config["landmarks"]
    route_templates = config["route_templates"]

    # 1. stops.txt
    stops_data = []
    stop_name_to_id = {}
    for i, (name, lat, lon) in enumerate(landmarks, 1):
        stop_id = f"ST_{city_name[:3].upper()}_{i:03d}"
        stop_name_to_id[name] = stop_id
        stops_data.append({
            "stop_id": stop_id,
            "stop_name": name,
            "stop_lat": round(lat, 6),
            "stop_lon": round(lon, 6),
            "zone_id": f"ZONE_{city_name[:3].upper()}_{i%4 + 1}"
        })
    df_stops = pd.DataFrame(stops_data)
    df_stops.to_csv(os.path.join(output_dir, "stops.txt"), index=False)

    # 2. calendar.txt
    df_calendar = pd.DataFrame([
        {
            "service_id": "WD_REGULAR",
            "monday": 1, "tuesday": 1, "wednesday": 1, "thursday": 1, "friday": 1,
            "saturday": 0, "sunday": 0,
            "start_date": "20260101", "end_date": "20261231"
        },
        {
            "service_id": "WE_REGULAR",
            "monday": 0, "tuesday": 0, "wednesday": 0, "thursday": 0, "friday": 0,
            "saturday": 1, "sunday": 1,
            "start_date": "20260101", "end_date": "20261231"
        }
    ])
    df_calendar.to_csv(os.path.join(output_dir, "calendar.txt"), index=False)

    # 3. routes.txt, trips.txt, stop_times.txt, shapes.txt
    routes_data = []
    trips_data = []
    stop_times_data = []
    shapes_data = []

    # Assign distinct operational profiles to routes
    # E.g. Overcrowded Trunk, Underperforming Feeder, Balanced Commuter, Redundant Duplicate
    profile_assignments = [
        "Overcrowded", "Underutilized", "Overcrowded", "Balanced",
        "Balanced", "Underutilized", "Balanced", "Overcrowded",
        "Underutilized", "Underutilized", "Balanced", "Overcrowded",
        "Underutilized", "Balanced", "Underutilized"
    ]

    for r_idx, (r_code, r_desc, path_stops) in enumerate(route_templates):
        route_id = f"RT_{city_name[:3].upper()}_{r_code}"
        profile = profile_assignments[r_idx % len(profile_assignments)]

        # Route color
        color_hex = "FF5733" if profile == "Overcrowded" else ("3399FF" if profile == "Balanced" else "999999")
        routes_data.append({
            "route_id": route_id,
            "agency_id": config["agency_name"],
            "route_short_name": r_code,
            "route_long_name": r_desc,
            "route_type": 3, # 3 = Bus
            "route_color": color_hex,
            "route_text_color": "FFFFFF"
        })

        # Frequency & trips determined by profile
        # Overcrowded: high passenger demand but insufficient trips
        # Underutilized: low demand but high frequency, or very low ridership
        # Balanced: well-matched capacity & demand
        if profile == "Overcrowded":
            num_trips_wd = 22  # Not enough buses for the heavy rush
            base_pax_capacity = 60
            demand_multiplier = 1.6
        elif profile == "Underutilized":
            num_trips_wd = 30  # High empty frequency
            base_pax_capacity = 60
            demand_multiplier = 0.35
        else: # Balanced
            num_trips_wd = 45
            base_pax_capacity = 60
            demand_multiplier = 0.85

        # Create trips throughout the day (06:00 to 22:00)
        shape_id = f"SHP_{route_id}"
        shape_pt_seq = 1

        # Calculate distances along the path stops
        for s_idx, stop_name in enumerate(path_stops):
            s_lat = df_stops.loc[df_stops["stop_name"] == stop_name, "stop_lat"].iloc[0]
            s_lon = df_stops.loc[df_stops["stop_name"] == stop_name, "stop_lon"].iloc[0]
            shapes_data.append({
                "shape_id": shape_id,
                "shape_pt_lat": s_lat,
                "shape_pt_lon": s_lon,
                "shape_pt_sequence": shape_pt_seq
            })
            shape_pt_seq += 1

        # Generate weekday trips
        start_hour = 6
        trip_intervals_mins = max(12, int((16 * 60) / num_trips_wd))

        for t_i in range(num_trips_wd):
            t_offset = t_i * trip_intervals_mins
            dep_hour = start_hour + (t_offset // 60)
            dep_min = t_offset % 60
            if dep_hour >= 23:
                break

            trip_id = f"TR_{route_id}_WD_{t_i:03d}"
            trips_data.append({
                "route_id": route_id,
                "service_id": "WD_REGULAR",
                "trip_id": trip_id,
                "trip_headsign": path_stops[-1],
                "direction_id": 0,
                "shape_id": shape_id
            })

            # Stop times
            current_minute = dep_hour * 60 + dep_min
            for s_seq, stop_name in enumerate(path_stops, 1):
                s_id = stop_name_to_id[stop_name]
                # calculate realistic running time between stops (approx 8-15 mins per segment)
                if s_seq > 1:
                    prev_stop = path_stops[s_seq - 2]
                    p_lat = df_stops.loc[df_stops["stop_name"] == prev_stop, "stop_lat"].iloc[0]
                    p_lon = df_stops.loc[df_stops["stop_name"] == prev_stop, "stop_lon"].iloc[0]
                    c_lat = df_stops.loc[df_stops["stop_name"] == stop_name, "stop_lat"].iloc[0]
                    c_lon = df_stops.loc[df_stops["stop_name"] == stop_name, "stop_lon"].iloc[0]
                    dist = haversine_dist_km(p_lat, p_lon, c_lat, c_lon)
                    # speed ~ 18 km/h in city traffic
                    travel_time = max(5, int((dist / 18.0) * 60 + random.randint(1, 4)))
                    current_minute += travel_time

                arr_h = current_minute // 60
                arr_m = current_minute % 60
                arr_str = f"{arr_h:02d}:{arr_m:02d}:00"
                dep_str = f"{arr_h:02d}:{(arr_m + 1)%60:02d}:00"

                stop_times_data.append({
                    "trip_id": trip_id,
                    "arrival_time": arr_str,
                    "departure_time": dep_str,
                    "stop_id": s_id,
                    "stop_sequence": s_seq,
                    "pickup_type": 0,
                    "drop_off_type": 0
                })

    df_routes = pd.DataFrame(routes_data)
    df_routes.to_csv(os.path.join(output_dir, "routes.txt"), index=False)

    df_trips = pd.DataFrame(trips_data)
    df_trips.to_csv(os.path.join(output_dir, "trips.txt"), index=False)

    df_stop_times = pd.DataFrame(stop_times_data)
    df_stop_times.to_csv(os.path.join(output_dir, "stop_times.txt"), index=False)

    df_shapes = pd.DataFrame(shapes_data)
    df_shapes.to_csv(os.path.join(output_dir, "shapes.txt"), index=False)

    print(f"Generated GTFS feed for {city_name} at {output_dir}: "
          f"{len(df_routes)} routes, {len(df_stops)} stops, {len(df_trips)} trips.")

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    sample_dir = os.path.join(base_dir, "data", "sample_gtfs")

    for city_name, config in CITY_CONFIGS.items():
        city_slug = city_name.lower()
        city_output_dir = os.path.join(sample_dir, city_slug)
        create_gtfs_for_city(city_name, config, city_output_dir)

if __name__ == "__main__":
    main()
