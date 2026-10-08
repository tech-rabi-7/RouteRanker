"""
RouteRanker - Comprehensive GTFS Dataset Generator for 23 Indian Cities
Generates authentic GTFS feeds for Delhi, Bengaluru, Mumbai, Hyderabad, Pune,
Chennai, Kolkata, Ahmedabad, Jaipur, Surat, Lucknow, Kochi, Chandigarh,
Bhopal, Indore, Nagpur, Visakhapatnam, Coimbatore, Patna, Guwahati,
Thiruvananthapuram, Bhubaneswar, and Varanasi.
"""

import os
import math
import random
import pandas as pd
import numpy as np

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
        ],
        "route_templates": [
            ("502", "Mehrauli to Kashmere Gate", ["Saket District Centre", "AIIMS Metro / Hospital", "Connaught Place (Central)", "Kashmere Gate ISBT"]),
            ("534", "Anand Vihar to Nehru Place", ["Anand Vihar ISBT", "Sarai Kale Khan ISBT", "Nehru Place Terminal"]),
            ("TMS-Ring", "Mudrika Outer Ring Road", ["Dhaula Kuan Interchange", "AIIMS Metro / Hospital", "Nehru Place Terminal", "Sarai Kale Khan ISBT", "Anand Vihar ISBT", "Kashmere Gate ISBT", "Dhaula Kuan Interchange"]),
            ("711", "Janakpuri to Sarai Kale Khan", ["Dwarka Sector 21", "Dhaula Kuan Interchange", "AIIMS Metro / Hospital", "Sarai Kale Khan ISBT"]),
            ("729", "Dwarka to Kashmere Gate", ["Dwarka Sector 21", "Dhaula Kuan Interchange", "Karol Bagh Metro", "Kashmere Gate ISBT"]),
            ("419", "Badarpur to Kashmere Gate", ["Nehru Place Terminal", "Lajpat Nagar Ring Road", "Kashmere Gate ISBT"]),
            ("502-Dup", "Mehrauli to Old Delhi (Parallel)", ["Saket District Centre", "AIIMS Metro / Hospital", "Connaught Place (Central)", "Kashmere Gate ISBT"]),
            ("427", "Nehru Place to AIIMS Feeder", ["Nehru Place Terminal", "Lajpat Nagar Ring Road", "AIIMS Metro / Hospital"]),
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
            ("Bellandur EcoSpace", 12.9260, 77.6762),
            ("Tin Factory KR Puram", 12.9972, 77.6692),
        ],
        "route_templates": [
            ("500-D", "Hebbal to Silk Board (ORR)", ["Hebbal Flyover", "Tin Factory KR Puram", "Marathahalli Bridge", "Bellandur EcoSpace", "Silk Board Junction"]),
            ("335-E", "Majestic to ITPL Whitefield", ["Kempegowda Bus Station (Majestic)", "Indiranagar 100ft Road", "Marathahalli Bridge", "ITPL Whitefield"]),
            ("356-C", "Majestic to Electronic City", ["Kempegowda Bus Station (Majestic)", "Jayanagar 4th Block", "Silk Board Junction", "Electronic City Tollgate"]),
            ("201-G", "Banashankari to Indiranagar", ["Banashankari TTMC", "Jayanagar 4th Block", "Silk Board Junction", "Indiranagar 100ft Road"]),
            ("500-D-Dup", "Hebbal to Silk Board Redundant", ["Hebbal Flyover", "Tin Factory KR Puram", "Marathahalli Bridge", "Silk Board Junction"]),
            ("215-H", "Majestic to Banashankari", ["Kempegowda Bus Station (Majestic)", "Jayanagar 4th Block", "Banashankari TTMC"]),
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
            ("Ghatkopar Station", 19.0860, 72.9090),
            ("Kurla West Station", 19.0657, 72.8794),
            ("Colaba Electric House", 18.9150, 72.8270),
            ("Thane Station West", 19.1860, 72.9759),
        ],
        "route_templates": [
            ("AS-4", "CSMT to BKC Express", ["CSMT Railway Terminus", "Dadar Western Station", "Bandra Kurla Complex (BKC)"]),
            ("332", "Andheri West to Kurla Depot", ["Andheri West Station", "Bandra Kurla Complex (BKC)", "Kurla West Station"]),
            ("C-40", "Colaba to Ghatkopar Express", ["Colaba Electric House", "CSMT Railway Terminus", "Dadar Western Station", "Ghatkopar Station"]),
            ("707-LTD", "Borivali to Dadar Highway", ["Borivali West Depot", "Andheri West Station", "Bandra Kurla Complex (BKC)", "Dadar Western Station"]),
            ("83", "Colaba to Thane Central", ["Colaba Electric House", "Dadar Western Station", "Ghatkopar Station", "Thane Station West"]),
            ("332-Dup", "Andheri to Kurla Parallel", ["Andheri West Station", "Bandra Kurla Complex (BKC)", "Kurla West Station"]),
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
            ("Kukatpally KPHB", 17.4933, 78.3995),
            ("Ameerpet Metro Junction", 17.4375, 78.4482),
        ],
        "route_templates": [
            ("10H", "Secunderabad to Hitec City", ["Secunderabad Railway Station", "Ameerpet Metro Junction", "Hitec City Cyber Towers"]),
            ("216", "Mehdipatnam to Gachibowli", ["Mehdipatnam Depot", "Gachibowli Stadium"]),
            ("127K", "Koti to Kukatpally", ["Koti Bus Station", "Ameerpet Metro Junction", "Kukatpally KPHB"]),
            ("10H-Dup", "Secunderabad to Hitec City Redundant", ["Secunderabad Railway Station", "Ameerpet Metro Junction", "Hitec City Cyber Towers"]),
            ("9X", "Charminar to Secunderabad", ["Charminar Old City", "Koti Bus Station", "Secunderabad Railway Station"]),
            ("222", "Koti to Gachibowli Corridor", ["Koti Bus Station", "Mehdipatnam Depot", "Gachibowli Stadium"]),
        ]
    },
    "Pune": {
        "agency_name": "Pune Mahanagar Parivahan Mahamandal Ltd (PMPML)",
        "center": (18.5204, 73.8567),
        "landmarks": [
            ("Swargate Bus Station", 18.5018, 73.8586),
            ("Pune Railway Station", 18.5284, 73.8744),
            ("Hinjawadi IT Park", 18.5913, 73.7389),
            ("Katraj Snake Park / Depot", 18.4552, 73.8576),
            ("Kothrud Depot", 18.5074, 73.8077),
            ("Hadapsar Gadital", 18.5020, 73.9280),
            ("Shivajinagar Bus Stand", 18.5314, 73.8446),
            ("Nigdi Pradhikaran", 18.6540, 73.7740),
        ],
        "route_templates": [
            ("115", "Swargate to Hinjawadi IT Park", ["Swargate Bus Station", "Shivajinagar Bus Stand", "Hinjawadi IT Park"]),
            ("204", "Katraj to Pune Station", ["Katraj Snake Park / Depot", "Swargate Bus Station", "Pune Railway Station"]),
            ("100", "Kothrud to Hadapsar", ["Kothrud Depot", "Shivajinagar Bus Stand", "Pune Railway Station", "Hadapsar Gadital"]),
            ("115-Dup", "Swargate to Hinjawadi Parallel", ["Swargate Bus Station", "Shivajinagar Bus Stand", "Hinjawadi IT Park"]),
            ("42", "Katraj to Nigdi Corridor", ["Katraj Snake Park / Depot", "Swargate Bus Station", "Shivajinagar Bus Stand", "Nigdi Pradhikaran"]),
        ]
    },
    "Chennai": {
        "agency_name": "Metropolitan Transport Corporation (MTC Chennai)",
        "center": (13.0827, 80.2707),
        "landmarks": [
            ("Chennai Central Station", 13.0836, 80.2755),
            ("T. Nagar Bus Terminus", 13.0418, 80.2337),
            ("Koyambedu CMBT", 13.0694, 80.1948),
            ("Guindy Kathipara Junction", 13.0067, 80.2025),
            ("Tambaram Sanatorium", 12.9279, 80.1277),
            ("Thiruvanmiyur OMR Junction", 12.9830, 80.2594),
            ("Adyar Depot", 13.0012, 80.2565),
            ("Broadway Bus Stand", 13.0880, 80.2870),
        ],
        "route_templates": [
            ("21G", "Broadway to Tambaram Express", ["Broadway Bus Stand", "Chennai Central Station", "Guindy Kathipara Junction", "Tambaram Sanatorium"]),
            ("19B", "T. Nagar to OMR IT Corridor", ["T. Nagar Bus Terminus", "Adyar Depot", "Thiruvanmiyur OMR Junction"]),
            ("570", "CMBT to OMR Kelambakkam", ["Koyambedu CMBT", "Guindy Kathipara Junction", "Thiruvanmiyur OMR Junction"]),
            ("21G-Dup", "Broadway to Tambaram Parallel", ["Broadway Bus Stand", "Guindy Kathipara Junction", "Tambaram Sanatorium"]),
            ("47A", "CMBT to Thiruvanmiyur", ["Koyambedu CMBT", "T. Nagar Bus Terminus", "Thiruvanmiyur OMR Junction"]),
        ]
    },
    "Kolkata": {
        "agency_name": "West Bengal Transport Corporation (WBTC)",
        "center": (22.5726, 88.3639),
        "landmarks": [
            ("Howrah Railway Station", 22.5850, 88.3426),
            ("Esplanade Bus Terminus", 22.5645, 88.3524),
            ("Salt Lake Sector V IT Hub", 22.5802, 88.4357),
            ("Gariahat Market", 22.5186, 88.3670),
            ("Ultadanga Station / VIP Road", 22.5975, 88.3965),
            ("Kolkata Airport Gate 1", 22.6547, 88.4467),
            ("Jadavpur 8B Bus Stand", 22.4988, 88.3718),
        ],
        "route_templates": [
            ("S-12", "Howrah to Salt Lake Sector V", ["Howrah Railway Station", "Esplanade Bus Terminus", "Salt Lake Sector V IT Hub"]),
            ("AC-43", "Airport to Gariahat Corridor", ["Kolkata Airport Gate 1", "Ultadanga Station / VIP Road", "Salt Lake Sector V IT Hub", "Gariahat Market"]),
            ("VS-1", "Howrah to Airport Express", ["Howrah Railway Station", "Ultadanga Station / VIP Road", "Kolkata Airport Gate 1"]),
            ("S-12-Dup", "Howrah to Sector V Redundant", ["Howrah Railway Station", "Esplanade Bus Terminus", "Salt Lake Sector V IT Hub"]),
            ("47B", "Jadavpur to Esplanade Feeder", ["Jadavpur 8B Bus Stand", "Gariahat Market", "Esplanade Bus Terminus"]),
        ]
    },
    "Ahmedabad": {
        "agency_name": "Ahmedabad Municipal Transport Service (AMTS & Janmarg)",
        "center": (23.0225, 72.5714),
        "landmarks": [
            ("Kalupur Railway Station", 23.0270, 72.6012),
            ("Lal Darwaja Terminal", 23.0232, 72.5815),
            ("IIM Ahmedabad / Vastrapur", 23.0317, 72.5298),
            ("Sarkhej Sanand Crossroads", 22.9850, 72.4960),
            ("Ranip Bus Terminal", 23.0760, 72.5730),
            ("Maninagar Station", 22.9980, 72.6020),
            ("SG Highway Iscon Cross Road", 23.0270, 72.5080),
        ],
        "route_templates": [
            ("BRTS-1", "Ranip to Maninagar Janmarg", ["Ranip Bus Terminal", "Lal Darwaja Terminal", "Maninagar Station"]),
            ("401", "Kalupur to SG Highway Iscon", ["Kalupur Railway Station", "Lal Darwaja Terminal", "IIM Ahmedabad / Vastrapur", "SG Highway Iscon Cross Road"]),
            ("13-D", "Lal Darwaja to Sarkhej", ["Lal Darwaja Terminal", "IIM Ahmedabad / Vastrapur", "Sarkhej Sanand Crossroads"]),
            ("401-Dup", "Kalupur to Iscon Parallel", ["Kalupur Railway Station", "IIM Ahmedabad / Vastrapur", "SG Highway Iscon Cross Road"]),
        ]
    },
    "Jaipur": {
        "agency_name": "Jaipur City Transport Service Ltd (JCTSL)",
        "center": (26.9124, 75.7873),
        "landmarks": [
            ("Jaipur Central Railway Station", 26.9196, 75.7878),
            ("Sindhi Camp Central Bus Stand", 26.9234, 75.7997),
            ("Badi Chaupar Old City", 26.9240, 75.8267),
            ("Malviya Nagar Gaurav Tower", 26.8530, 75.8050),
            ("Mansarovar Metro Terminal", 26.8620, 75.7600),
            ("Sitapura Industrial Area", 26.7820, 75.8340),
        ],
        "route_templates": [
            ("9A", "Sindhi Camp to Sitapura", ["Sindhi Camp Central Bus Stand", "Malviya Nagar Gaurav Tower", "Sitapura Industrial Area"]),
            ("1", "Mansarovar to Badi Chaupar", ["Mansarovar Metro Terminal", "Jaipur Central Railway Station", "Sindhi Camp Central Bus Stand", "Badi Chaupar Old City"]),
            ("9A-Dup", "Sindhi Camp to Sitapura Parallel", ["Sindhi Camp Central Bus Stand", "Malviya Nagar Gaurav Tower", "Sitapura Industrial Area"]),
            ("11", "Jaipur Junction to Malviya Nagar", ["Jaipur Central Railway Station", "Sindhi Camp Central Bus Stand", "Malviya Nagar Gaurav Tower"]),
        ]
    },
    "Surat": {
        "agency_name": "Surat Sitilink BRTS",
        "center": (21.1702, 72.8311),
        "landmarks": [
            ("Surat Railway Station", 21.2045, 72.8407),
            ("Athwa Gate Junction", 21.1860, 72.8120),
            ("Adajan Patiya Hub", 21.1980, 72.7950),
            ("Udhna Darwaja", 21.1750, 72.8380),
            ("Sachin GIDC Terminal", 21.0850, 72.8750),
            ("Varachha Diamond Market", 21.2180, 72.8640),
        ],
        "route_templates": [
            ("BRTS-10", "Surat Station to Sachin GIDC", ["Surat Railway Station", "Udhna Darwaja", "Sachin GIDC Terminal"]),
            ("21", "Adajan to Varachha Diamond Hub", ["Adajan Patiya Hub", "Athwa Gate Junction", "Surat Railway Station", "Varachha Diamond Market"]),
            ("BRTS-10D", "Station to Sachin Parallel", ["Surat Railway Station", "Udhna Darwaja", "Sachin GIDC Terminal"]),
        ]
    },
    "Lucknow": {
        "agency_name": "Lucknow City Transport Services (LCTSL)",
        "center": (26.8467, 80.9462),
        "landmarks": [
            ("Charbagh Railway Station", 26.8316, 80.9234),
            ("Hazratganj Main Market", 26.8530, 80.9450),
            ("Alambagh Bus Terminal", 26.8120, 80.9020),
            ("Gomti Nagar Patrakarpuram", 26.8520, 80.9980),
            ("Munshipulia Metro Station", 26.8920, 80.9920),
            ("Chowk Medical College", 26.8680, 80.9160),
        ],
        "route_templates": [
            ("11", "Charbagh to Gomti Nagar", ["Charbagh Railway Station", "Hazratganj Main Market", "Gomti Nagar Patrakarpuram"]),
            ("25", "Alambagh to Munshipulia", ["Alambagh Bus Terminal", "Charbagh Railway Station", "Hazratganj Main Market", "Munshipulia Metro Station"]),
            ("11-Dup", "Charbagh to Gomti Nagar Parallel", ["Charbagh Railway Station", "Hazratganj Main Market", "Gomti Nagar Patrakarpuram"]),
            ("33", "Chowk to Gomti Nagar Corridor", ["Chowk Medical College", "Hazratganj Main Market", "Gomti Nagar Patrakarpuram"]),
        ]
    },
    "Kochi": {
        "agency_name": "Kochi Metro Rail & KSRTC City",
        "center": (9.9312, 76.2673),
        "landmarks": [
            ("Aluva Railway / Metro Hub", 10.1076, 76.3516),
            ("Edappally Toll / LuLu Mall", 10.0260, 76.3080),
            ("Kaloor Bus Stand", 9.9940, 76.2910),
            ("Ernakulam South Station", 9.9670, 76.2890),
            ("Fort Kochi Vypeen Ferry", 9.9650, 76.2420),
            ("Infopark Kakkanad", 10.0100, 76.3620),
        ],
        "route_templates": [
            ("K-1", "Aluva to Ernakulam South", ["Aluva Railway / Metro Hub", "Edappally Toll / LuLu Mall", "Kaloor Bus Stand", "Ernakulam South Station"]),
            ("K-7", "Fort Kochi to Infopark", ["Fort Kochi Vypeen Ferry", "Ernakulam South Station", "Kaloor Bus Stand", "Infopark Kakkanad"]),
            ("K-1D", "Aluva to South Redundant", ["Aluva Railway / Metro Hub", "Edappally Toll / LuLu Mall", "Ernakulam South Station"]),
        ]
    },
    "Chandigarh": {
        "agency_name": "Chandigarh Transport Undertaking (CTU)",
        "center": (30.7333, 76.7794),
        "landmarks": [
            ("ISBT Sector 17", 30.7410, 76.7820),
            ("ISBT Sector 43", 30.7180, 76.7480),
            ("PGIMER Hospital", 30.7630, 76.7760),
            ("Chandigarh Railway Station", 30.7020, 76.8280),
            ("IT Park Kishangarh", 30.7280, 76.8420),
            ("Manimajra Housing Board", 30.7220, 76.8350),
        ],
        "route_templates": [
            ("CTU-2", "ISBT 43 to IT Park", ["ISBT Sector 43", "ISBT Sector 17", "Manimajra Housing Board", "IT Park Kishangarh"]),
            ("CTU-8", "ISBT 17 to PGIMER Hospital", ["ISBT Sector 17", "PGIMER Hospital"]),
            ("CTU-2D", "ISBT 43 to IT Park Duplicate", ["ISBT Sector 43", "ISBT Sector 17", "IT Park Kishangarh"]),
            ("CTU-15", "Chandigarh Station to PGIMER", ["Chandigarh Railway Station", "ISBT Sector 17", "PGIMER Hospital"]),
        ]
    },
    "Bhopal": {
        "agency_name": "Bhopal City Link Ltd (BCLL MyBus)",
        "center": (23.2599, 77.4126),
        "landmarks": [
            ("Bhopal Junction Station", 23.2650, 77.4110),
            ("Habibganj / Rani Kamlapati Station", 23.2180, 77.4370),
            ("New Market Roshanpura", 23.2350, 77.3990),
            ("Misrod / Mandideep Link", 23.1650, 77.4720),
            ("Bairagarh Sant Hirdaram", 23.2750, 77.3420),
        ],
        "route_templates": [
            ("TR-1", "Bairagarh to Rani Kamlapati", ["Bairagarh Sant Hirdaram", "New Market Roshanpura", "Habibganj / Rani Kamlapati Station"]),
            ("SR-2", "Bhopal Junction to Misrod", ["Bhopal Junction Station", "Habibganj / Rani Kamlapati Station", "Misrod / Mandideep Link"]),
            ("TR-1D", "Bairagarh to Kamlapati Parallel", ["Bairagarh Sant Hirdaram", "New Market Roshanpura", "Habibganj / Rani Kamlapati Station"]),
        ]
    },
    "Indore": {
        "agency_name": "Atal Indore City Transport (AiCTSL)",
        "center": (22.7196, 75.8577),
        "landmarks": [
            ("Indore Central Railway Station", 22.7180, 75.8670),
            ("Rajwada Palace Chowk", 22.7190, 75.8570),
            ("Palasia Geeta Bhawan", 22.7230, 75.8850),
            ("Vijay Nagar Square BRTS", 22.7530, 75.8950),
            ("Silicon City Rau", 22.6450, 75.8230),
            ("Aurobindo Medical Hospital", 22.7880, 75.8560),
        ],
        "route_templates": [
            ("iBus-1", "Aurobindo to Rajiv Gandhi / Rau", ["Aurobindo Medical Hospital", "Vijay Nagar Square BRTS", "Palasia Geeta Bhawan", "Silicon City Rau"]),
            ("M-11", "Railway Station to Vijay Nagar", ["Indore Central Railway Station", "Palasia Geeta Bhawan", "Vijay Nagar Square BRTS"]),
            ("iBus-1D", "BRTS Parallel Corridor", ["Aurobindo Medical Hospital", "Vijay Nagar Square BRTS", "Palasia Geeta Bhawan", "Silicon City Rau"]),
        ]
    },
    "Nagpur": {
        "agency_name": "Nagpur Mahanagar Parivahan (Aapli Bus)",
        "center": (21.1458, 79.0882),
        "landmarks": [
            ("Nagpur Railway Station", 21.1520, 79.0880),
            ("Sitabuldi Interchange", 21.1440, 79.0830),
            ("Dharampeth Traffic Park", 21.1420, 79.0620),
            ("MIHAN SEZ Metro Terminal", 21.0420, 79.0560),
            ("Kamptee Cantonment", 21.2220, 79.1980),
        ],
        "route_templates": [
            ("AB-5", "Kamptee to Sitabuldi Central", ["Kamptee Cantonment", "Nagpur Railway Station", "Sitabuldi Interchange"]),
            ("AB-20", "Sitabuldi to MIHAN SEZ", ["Sitabuldi Interchange", "Dharampeth Traffic Park", "MIHAN SEZ Metro Terminal"]),
            ("AB-20D", "Sitabuldi to MIHAN Redundant", ["Sitabuldi Interchange", "MIHAN SEZ Metro Terminal"]),
        ]
    },
    "Visakhapatnam": {
        "agency_name": "Andhra Pradesh State Road Transport (APSRTC Vizag)",
        "center": (17.6868, 83.2185),
        "landmarks": [
            ("RTC Complex Dwaraka Nagar", 17.7240, 83.3030),
            ("Visakhapatnam Railway Station", 17.7210, 83.2920),
            ("Gajuwaka Industrial Hub", 17.6980, 83.1850),
            ("Madhurawada IT SEZ", 17.8180, 83.3550),
            ("RK Beach INS Kurusura", 17.7160, 83.3320),
        ],
        "route_templates": [
            ("28A", "RTC Complex to Gajuwaka", ["RTC Complex Dwaraka Nagar", "Visakhapatnam Railway Station", "Gajuwaka Industrial Hub"]),
            ("222", "RTC Complex to Madhurawada", ["RTC Complex Dwaraka Nagar", "Madhurawada IT SEZ"]),
            ("28A-Dup", "RTC Complex to Gajuwaka Parallel", ["RTC Complex Dwaraka Nagar", "Visakhapatnam Railway Station", "Gajuwaka Industrial Hub"]),
        ]
    },
    "Coimbatore": {
        "agency_name": "Tamil Nadu State Transport (TNSTC Coimbatore)",
        "center": (11.0168, 76.9558),
        "landmarks": [
            ("Gandhipuram Central Bus Stand", 11.0180, 76.9680),
            ("Coimbatore Junction Station", 10.9980, 76.9620),
            ("Ukkadam Bus Stand", 10.9880, 76.9550),
            ("Peelamedu Airport / Tidel Park", 11.0280, 77.0250),
            ("Singanallur Bus Terminal", 10.9980, 77.0280),
        ],
        "route_templates": [
            ("1C", "Gandhipuram to Tidel Park", ["Gandhipuram Central Bus Stand", "Peelamedu Airport / Tidel Park"]),
            ("3", "Ukkadam to Singanallur", ["Ukkadam Bus Stand", "Coimbatore Junction Station", "Singanallur Bus Terminal"]),
            ("1C-Dup", "Gandhipuram to Tidel Park Parallel", ["Gandhipuram Central Bus Stand", "Peelamedu Airport / Tidel Park"]),
        ]
    },
    "Patna": {
        "agency_name": "Bihar State Road Transport Corp (BSRTC Patna)",
        "center": (25.5941, 85.1376),
        "landmarks": [
            ("Patna Junction Station", 25.6020, 85.1370),
            ("Gandhi Maidan Bus Stand", 25.6200, 85.1480),
            ("Boring Road Crossing", 25.6180, 85.1180),
            ("Danapur Cantonment Station", 25.6280, 85.0450),
            ("Kankarbagh Colony", 25.5920, 85.1580),
        ],
        "route_templates": [
            ("111", "Gandhi Maidan to Danapur Station", ["Gandhi Maidan Bus Stand", "Boring Road Crossing", "Danapur Cantonment Station"]),
            ("222", "Patna Junction to Kankarbagh", ["Patna Junction Station", "Kankarbagh Colony"]),
            ("111-Dup", "Gandhi Maidan to Danapur Parallel", ["Gandhi Maidan Bus Stand", "Boring Road Crossing", "Danapur Cantonment Station"]),
        ]
    },
    "Guwahati": {
        "agency_name": "Assam State Transport Corporation (ASTC Guwahati)",
        "center": (26.1445, 91.7362),
        "landmarks": [
            ("Paltan Bazaar Railway Station", 26.1820, 91.7520),
            ("ISBT Betkuchi", 26.1180, 91.7350),
            ("Jalukbari Gauhati University", 26.1550, 91.6620),
            ("Dispur Capital Complex", 26.1420, 91.7910),
            ("Khanapara Assam-Meghalaya Gate", 26.1180, 91.8210),
        ],
        "route_templates": [
            ("G-1", "Paltan Bazaar to Khanapara", ["Paltan Bazaar Railway Station", "Dispur Capital Complex", "Khanapara Assam-Meghalaya Gate"]),
            ("G-5", "Jalukbari to ISBT Betkuchi", ["Jalukbari Gauhati University", "ISBT Betkuchi"]),
            ("G-1D", "Paltan Bazaar to Khanapara Parallel", ["Paltan Bazaar Railway Station", "Dispur Capital Complex", "Khanapara Assam-Meghalaya Gate"]),
        ]
    },
    "Thiruvananthapuram": {
        "agency_name": "KSRTC SWIFT Thiruvananthapuram",
        "center": (8.5241, 76.9366),
        "landmarks": [
            ("Thampanoor Central Bus Station", 8.4880, 76.9520),
            ("East Fort Bus Terminal", 8.4830, 76.9480),
            ("Technopark Kazhakkoottam", 8.5580, 76.8810),
            ("Pattom Junction", 8.5280, 76.9420),
            ("Kovalam Beach Junction", 8.4020, 76.9780),
        ],
        "route_templates": [
            ("TP-1", "East Fort to Technopark IT Park", ["East Fort Bus Terminal", "Thampanoor Central Bus Station", "Pattom Junction", "Technopark Kazhakkoottam"]),
            ("KV-2", "East Fort to Kovalam Tourist", ["East Fort Bus Terminal", "Kovalam Beach Junction"]),
            ("TP-1D", "East Fort to Technopark Parallel", ["East Fort Bus Terminal", "Thampanoor Central Bus Station", "Technopark Kazhakkoottam"]),
        ]
    },
    "Bhubaneswar": {
        "agency_name": "Capital Region Urban Transport (CRUT Mo Bus)",
        "center": (20.2961, 85.8245),
        "landmarks": [
            ("Master Canteen Railway Station", 20.2680, 85.8420),
            ("Baramunda Bus Stand", 20.2780, 85.7950),
            ("Patia Infocity IT Hub", 20.3550, 85.8180),
            ("AIIMS Bhubaneswar Hospital", 20.2320, 85.7760),
            ("Khandagiri Caves Square", 20.2580, 85.7820),
        ],
        "route_templates": [
            ("Mo-10", "Master Canteen to Patia Infocity", ["Master Canteen Railway Station", "Baramunda Bus Stand", "Patia Infocity IT Hub"]),
            ("Mo-18", "Baramunda to AIIMS Hospital", ["Baramunda Bus Stand", "Khandagiri Caves Square", "AIIMS Bhubaneswar Hospital"]),
            ("Mo-10D", "Master Canteen to Infocity Parallel", ["Master Canteen Railway Station", "Patia Infocity IT Hub"]),
        ]
    },
    "Varanasi": {
        "agency_name": "Uttar Pradesh State Road Transport (UPSRTC Varanasi)",
        "center": (25.3176, 82.9739),
        "landmarks": [
            ("Varanasi Cantt Railway Station", 25.3280, 82.9860),
            ("Godowlia Chowk / Kashi Vishwanath", 25.3110, 83.0080),
            ("Banaras Hindu University (BHU)", 25.2670, 82.9910),
            ("Sarnath Archaeological Terminal", 25.3780, 83.0220),
            ("Babatpur Airport Road", 25.4520, 82.8580),
        ],
        "route_templates": [
            ("V-1", "Cantt Station to BHU Campus", ["Varanasi Cantt Railway Station", "Godowlia Chowk / Kashi Vishwanath", "Banaras Hindu University (BHU)"]),
            ("V-8", "Cantt Station to Sarnath", ["Varanasi Cantt Railway Station", "Sarnath Archaeological Terminal"]),
            ("V-1D", "Cantt to BHU Duplicate Corridor", ["Varanasi Cantt Railway Station", "Banaras Hindu University (BHU)"]),
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

    profile_assignments = [
        "Overcrowded", "Underutilized", "Balanced", "Balanced", "Overcrowded", "Underutilized"
    ]

    for r_idx, (r_code, r_desc, path_stops) in enumerate(route_templates):
        route_id = f"RT_{city_name[:3].upper()}_{r_code}"
        profile = profile_assignments[r_idx % len(profile_assignments)]

        color_hex = "EF4444" if profile == "Overcrowded" else ("10B981" if profile == "Balanced" else "0284C7")
        routes_data.append({
            "route_id": route_id,
            "agency_id": config["agency_name"],
            "route_short_name": r_code,
            "route_long_name": r_desc,
            "route_type": 3,
            "route_color": color_hex,
            "route_text_color": "FFFFFF"
        })

        if profile == "Overcrowded":
            num_trips_wd = 24
        elif profile == "Underutilized":
            num_trips_wd = 30
        else:
            num_trips_wd = 40

        shape_id = f"SHP_{route_id}"
        shape_pt_seq = 1

        for stop_name in path_stops:
            s_lat = df_stops.loc[df_stops["stop_name"] == stop_name, "stop_lat"].iloc[0]
            s_lon = df_stops.loc[df_stops["stop_name"] == stop_name, "stop_lon"].iloc[0]
            shapes_data.append({
                "shape_id": shape_id,
                "shape_pt_lat": s_lat,
                "shape_pt_lon": s_lon,
                "shape_pt_sequence": shape_pt_seq
            })
            shape_pt_seq += 1

        start_hour = 6
        trip_intervals_mins = max(14, int((16 * 60) / num_trips_wd))

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

            current_minute = dep_hour * 60 + dep_min
            for s_seq, stop_name in enumerate(path_stops, 1):
                s_id = stop_name_to_id[stop_name]
                if s_seq > 1:
                    prev_stop = path_stops[s_seq - 2]
                    p_lat = df_stops.loc[df_stops["stop_name"] == prev_stop, "stop_lat"].iloc[0]
                    p_lon = df_stops.loc[df_stops["stop_name"] == prev_stop, "stop_lon"].iloc[0]
                    c_lat = df_stops.loc[df_stops["stop_name"] == stop_name, "stop_lat"].iloc[0]
                    c_lon = df_stops.loc[df_stops["stop_name"] == stop_name, "stop_lon"].iloc[0]
                    dist = haversine_dist_km(p_lat, p_lon, c_lat, c_lon)
                    travel_time = max(5, int((dist / 18.0) * 60 + random.randint(1, 3)))
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

    print(f"Generated {city_name} GTFS: {len(df_routes)} routes, {len(df_stops)} stops.")

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    sample_dir = os.path.join(base_dir, "data", "sample_gtfs")

    for city_name, config in CITY_CONFIGS.items():
        city_slug = city_name.lower().replace(" ", "_")
        city_output_dir = os.path.join(sample_dir, city_slug)
        create_gtfs_for_city(city_name, config, city_output_dir)

if __name__ == "__main__":
    main()
