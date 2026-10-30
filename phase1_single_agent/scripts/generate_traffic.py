#!/usr/bin/env python3
"""
scripts/generate_traffic.py
Generates realistic mixed traffic demand matching Pokhara urban conditions:
  - Motorcycles / Scooters: 50% (agile, short, fast acceleration)
  - Passenger Cars / Taxis: 40% (standard urban cars)
  - Buses / Microbuses:    10% (public transit, long, slower acceleration)
Total simulation period: 3600 seconds (1 hour)
Medium demand: ~700 vehicles/hour (period ~5.14s)
"""

import os
import sys
import subprocess
import random
import xml.etree.ElementTree as ET

def generate_traffic(net_file, output_routes, end_time=3600, period=5.14, seed=42):
    random.seed(seed)
    sumo_home = os.environ.get("SUMO_HOME", "/usr/share/sumo")
    random_trips_script = os.path.join(sumo_home, "tools", "randomTrips.py")

    if not os.path.exists(random_trips_script):
        raise FileNotFoundError(f"randomTrips.py not found at {random_trips_script}")

    temp_trips = output_routes + ".trips.tmp.xml"

    print(f"[generate_traffic] Running randomTrips.py on {net_file} (duration: {end_time}s, period: {period}s)...")
    cmd = [
        sys.executable,
        random_trips_script,
        "-n", net_file,
        "-o", temp_trips,
        "-r", output_routes,
        "-e", str(end_time),
        "-p", str(period),
        "--seed", str(seed),
        "--validate"
    ]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0:
        print("[generate_traffic] Error running randomTrips.py:", res.stderr)
        raise RuntimeError("randomTrips failed")

    # Now parse the generated routes file and prepend heterogeneous vehicle types
    # and assign types according to the 50/40/10 distribution
    tree = ET.parse(output_routes)
    root = tree.getroot()

    # Define heterogeneous vehicle types
    vtypes = [
        ET.Element("vType", {
            "id": "motorcycle",
            "vClass": "motorcycle",
            "length": "2.2",
            "width": "0.8",
            "maxSpeed": "16.67",
            "accel": "2.8",
            "decel": "4.5",
            "sigma": "0.5",
            "guiShape": "motorcycle",
            "color": "255,215,0"  # Gold / Yellow
        }),
        ET.Element("vType", {
            "id": "passenger",
            "vClass": "passenger",
            "length": "4.5",
            "width": "1.8",
            "maxSpeed": "13.89",
            "accel": "2.0",
            "decel": "4.5",
            "sigma": "0.5",
            "guiShape": "passenger",
            "color": "30,144,255"  # Dodger Blue
        }),
        ET.Element("vType", {
            "id": "bus",
            "vClass": "bus",
            "length": "9.5",
            "width": "2.5",
            "maxSpeed": "11.11",
            "accel": "1.2",
            "decel": "3.5",
            "sigma": "0.5",
            "guiShape": "bus",
            "color": "220,20,60"   # Crimson
        })
    ]

    # Insert vTypes at the beginning of the routes file
    for vt in reversed(vtypes):
        root.insert(0, vt)

    # Distribute types across vehicles: 50% motorcycle, 40% passenger, 10% bus
    type_choices = ["motorcycle"] * 50 + ["passenger"] * 40 + ["bus"] * 10

    veh_count = 0
    type_counts = {"motorcycle": 0, "passenger": 0, "bus": 0}
    for elem in root.findall("vehicle"):
        chosen_type = random.choice(type_choices)
        elem.attrib["type"] = chosen_type
        type_counts[chosen_type] += 1
        veh_count += 1

    tree.write(output_routes, encoding="utf-8", xml_declaration=True)

    # Clean up temp trips file
    if os.path.exists(temp_trips):
        os.remove(temp_trips)

    print(f"[generate_traffic] Successfully generated {veh_count} vehicles:")
    print(f"  - Motorcycles: {type_counts['motorcycle']} ({type_counts['motorcycle']/max(1,veh_count)*100:.1f}%)")
    print(f"  - Passenger:   {type_counts['passenger']} ({type_counts['passenger']/max(1,veh_count)*100:.1f}%)")
    print(f"  - Buses:       {type_counts['bus']} ({type_counts['bus']/max(1,veh_count)*100:.1f}%)")
    return output_routes

if __name__ == "__main__":
    net = sys.argv[1] if len(sys.argv) > 1 else "phase1_single_agent/data/paschimanchal.net.xml"
    out = sys.argv[2] if len(sys.argv) > 2 else "phase1_single_agent/data/routes.rou.xml"
    generate_traffic(net, out)
