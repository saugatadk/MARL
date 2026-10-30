#!/usr/bin/env python3
"""
scripts/fetch_osm_map.py
Fetches OpenStreetMap (OSM) map data for IOE Pashchimanchal Campus (WRC),
Lamachaur Chowk & Baglung-Pokhara Highway corridor, Pokhara, Nepal.

Coordinates (Bounding Box):
  - Min Longitude: 83.9680
  - Min Latitude:  28.2480
  - Max Longitude: 83.9820
  - Max Latitude:  28.2620
"""

import os
import sys
import argparse
import urllib.request
import urllib.error
import xml.etree.ElementTree as ET

DEFAULT_BBOX = (83.9680, 28.2480, 83.9820, 28.2620)
DEFAULT_OUTPUT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "paschimanchal.osm"
)

OVERPASS_ENDPOINTS = [
    "https://api.openstreetmap.org/api/0.6/map?bbox={min_lon},{min_lat},{max_lon},{max_lat}",
    "https://overpass-api.de/api/interpreter?data=[out:xml][timeout:60];(node({min_lat},{min_lon},{max_lat},{max_lon});way({min_lat},{min_lon},{max_lat},{max_lon});relation({min_lat},{min_lon},{max_lat},{max_lon}););(._;>;);out%20body;",
    "https://lz4.overpass-api.de/api/interpreter?data=[out:xml][timeout:60];(node({min_lat},{min_lon},{max_lat},{max_lon});way({min_lat},{min_lon},{max_lat},{max_lon});relation({min_lat},{min_lon},{max_lat},{max_lon}););(._;>;);out%20body;",
]

def fetch_osm_data(bbox=DEFAULT_BBOX, output_path=DEFAULT_OUTPUT, force=False):
    min_lon, min_lat, max_lon, max_lat = bbox

    if os.path.exists(output_path) and not force:
        file_size = os.path.getsize(output_path)
        if file_size > 10000:
            print(f"[fetch_osm_map] File already exists ({file_size} bytes): {output_path}")
            print("[fetch_osm_map] Use --force to re-download.")
            return True

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    print(f"[fetch_osm_map] Requesting OSM bounding box: min_lon={min_lon}, min_lat={min_lat}, max_lon={max_lon}, max_lat={max_lat}")

    headers = {
        "User-Agent": "MajorProject-WRC-TrafficRL/1.0 (ioe.paschimanchal.traffic.rl)"
    }

    success = False
    for endpoint_template in OVERPASS_ENDPOINTS:
        url = endpoint_template.format(
            min_lon=min_lon, min_lat=min_lat, max_lon=max_lon, max_lat=max_lat
        )
        endpoint_name = url.split("?")[0]
        print(f"[fetch_osm_map] Attempting endpoint: {endpoint_name}...")

        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=60) as resp:
                content = resp.read()

                # Basic validation
                if len(content) < 500:
                    print(f"[fetch_osm_map] Warning: Response too short ({len(content)} bytes), skipping...")
                    continue

                try:
                    root = ET.fromstring(content)
                    if root.tag != "osm":
                        print(f"[fetch_osm_map] Warning: Root element is '{root.tag}', expected 'osm', skipping...")
                        continue
                except ET.ParseError as pe:
                    print(f"[fetch_osm_map] XML parse error from {endpoint_name}: {pe}, skipping...")
                    continue

                with open(output_path, "wb") as f:
                    f.write(content)

                node_count = len(root.findall("node"))
                way_count = len(root.findall("way"))
                print(f"[fetch_osm_map] Successfully downloaded and saved to: {output_path}")
                print(f"[fetch_osm_map] OSM Entities: {node_count} nodes, {way_count} ways ({len(content):,} bytes)")
                success = True
                break

        except urllib.error.URLError as e:
            print(f"[fetch_osm_map] Network error with {endpoint_name}: {e}")
        except Exception as e:
            print(f"[fetch_osm_map] Error with {endpoint_name}: {e}")

    if not success:
        if os.path.exists(output_path) and os.path.getsize(output_path) > 1000:
            print(f"[fetch_osm_map] Download failed but existing valid file found at {output_path}. Continuing.")
            return True
        else:
            raise RuntimeError(f"[fetch_osm_map] Failed to download OSM data from all endpoints and no existing file at {output_path}.")

    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download OSM map for IOE Pashchimanchal Campus, Pokhara")
    parser.add_argument("--bbox", nargs=4, type=float, default=DEFAULT_BBOX,
                        metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"),
                        help="Bounding box: min_lon min_lat max_lon max_lat")
    parser.add_argument("--output", "-o", type=str, default=DEFAULT_OUTPUT,
                        help="Output .osm file path")
    parser.add_argument("--force", "-f", action="store_true",
                        help="Force re-download even if file exists")
    args = parser.parse_args()

    fetch_osm_data(bbox=tuple(args.bbox), output_path=args.output, force=args.force)
