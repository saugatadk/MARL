#!/usr/bin/env python3
"""
scripts/configure_tls.py
Configures the signalized traffic light junction at Lamachaur Chowk (junction 3700742618)
with the required 4-phase scheme:
  - Phase 0: North-South Green (duration: 27s, minDur: 10s)
  - Phase 1: North-South Yellow (duration: 3s)
  - Phase 2: East-West Green   (duration: 27s, minDur: 10s)
  - Phase 3: East-West Yellow   (duration: 3s)
"""

import os
import sys
import xml.etree.ElementTree as ET

def configure_network_tls(net_file_path, junction_id="3700742618"):
    if not os.path.exists(net_file_path):
        raise FileNotFoundError(f"Network file not found: {net_file_path}")

    tree = ET.parse(net_file_path)
    root = tree.getroot()

    # Find or verify junction
    junction = None
    for j in root.findall("junction"):
        if j.attrib.get("id") == junction_id:
            junction = j
            break

    if junction is None:
        print(f"[configure_tls] Warning: Junction '{junction_id}' not found in {net_file_path}")
        return False

    junction.attrib["type"] = "traffic_light"

    # Find connections for this junction to check TLS link count
    tl_links = []
    for conn in root.findall("connection"):
        if conn.attrib.get("tl") == junction_id:
            link_idx = int(conn.attrib.get("linkIndex", -1))
            if link_idx >= 0:
                tl_links.append((link_idx, conn.attrib.get("from"), conn.attrib.get("to")))

    num_links = len(tl_links) if tl_links else 18
    print(f"[configure_tls] Junction '{junction_id}' has {num_links} traffic light connections.")

    # Remove existing tlLogic for this junction if any
    for tl in list(root.findall("tlLogic")):
        if tl.attrib.get("id") == junction_id:
            root.remove(tl)

    # Construct the 4-phase tlLogic
    # North-South: Links 4..7 and 13..17
    # East-West: Links 0..3 and 8..12
    ns_green = ["r"] * num_links
    ns_yellow = ["r"] * num_links
    ew_green = ["r"] * num_links
    ew_yellow = ["r"] * num_links

    # Map links based on connections
    for link_idx, from_edge, to_edge in tl_links:
        # Determine orientation from edge IDs
        # North-South corridor: edges 421588528, 421588525
        # East-West corridor: edges 670034955, 364217805
        if "421588528" in from_edge or "421588525" in from_edge:
            ns_green[link_idx] = "G"
            ns_yellow[link_idx] = "y"
        elif "670034955" in from_edge or "364217805" in from_edge:
            ew_green[link_idx] = "G"
            ew_yellow[link_idx] = "y"
        else:
            # Fallback based on link index
            if 4 <= link_idx <= 7 or 13 <= link_idx <= 17:
                ns_green[link_idx] = "G"
                ns_yellow[link_idx] = "y"
            else:
                ew_green[link_idx] = "G"
                ew_yellow[link_idx] = "y"

    state_ns_green = "".join(ns_green)
    state_ns_yellow = "".join(ns_yellow)
    state_ew_green = "".join(ew_green)
    state_ew_yellow = "".join(ew_yellow)

    # Fallback to standard 18-char string if no connections parsed
    if num_links == 18 and state_ns_green == "r" * 18:
        state_ns_green = "rrrrGGGGrrrrrGGGGG"
        state_ns_yellow = "rrrryyyyrrrrryyyyy"
        state_ew_green = "GGGGrrrrrGGGGGrrrrr"
        state_ew_yellow = "yyyyrrrrryyyyyrrrrr"

    print(f"[configure_tls] Phase 0 (NS Green):  {state_ns_green}")
    print(f"[configure_tls] Phase 1 (NS Yellow): {state_ns_yellow}")
    print(f"[configure_tls] Phase 2 (EW Green):  {state_ew_green}")
    print(f"[configure_tls] Phase 3 (EW Yellow): {state_ew_yellow}")

    # Find insertion point: before first connection element or after junctions
    insert_idx = len(root)
    for idx, child in enumerate(root):
        if child.tag == "connection":
            insert_idx = idx
            break

    tl_elem = ET.Element("tlLogic", {
        "id": junction_id,
        "type": "static",
        "programID": "0",
        "offset": "0"
    })

    ET.SubElement(tl_elem, "phase", {
        "duration": "27",
        "state": state_ns_green,
        "minDur": "10",
        "maxDur": "60"
    })
    ET.SubElement(tl_elem, "phase", {
        "duration": "3",
        "state": state_ns_yellow
    })
    ET.SubElement(tl_elem, "phase", {
        "duration": "27",
        "state": state_ew_green,
        "minDur": "10",
        "maxDur": "60"
    })
    ET.SubElement(tl_elem, "phase", {
        "duration": "3",
        "state": state_ew_yellow
    })

    root.insert(insert_idx, tl_elem)

    tree.write(net_file_path, encoding="utf-8", xml_declaration=True)
    print(f"[configure_tls] Successfully configured 4-phase TLS logic in {net_file_path}")
    return True

if __name__ == "__main__":
    net_path = sys.argv[1] if len(sys.argv) > 1 else "phase1_single_agent/data/paschimanchal.net.xml"
    junc_id = sys.argv[2] if len(sys.argv) > 2 else "3700742618"
    configure_network_tls(net_path, junc_id)
