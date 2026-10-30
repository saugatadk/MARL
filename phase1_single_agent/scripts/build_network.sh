#!/usr/bin/env bash
# ==============================================================================
# build_network.sh
# End-to-end network build script for IOE Pashchimanchal Campus, Pokhara
# Converts OSM data, injects traffic light logic, and generates mixed vehicle routes.
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(dirname "$SCRIPT_DIR")"
DATA_DIR="$BASE_DIR/data"

export SUMO_HOME="${SUMO_HOME:-/usr/share/sumo}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

# Prefer virtual environment python if available
if [[ -f "$BASE_DIR/../venv/bin/python3" ]]; then
    PYTHON_BIN="$BASE_DIR/../venv/bin/python3"
fi

echo "=========================================================="
echo " Building SUMO Road Network: IOE Pashchimanchal Campus"
echo " SUMO_HOME: $SUMO_HOME"
echo " PYTHON_BIN: $PYTHON_BIN"
echo " DATA_DIR:  $DATA_DIR"
echo "=========================================================="

mkdir -p "$DATA_DIR"

# 1. Fetch OSM data if not present
if [[ ! -f "$DATA_DIR/paschimanchal.osm" || ! -s "$DATA_DIR/paschimanchal.osm" ]]; then
    echo "[Step 1/5] Fetching OpenStreetMap data via API..."
    "$PYTHON_BIN" "$SCRIPT_DIR/fetch_osm_map.py" --output "$DATA_DIR/paschimanchal.osm"
else
    echo "[Step 1/5] paschimanchal.osm already exists. Skipping download."
fi

# 2. Run netconvert
echo "[Step 2/5] Running netconvert to generate road network..."
netconvert --osm-files "$DATA_DIR/paschimanchal.osm" \
           --type-files "$DATA_DIR/typemap.xml" \
           --output-file "$DATA_DIR/paschimanchal.net.xml" \
           --geometry.remove \
           --roundabouts.guess \
           --ramps.guess \
           --junctions.join \
           --tls.guess-signals \
           --tls.discard-simple \
           --tls.set 3700742618 \
           --tls.default-type actuated

# 3. Configure primary 4-phase traffic light logic
echo "[Step 3/5] Configuring primary traffic light at Lamachaur Chowk (3700742618)..."
"$PYTHON_BIN" "$SCRIPT_DIR/configure_tls.py" "$DATA_DIR/paschimanchal.net.xml" "3700742618"

# 4. Generate campus building and terrain polygons
echo "[Step 4/5] Generating campus polygons (polyconvert)..."
POLY_TYPEMAP="$SUMO_HOME/data/typemap/osmPolyconvert.typ.xml"
if [[ -f "$POLY_TYPEMAP" ]]; then
    polyconvert --net-file "$DATA_DIR/paschimanchal.net.xml" \
                --osm-files "$DATA_DIR/paschimanchal.osm" \
                --type-file "$POLY_TYPEMAP" \
                -o "$DATA_DIR/paschimanchal.poly.xml" || true
else
    echo "Warning: Poly typemap not found at $POLY_TYPEMAP. Creating empty poly file."
    echo '<additional/>' > "$DATA_DIR/paschimanchal.poly.xml"
fi

# 5. Generate realistic mixed traffic demand (~700 veh/hr)
echo "[Step 5/5] Generating realistic heterogeneous traffic demand (motorcycles 50%, passenger 40%, bus 10%)..."
"$PYTHON_BIN" "$SCRIPT_DIR/generate_traffic.py" \
    "$DATA_DIR/paschimanchal.net.xml" \
    "$DATA_DIR/routes.rou.xml"

echo "=========================================================="
echo " Network build complete!"
echo " Net:   $DATA_DIR/paschimanchal.net.xml"
echo " Poly:  $DATA_DIR/paschimanchal.poly.xml"
echo " Route: $DATA_DIR/routes.rou.xml"
echo " Cfg:   $DATA_DIR/simulation.sumocfg"
echo "=========================================================="
