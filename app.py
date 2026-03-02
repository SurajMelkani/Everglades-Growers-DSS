import streamlit as st
import folium
from folium.plugins import Draw, Geocoder
from streamlit_folium import st_folium
import geopandas as gpd
from shapely.geometry import Point, shape
import numpy as np
import pandas as pd
import os

# --- PAGE CONFIGURATION & STYLING ---
st.set_page_config(page_title="Everglades Growers Decision Support System", page_icon="🌾", layout="wide")

# =====================================================================
# 🚨 PROPRIETARY MODEL PLACEHOLDER 🚨
# The core predictive machine learning models driving this DSS were 
# trained on a rigorous library of 700+ EAA Histosol samples. 
#
# Because this research is currently pending academic publication, the 
# live prediction engine has been temporarily replaced with a randomized 
# baseline model to demonstrate the interface, mass-balance logic, and 
# spatial mapping capabilities. 
#
# FUTURE IMPLEMENTATION (Post-Publication):
# import joblib
# real_model = joblib.load("eaa_spatial_model.joblib") 
# real_data = pd.read_csv("eaa_calibration_data.csv") 
# =====================================================================

st.markdown("""
    <style>
    .step-header {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
        border-left: 5px solid #1E88E5;
        color: #111111;
        font-weight: bold;
        margin-bottom: 20px;
    }
    .summary-box {
        background-color: #f9f9f9;
        border: 1px solid #ddd;
        border-radius: 6px;
        padding: 10px 14px;
        font-size: 14px;
        margin-bottom: 12px;
        color: #111111; 
    }
    .zone-alert {
        background-color: #e3f2fd;
        border-left: 4px solid #1E88E5;
        padding: 10px 14px;
        border-radius: 4px;
        font-size: 14px;
        color: #0d47a1;
        margin-bottom: 15px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- INITIALIZE SESSION STATE ---
defaults = {
    "step": 1,
    "clicked_lat": None,
    "clicked_lon": None,
    "est_som": None,
    "est_depth": None,
    "display_ph": 6.5,
    "display_som": 0.0,
    "saved_farm_size": 100,
    "selected_crop": "Sugarcane",
    "drawn_area_ha": None,
    "selection_mode": "point", 
    "map_key": 0 
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

def reset_survey():
    for k, v in defaults.items():
        if k != "map_key":
            st.session_state[k] = v
    st.session_state.map_key += 1 

def navigation_buttons(back_step=None, forward_fn=None, forward_label="Continue ➡️"):
    col_back, col_next = st.columns(2)
    if back_step is not None:
        if col_back.button("⬅️ Back"):
            st.session_state.step = back_step
            st.rerun()
    if forward_fn is not None:
        if col_next.button(forward_label, type="primary"):
            forward_fn()
            st.rerun()

# --- 1. LOAD SHAPEFILE ---
@st.cache_data(show_spinner="Loading EAA Spatial Data...")
def load_shapefile():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, "EAA Shapefile", "Export_Output.shp")

    if not os.path.exists(file_path):
        st.error(f"❌ Could not find shapefile at `{file_path}`.")
        st.stop()
    try:
        gdf = gpd.read_file(file_path)
        gdf["geometry"] = gdf.geometry.buffer(0)
        gdf = gdf.to_crs(epsg=4326)
        unified_poly = gdf.union_all() if hasattr(gdf, "union_all") else gdf.unary_union
        bounds = unified_poly.bounds
        return gdf, unified_poly, [unified_poly.centroid.y, unified_poly.centroid.x], bounds
    except Exception as e:
        st.error(f"❌ Error loading shapefile: {e}")
        st.stop()

eaa_gdf, eaa_poly, map_center, map_bounds = load_shapefile()

# --- 2. CROP DATA & MODELS ---
crop_data = pd.DataFrame({
    "Crop": ["Sugarcane", "Flooded Rice", "Sweet Corn", "Lettuce",
             "Turf Grass", "Legume Mix", "Sunn Hemp", "Cowpea"],
    "Carbon Credits (tons/ha/yr)": [6.75, 9.12, 4.89, 3.21, 7.43, 5.38, 8.27, 6.12],
    "CO₂ Released (tons/ha/yr)":   [3.42, 0.40, 2.65, 1.92, 2.87, 1.75, 1.48, 1.95],
})

def calculate_carbon_impact(base_credits, farm_size, depth_cm, som_pct):
    depth_factor = min(depth_cm / 30.0, 1.0)
    som_factor   = min(som_pct  / 60.0, 1.0)
    return base_credits * farm_size * depth_factor * som_factor

def get_crop_recommendations(crop):
    recs = {
        "Sugarcane": [
            "🟢 **Rotate with Flooded Rice** → Rice as a cover crop retains soil moisture and organic matter, reducing emissions by ~2.5 tons CO₂/ha per year.",
            "⏳ **Use Fallow Periods** → Keeping fields submerged for 3–4 months before replanting slows organic matter breakdown.",
            "🛑 **Limit Ratoon Cycles to 2** → More ratoons accelerate soil organic matter depletion.",
            "🍂 **Mulch with Sugarcane Residue** → Incorporating trash adds ~1.2 tons of carbon/ha per year.",
        ],
        "Flooded Rice": [
            "🔄 **Rotate with Sugarcane** → Maintains soil health and reduces subsidence by ~3.2 tons CO₂/ha per year.",
            "🛑 **Limit Sugarcane Ratoon Cycles to 2** → Prolonged ratooning leads to excessive soil depletion.",
            "🌾 **Mulch with Sugarcane Residue** → Leftover biomass adds ~1.5 tons of carbon/ha.",
        ],
        "Sweet Corn": [
            "🌱 **Rotate with Legume Cover Crops** → Sunn Hemp or Cowpea fixes nitrogen, saving ~1.5 tons CO₂/ha.",
            "🚜 **Use Minimum Tillage** → Reducing disturbance preserves organic matter.",
            "🌽 **Incorporate Crop Residue** → Leaving corn stalks improves structure and adds ~0.9 tons of carbon/ha.",
        ],
        "Lettuce": [
            "💧 **Rotate with Flooded Rice** → Stabilizes soil moisture and reduces CO₂ loss by ~1 ton/ha.",
            "🥬 **Use Cover Crops** → Sunn Hemp or Ryegrass adds ~1.1 tons of carbon/ha.",
            "🚰 **Drip Irrigation** → Reduces water loss and lowers emissions by ~15%.",
            "🚜 **Minimize Tillage** → Prevents carbon loss and slows organic matter breakdown.",
        ],
    }
    cover_crops = ["Legume Mix", "Sunn Hemp", "Cowpea"]
    if crop in cover_crops:
        return [
            "⚡ **Use Before Sugarcane** → Fixes nitrogen, reducing synthetic fertilizer need by ~30%.",
            "🌼 **Incorporate Before Flowering** → Maximizes nitrogen release into soil.",
            "💧 **Manage Water Table** → High moisture levels further slow CO₂ loss.",
            "📈 **Increase Organic Matter** → Boosts microbial activity and reduces fertilizer dependency.",
        ]
    return recs.get(crop, [
        "💧 **Manage Water Table** → Keeping soil moisture levels high slows organic matter oxidation.",
    ])


# FUTURE IMPLEMENTATION:
# import joblib
# real_model = joblib.load("eaa_spatial_model.joblib") 
# real_data = pd.read_csv("eaa_calibration_data.csv") 
#
# Currently, the function below generates representative randomized 
# values based on coordinates to demonstrate the UI capabilities, 
# mass-balance logic, and spatial mapping of the DSS.
def predict_soil_metrics(lat, lon):
    rng = np.random.default_rng(seed=int((lat + lon) * 10000) % (2**31))
    som = round(rng.uniform(25.0, 85.0), 1)
    
    # Depth Distribution logic (40/50/10 probability)
    depth_choice = rng.choice(['shallow', 'adequate', 'deep'], p=[0.4, 0.5, 0.1])
    
    if depth_choice == 'shallow':
        depth = int(rng.integers(15, 30))
    elif depth_choice == 'adequate':
        depth = int(rng.integers(30, 101))
    else:
        depth = int(rng.integers(101, 151))
        
    return som, depth

# --- 4. UI LAYOUT ---
st.title("🌱 :green[Everglades Growers Decision Support System]")
st.markdown("Maximize your yield while preserving your land. Get field-specific recommendations to slow soil loss, build organic matter, and keep your farm productive.")
st.divider()

col1, col2 = st.columns([1.2, 1], gap="large")

# --- LEFT COLUMN: MAP ---
with col1:
    st.markdown('<div class="step-header">📍 Farm Location & Boundary</div>', unsafe_allow_html=True)
    st.caption("**Instructions:** Click the map to select a point, or use the shape tool (polygon icon) to draw your farm's boundary. Use the search bar to find a specific address.")

    m = folium.Map(location=map_center, zoom_start=10, tiles="CartoDB positron")
    m.fit_bounds([[map_bounds[1], map_bounds[0]], [map_bounds[3], map_bounds[2]]])

    Geocoder(collapsed=False, position='topright', add_marker=True).add_to(m)

    Draw(export=False, draw_options={
        "polygon": True, "rectangle": True, "marker": False,
        "polyline": False, "circle": False, "circlemarker": False,
    }).add_to(m)

    # FIXED: "weight" now has correct quotes to prevent NameError
    folium.GeoJson(eaa_gdf, name="EAA Boundary", style_function=lambda x: {
        "fillColor": "#2c7fb8", "color": "#253494", "weight": 2, "fillOpacity": 0.15,
    }).add_to(m)

    map_data = st_folium(m, width=600, height=500, key=f"map_{st.session_state.map_key}")
    
    drawing_processed = False
    if map_data and map_data.get("all_drawings"):
        last_drawing = map_data["all_drawings"][-1]
        if last_drawing.get("geometry", {}).get("type") in ("Polygon", "MultiPolygon"):
            try:
                drawn_shape = shape(last_drawing["geometry"])
                lon, lat = drawn_shape.centroid.x, drawn_shape.centroid.y
                if st.session_state.clicked_lat != lat:
                    if eaa_poly.contains(Point(lon, lat)):
                        drawn_gdf = gpd.GeoDataFrame(geometry=[drawn_shape], crs="EPSG:4326")
                        area_ha = drawn_gdf.to_crs("EPSG:32617").geometry.area.iloc[0] / 10000
                        st.session_state.drawn_area_ha = round(area_ha, 1)
                        st.session_state.clicked_lat, st.session_state.clicked_lon = lat, lon
                        st.session_state.selection_mode = "polygon"
                        som, depth = predict_soil_metrics(lat, lon)
                        st.session_state.est_som, st.session_state.est_depth = som, depth
                        st.session_state.step = 2
                        st.rerun()
                    else:
                        st.toast("⚠️ Centroid outside EAA boundary.", icon="⚠️")
                drawing_processed = True
            except: pass

    if not drawing_processed and map_data and map_data.get("last_clicked"):
        lat, lon = map_data["last_clicked"]["lat"], map_data["last_clicked"]["lng"]
        if eaa_poly.contains(Point(lon, lat)):
            if st.session_state.clicked_lat != lat:
                st.session_state.clicked_lat, st.session_state.clicked_lon = lat, lon
                st.session_state.drawn_area_ha, st.session_state.selection_mode = None, "point"
                som, depth = predict_soil_metrics(lat, lon)
                st.session_state.est_som, st.session_state.est_depth = som, depth
                st.session_state.step = 2
                st.rerun()
        else:
            st.toast("⚠️ Out of Bounds — click inside the blue EAA boundary.", icon="⚠️")

# --- RIGHT COLUMN: SURVEY STEPS ---
with col2:
    if st.session_state.step == 1:
        st.info("👈 **Awaiting Input:** Click the map or draw your field boundary to begin.")

    elif st.session_state.step == 2:
        st.markdown('<div class="step-header">Step 1: Soil Data Input</div>', unsafe_allow_html=True)
        if st.session_state.selection_mode == "polygon":
            st.markdown(f'<div class="zone-alert">📐 <b>Area Detected: {st.session_state.drawn_area_ha} hectares.</b><br>The spatial model will provide an aggregated baseline for this field area.</div>', unsafe_allow_html=True)
        else:
            st.success(f"✅ **Coordinate Locked:** `{st.session_state.clicked_lat:.4f}, {st.session_state.clicked_lon:.4f}`")

        st.write("Do you have recent soil test results?")
        has_test = st.radio("Test Results:", ["No, run spatial models", "Yes, I will input my data"], index=0, label_visibility="collapsed")

        if has_test == "Yes, I will input my data":
            col_a, col_b = st.columns(2)
            with col_a:
                ph_choice = st.selectbox("Soil pH range:", ["Acidic (Below 5.5)", "Slightly Acidic (5.5–6.5)", "Neutral (6.5–7.5)", "Alkaline (Above 7.5)"])
            with col_b:
                som_choice = st.selectbox("Organic Matter rating:", ["Low (Below 40%)", "Moderate (40%–70%)", "High (Above 70%)"], index=1)

        def go_to_step3():
            if has_test == "Yes, I will input my data":
                ph_map = {"Below": 5.0, "5.5": 6.0, "6.5": 7.0, "Above": 8.0}
                som_map = {"Low": 30.0, "Moderate": 55.0, "High": 77.5}
                pk = next((k for k in ph_map if k in ph_choice), "6.5")
                sk = next((k for k in som_map if k in som_choice), "Moderate")
                st.session_state.display_ph, st.session_state.display_som = ph_map[pk], som_map[sk]
            else:
                st.session_state.display_ph, st.session_state.display_som = 6.5, st.session_state.est_som
            st.session_state.step = 3
        navigation_buttons(back_step=1, forward_fn=go_to_step3, forward_label="Next ➡️")

    elif st.session_state.step == 3:
        st.markdown('<div class="step-header">📊 Step 2: Soil Profile & Health Diagnostics</div>', unsafe_allow_html=True)
        
        som = st.session_state.display_som
        ph = st.session_state.display_ph
        depth = st.session_state.est_depth
        
        if som < 40: som_label, som_color = "⚠ Low Organic Matter", "inverse"
        elif 40 <= som <= 70: som_label, som_color = "✓ Moderate Organic Matter", "normal"
        else: som_label, som_color = "✓ Good Organic Matter", "normal"

        if ph < 5.5: ph_label, ph_color = "⚠ Extremely Low", "inverse"
        elif 5.5 <= ph < 6.5: ph_label, ph_color = "⚠ Low pH", "normal"
        elif 6.5 <= ph <= 7.5: ph_label, ph_color = "✓ Good pH", "normal"
        elif 7.5 < ph <= 8.5: ph_label, ph_color = "⚠ High pH", "normal"
        else: ph_label, ph_color = "⚠ Extremely High", "inverse"

        if depth < 30: depth_label, depth_color = "⚠ Very Low Depth", "inverse"
        elif 30 <= depth <= 100: depth_label, depth_color = "✓ Adequate Depth", "normal"
        else: depth_label, depth_color = "✓ Very High Depth", "normal"

        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("Organic Matter", f"{som}%", delta=som_label, delta_color=som_color)
        mc2.metric("Soil pH", f"{ph}", delta=ph_label, delta_color=ph_color)
        mc3.metric("Soil Depth", f"{depth} cm", delta=depth_label, delta_color=depth_color)
        
        st.divider()
        if depth < 30 or som < 40:
            st.warning("🚨 **Soil Alert:** Management should focus on building organic matter and protecting limited soil depth.")
        else:
            st.success("✅ **Balanced Soil Baseline:** Focus on maintaining current organic matter levels through optimal rotations.")
        navigation_buttons(back_step=2, forward_fn=lambda: setattr(st.session_state, "step", 4), forward_label="Next ➡️")

    elif st.session_state.step == 4:
        st.markdown('<div class="step-header">🚜 Step 3: Crop Planning & Carbon Impact</div>', unsafe_allow_html=True)
        default_size = min(int(st.session_state.drawn_area_ha) if st.session_state.drawn_area_ha else st.session_state.saved_farm_size, 500000)
        input_size = st.number_input("Farm size (hectares):", min_value=1, max_value=500000, value=default_size, step=10)
        crop_decision = st.radio("Crop Selection:", ["I have a crop in mind", "Recommend crops for me"], horizontal=True)

        if crop_decision == "Recommend crops for me":
            st.info("**Top Recommendations for your soil:**\n- **Flooded Rice:** Best for slowing subsidence.\n- **Sunn Hemp:** Rapidly builds organic matter.")
            selected_crop = st.selectbox("Select to proceed:", ["Flooded Rice", "Sweet Corn", "Sunn Hemp", "Cowpea"])
        else:
            selected_crop = st.selectbox("Crop selection:", crop_data["Crop"].tolist())
            with st.expander("🌍 Carbon & Emissions Data", expanded=False):
                display_df = crop_data.copy()
                display_df["Carbon Credits (tons/ha/yr)"] = display_df["Carbon Credits (tons/ha/yr)"].map("{:.2f}".format)
                display_df["CO₂ Released (tons/ha/yr)"] = display_df["CO₂ Released (tons/ha/yr)"].map("{:.2f}".format)
                st.dataframe(display_df.style.apply(lambda r: ["background-color: #e8f5e9; color: #111111;" if r["Crop"] == selected_crop else "" for _ in r], axis=1), hide_index=True, use_container_width=True)
                st.info("💡 **Legend:** 1 Carbon Credit = 1 ton of CO₂ stored. Car average = 4.6 tons CO₂/yr.")

        navigation_buttons(back_step=3, forward_fn=lambda: (setattr(st.session_state, 'selected_crop', selected_crop), setattr(st.session_state, 'saved_farm_size', input_size), setattr(st.session_state, 'step', 5)), forward_label="Yes, plan my crop ➡️")

    elif st.session_state.step == 5:
        crop, size = st.session_state.selected_crop, st.session_state.saved_farm_size
        adj_credits = calculate_carbon_impact(crop_data[crop_data["Crop"] == crop].iloc[0]["Carbon Credits (tons/ha/yr)"], size, st.session_state.est_depth, st.session_state.display_som)
        
        st.markdown(f'<div class="step-header">📋 Step 4: Management Protocols for {crop}</div>', unsafe_allow_html=True)
        with st.expander("📋 Assessment Summary", expanded=True):
            loc = f"{st.session_state.drawn_area_ha} ha Area" if st.session_state.selection_mode == "polygon" else f"Lat: {st.session_state.clicked_lat:.4f}"
            st.markdown(f'<div class="summary-box">📍 <b>Location:</b> {loc}<br>🧪 <b>Organic Matter:</b> {st.session_state.display_som}% | <b>pH:</b> {st.session_state.display_ph} | <b>Depth:</b> {st.session_state.est_depth} cm<br>🌾 <b>Crop:</b> {crop} | <b>Size:</b> {size} ha</div>', unsafe_allow_html=True)

        with st.container(border=True):
            st.markdown("**Location & Crop-Specific Management Practices:**")
            for protocol in get_crop_recommendations(crop): st.markdown(f"- {protocol}")

        cars = (adj_credits / size / 4.6) if size > 0 else 0
        st.info(f"💡 **Understanding Results:** 1 Credit = 1 ton stored CO₂.\n\n🚗 **Impact:** For every **1 hectare**, you offset **{cars:.1f} cars** per year. Total: **{adj_credits:,.0f} tradable Carbon Credits**!")
        
        if st.button("🔄 Start New Assessment", type="primary"): reset_survey(); st.rerun()


