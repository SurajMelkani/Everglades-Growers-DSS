# 🌾 EAA Growers DSS: Spatial Decision Support for Soil Health

Interactive decision-support system predicting soil organic matter, subsidence risk, and management guidance based on spatial selection within the Everglades Agricultural Area (EAA).

**[Launch Live Decision Support System](https://eaa-growers-dss.streamlit.app/)**

---

### 🎓 Academic Context
This Decision Support System (DSS) is a specialized research tool developed as part of doctoral research at the University of Florida within the Department of Soil, Water, and Ecosystem Sciences. It serves as a digital twin to broader investigations into the drivers of soil carbon stability in South Florida, utilizing machine learning and spectroscopy approaches.

> **🚨 NOTE ON DEMONSTRATION MODE**
> The core predictive machine learning models driving this architecture were trained on a rigorous library of 700+ EAA Histosol samples. Because this research is currently pending academic publication, the live prediction engine has been temporarily replaced with a representative baseline model. The fully calibrated `.joblib` model and `.csv` datasets will be integrated upon publication.

---

### 🛠️ Technical Core & Methodology
The platform integrates several advanced components into a single interactive interface:

* **Spatial Predictive Modeling:** Utilizes a coordinate-based model architecture to estimate Soil Organic Matter dynamics and Soil Depth across the diverse landscape of the EAA.
* **Carbon Dynamics Logic:** The backend contains mathematical models designed to calculate carbon sequestration potential and CO₂ emission offsets based on specific crop selections, soil depths, and historical profiles.
* **Management Protocol Engine:** Provides guidance driven by a diagnostic logic that interprets soil health indicators (such as pH and SOM levels) to suggest targeted Best Management Practices (BMPs).
* **Precision Geography:** Leverages `GeoPandas` and `Folium` for high-fidelity spatial selection, enabling growers to draw specific field boundaries for localized, aggregated analysis.

---

### 🧪 Future Research Integration 
Future iterations of this DSS are designed to expand the predictive loop by integrating high-resolution earth observation data. This includes incorporating multispectral imagery to further refine spatial subsidence risk and soil health indicators.
