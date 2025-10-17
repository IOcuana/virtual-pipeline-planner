# Virtual Pipeline Planner

A Streamlit app for planning **compressed gas transport** (CNG or Hydrogen) via modular road-train combinations.

This planner models **payload capacity**, **fleet utilisation**, **driver compliance**, **infrastructure and transport costs**, and **carbon-credit benefits**.  
It computes the **Levelized Cost ($/GJ)** and **Levelized Benefit ($/GJ)** for complete virtual pipeline scenarios.

---

## 🚀 Run online (recommended)
The easiest way is to open the hosted version on **Streamlit Cloud**:

👉 [https://iocuana-virtual-pipeline-planner.streamlit.app](https://iocuana-virtual-pipeline-planner.streamlit.app)

*(If the link doesn’t load yet, wait 1-2 minutes for the initial build to complete.)*

---

## 💻 Run locally
If you prefer to run it on your own machine:

```bash
# Clone this repo
git clone https://github.com/IOcuana/virtual-pipeline-planner.git
cd virtual-pipeline-planner

# Create a virtual environment
python -m venv .venv
.\.venv\Scripts\activate

# Install requirements
python -m pip install -r requirements.txt

# Run the app
streamlit run cng_planner_app_updated_v13_9.py
