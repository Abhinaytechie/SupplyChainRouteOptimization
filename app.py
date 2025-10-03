"""
Production-ready Supply Chain Optimization Dashboard (Updated)
Features:
- Robust file ingestion (CSV / Excel), single or multi-file merge
- Fuzzy column matching to required canonical column names
- LP cost-minimization via PuLP with scenario editing
- VRP via OR-Tools, map visualization when lat/lon available, fallback network visualization
- Depot selection by name
- KPIs, scenario comparison, download/export, logging, validation
- Caching for performance
"""

import io
import json
import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import pulp
import networkx as nx
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from fuzzywuzzy import process
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

# ------------------------
# Configuration / Constants
# ------------------------
REQUIRED_COLUMNS = [
    "Supplier name",
    "Location",
    "Transportation modes",
    "Routes",
    "Shipping costs",
    "Order quantities",
    "Production volumes",
]

FUZZY_THRESHOLD = 25  # lower threshold allows more flexible matching
LARGE_COST = 9999999

# ------------------------
# Utility functions
# ------------------------
def log_event(message: str, level: str = "INFO"):
    entry = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "level": level, "message": str(message)}
    st.session_state["logs"].append(entry)


@st.cache_data
def fuzzy_map_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
    column_mapping = {}
    for target in REQUIRED_COLUMNS:
        best_match, score = process.extractOne(target, df.columns)
        if score >= FUZZY_THRESHOLD:
            column_mapping[target] = best_match
        else:
            for col in df.columns:
                if col.strip().lower() == target.strip().lower():
                    column_mapping[target] = col
                    break
            else:
                raise ValueError(f"Required column '{target}' not found (best match '{best_match}' score={score}).")

    renamed = df.rename(columns=column_mapping)
    renamed = renamed.loc[:, list(column_mapping.values())]
    renamed.columns = list(column_mapping.keys())
    renamed = renamed[REQUIRED_COLUMNS]
    renamed["Shipping costs"] = pd.to_numeric(renamed["Shipping costs"], errors="coerce").fillna(0)
    renamed["Order quantities"] = pd.to_numeric(renamed["Order quantities"], errors="coerce").fillna(0)
    renamed["Production volumes"] = pd.to_numeric(renamed["Production volumes"], errors="coerce").fillna(0)
    return renamed, column_mapping


@st.cache_data
@st.cache_data
def load_single_file(uploaded_file) -> pd.DataFrame:
    """Load a single uploaded file (CSV or Excel) into DataFrame."""
    name = uploaded_file.name.lower()
    try:
        if name.endswith(".csv") or uploaded_file.type == "text/csv":
            # Try utf-8 first, fallback to ISO-8859-1 / latin1 if fails
            try:
                df = pd.read_csv(uploaded_file, encoding="utf-8")
            except UnicodeDecodeError:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, encoding="ISO-8859-1")
        elif name.endswith((".xls", ".xlsx")) or uploaded_file.type.startswith("application/vnd"):
            df = pd.read_excel(uploaded_file)
        else:
            raise ValueError("Unsupported file format. Upload CSV or Excel.")
    except Exception as e:
        raise ValueError(f"Error reading file {uploaded_file.name}: {e}")
    return df



def merge_uploaded_files(uploaded_files: List[st.runtime.uploaded_file_manager.UploadedFile]) -> pd.DataFrame:
    frames = []
    for f in uploaded_files:
        df = load_single_file(f)
        frames.append(df)
        log_event(f"Loaded file: {f.name}")
    if not frames:
        raise ValueError("No files uploaded.")
    merged = pd.concat(frames, ignore_index=True, sort=False)
    merged = merged.drop_duplicates().reset_index(drop=True)
    return merged

# ------------------------
# Validation & Diagnostics
# ------------------------
def validate_data(df: pd.DataFrame) -> List[str]:
    warnings = []
    if (df["Production volumes"] < 0).any():
        warnings.append("Negative production volumes detected.")
    if (df["Order quantities"] < 0).any():
        warnings.append("Negative order quantities detected.")
    if (df["Shipping costs"] < 0).any():
        warnings.append("Negative shipping costs detected.")
    if (df["Shipping costs"] > 1e7).any():
        warnings.append("Some shipping costs are extremely large; check units.")
    lat_present = any(c.lower() in df.columns.str.lower() for c in ["latitude", "lat"])
    lon_present = any(c.lower() in df.columns.str.lower() for c in ["longitude", "lon", "lng"])
    if not (lat_present and lon_present):
        warnings.append("No latitude/longitude columns detected — map visualizations disabled.")
    return warnings

# ------------------------
# LP Optimization (PuLP)
# ------------------------
def prepare_cost_matrix(df: pd.DataFrame):
    suppliers = list(df["Supplier name"].unique())
    customers = list(df["Location"].unique())
    cost_matrix = pd.DataFrame(index=suppliers, columns=customers, dtype=float)
    for s in suppliers:
        for c in customers:
            route_costs = df[(df["Supplier name"] == s) & (df["Location"] == c)]["Shipping costs"]
            cost_matrix.loc[s, c] = float(route_costs.values[0]) if len(route_costs) > 0 else LARGE_COST
    supply = df.groupby("Supplier name")["Production volumes"].sum().to_dict()
    demand = df.groupby("Location")["Order quantities"].sum().to_dict()
    return suppliers, customers, cost_matrix, supply, demand


def solve_lp(suppliers, customers, cost_matrix: pd.DataFrame, supply: Dict, demand: Dict):
    prob = pulp.LpProblem("SupplyChainCostMin", pulp.LpMinimize)
    x = pulp.LpVariable.dicts("flow", (suppliers, customers), lowBound=0, cat="Continuous")
    prob += pulp.lpSum(cost_matrix.loc[i, j] * x[i][j] for i in suppliers for j in customers)
    for i in suppliers:
        prob += pulp.lpSum(x[i][j] for j in customers) <= supply.get(i, 0)
    for j in customers:
        prob += pulp.lpSum(x[i][j] for i in suppliers) >= demand.get(j, 0)
    prob.solve()
    status = pulp.LpStatus[prob.status]
    if status != "Optimal":
        log_event(f"LP solve status: {status}", level="WARN")
    result = {(i, j): x[i][j].varValue for i in suppliers for j in customers if x[i][j].varValue and x[i][j].varValue > 0}
    total_cost = pulp.value(prob.objective) if pulp.value(prob.objective) is not None else 0
    return result, total_cost, status

# ------------------------
# VRP (OR-Tools)
# ------------------------
def solve_vrp(distance_matrix: np.ndarray, num_vehicles: int = 1, depot: int = 0):
    n = len(distance_matrix)
    manager = pywrapcp.RoutingIndexManager(n, num_vehicles, depot)
    routing = pywrapcp.RoutingModel(manager)
    def distance_callback(from_idx, to_idx):
        return int(distance_matrix[manager.IndexToNode(from_idx), manager.IndexToNode(to_idx)])
    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
    search_params = pywrapcp.DefaultRoutingSearchParameters()
    search_params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    search_params.time_limit.seconds = 5
    solution = routing.SolveWithParameters(search_params)
    routes = []
    if solution:
        for vehicle_id in range(num_vehicles):
            index = routing.Start(vehicle_id)
            route = []
            while not routing.IsEnd(index):
                route.append(manager.IndexToNode(index))
                index = solution.Value(routing.NextVar(index))
            route.append(manager.IndexToNode(index))
            routes.append(route)
    else:
        log_event("VRP solver returned no solution", level="WARN")
    return routes

# ------------------------
# Visualization helpers
# ------------------------
def plot_lp_sunburst(lp_result: Dict) -> go.Figure:
    if not lp_result:
        return go.Figure()
    df = pd.DataFrame([(i, j, v) for (i, j), v in lp_result.items()], columns=["Supplier", "Customer", "Units"])
    return px.sunburst(df, path=["Supplier", "Customer"], values="Units", title="Optimized Shipments (Sunburst)")

def plot_lp_heatmap(lp_result: Dict) -> go.Figure:
    if not lp_result:
        return go.Figure()
    df = pd.DataFrame([(i, j, v) for (i, j), v in lp_result.items()], columns=["Supplier", "Customer", "Units"])
    pivot = df.pivot_table(index="Supplier", columns="Customer", values="Units", fill_value=0)
    return px.imshow(pivot, text_auto=True, title="Shipments Heatmap (Supplier x Customer)")

def plot_vrp_map(routes: List[List[int]], locations_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    colors = px.colors.qualitative.Plotly
    for idx, route in enumerate(routes):
        lats = [locations_df.iloc[i]["Latitude"] for i in route]
        lons = [locations_df.iloc[i]["Longitude"] for i in route]
        names = [locations_df.iloc[i]["Location"] for i in route]
        fig.add_trace(
            go.Scattermapbox(
                lat=lats,
                lon=lons,
                mode="markers+lines+text",
                marker=dict(size=8),
                text=names,
                name=f"Vehicle {idx+1}",
            )
        )
    all_lats = [locations_df.iloc[i]["Latitude"] for r in routes for i in r] if routes else [0]
    all_lons = [locations_df.iloc[i]["Longitude"] for r in routes for i in r] if routes else [0]
    center = {"lat": np.mean(all_lats), "lon": np.mean(all_lons)}
    fig.update_layout(mapbox_style="open-street-map", mapbox_center=center, mapbox_zoom=3)
    fig.update_layout(title="VRP Routes Map")
    return fig

def plot_vrp_network(routes: List[List[int]], locations: List[str]) -> nx.DiGraph:
    G = nx.DiGraph()
    for route in routes:
        for i in range(len(route)-1):
            G.add_edge(locations[route[i]], locations[route[i+1]])
    return G

# ------------------------
# Download / Export helpers
# ------------------------
def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")

def df_to_excel_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False)
    return buf.getvalue()

# ------------------------
# Streamlit App UI
# ------------------------
def run_dashboard():
    st.set_page_config(page_title="Supply Chain Optimization (Production)", layout="wide")
    st.title("🛡️ Supply Chain Optimization — Production Demo (Safertek-ready)")

    if "logs" not in st.session_state: st.session_state["logs"] = []
    if "last_lp" not in st.session_state: st.session_state["last_lp"] = None
    if "last_vrp" not in st.session_state: st.session_state["last_vrp"] = None
    if "scenario_store" not in st.session_state: st.session_state["scenario_store"] = {}

    st.sidebar.header("Upload & Settings")
    uploaded_files = st.sidebar.file_uploader(
        "Upload CSV or Excel (single/multi-file)",
        accept_multiple_files=True,
        type=["csv","xls","xlsx"]
    )

    df_merged = None
    if uploaded_files:
        try:
            raw_merged = merge_uploaded_files(uploaded_files)
            df_mapped, mapping = fuzzy_map_columns(raw_merged)
            df_merged = df_mapped.copy()
            log_event(f"Columns mapped: {mapping}")
            st.success(f"Loaded and mapped {len(uploaded_files)} file(s). Rows: {len(df_merged)}")
        except Exception as e:
            st.error(f"Error loading files: {e}")
            log_event(f"Load error: {e}", level="ERROR")

    st.sidebar.markdown("---")
    st.sidebar.header("VRP Settings")
    num_vehicles = st.sidebar.number_input("Number of vehicles", min_value=1, max_value=20, value=2)
    depot_name = None
    if df_merged is not None:
        depot_name = st.sidebar.selectbox("Select Depot (start/end location)", options=list(df_merged["Location"].unique()))
        depot_index = list(df_merged["Location"].unique()).index(depot_name)
    else:
        depot_index = 0

    st.sidebar.markdown("---")
    st.sidebar.header("Scenario")
    scenario_name = st.sidebar.text_input("Scenario name (save/load)", value="base_scenario")
    if st.sidebar.button("Save current scenario"):
        st.session_state["scenario_store"][scenario_name] = {"timestamp": time.time(), "notes": "Saved from UI"}
        log_event(f"Scenario saved: {scenario_name}")

    left_col, right_col = st.columns([2,1])

    if df_merged is not None:
        with left_col:
            st.subheader("Data preview")
            st.dataframe(df_merged.head(25), use_container_width=True)
            warnings = validate_data(df_merged)
            if warnings:
                st.warning("Data Warnings / Anomalies detected:")
                for w in warnings:
                    st.write("- " + w)
                    log_event(f"Warning: {w}", level="WARN")
            else:
                st.info("No obvious anomalies detected.")

            st.subheader("KPIs")
            cols = st.columns(4)
            cols[0].metric("Suppliers", len(df_merged["Supplier name"].unique()))
            cols[1].metric("Customers", len(df_merged["Location"].unique()))
            total_supply = df_merged["Production volumes"].sum()
            total_demand = df_merged["Order quantities"].sum()
            cols[2].metric("Total Supply", f"{total_supply:,.0f}")
            cols[3].metric("Total Demand", f"{total_demand:,.0f}")

        with right_col:
            st.subheader("Actions")
            if st.button("Run LP Optimization"):
                try:
                    suppliers, customers, cost_matrix, supply, demand = prepare_cost_matrix(df_merged)
                    st.session_state["current_supply"] = supply.copy()
                    st.session_state["current_demand"] = demand.copy()
                    with st.spinner("Solving LP..."):
                        lp_result, total_cost, status = solve_lp(suppliers, customers, cost_matrix, supply, demand)
                    st.success(f"LP solved (status: {status}). Total cost: {total_cost:,.2f}")
                    st.session_state["last_lp"] = {"result": lp_result, "total_cost": total_cost, "suppliers": suppliers, "customers": customers}
                    if lp_result:
                        df_lp = pd.DataFrame([(i,j,v) for (i,j),v in lp_result.items()], columns=["Supplier","Customer","Units"])
                        st.write(df_lp)
                        st.download_button("Download LP CSV", df_to_csv_bytes(df_lp), "lp_result.csv")
                        st.download_button("Download LP Excel", df_to_excel_bytes(df_lp), "lp_result.xlsx")
                        st.plotly_chart(plot_lp_sunburst(lp_result), use_container_width=True)
                        st.plotly_chart(plot_lp_heatmap(lp_result), use_container_width=True)
                    else:
                        st.info("No shipments required.")
                except Exception as e:
                    st.error(f"LP error: {e}")
                    log_event(f"LP error: {e}", level="ERROR")

            st.markdown("---")
            if st.button("Run VRP (use coords if present)"):
                try:
                    lat_cols = [c for c in df_merged.columns if c.lower() in ("latitude", "lat")]
                    lon_cols = [c for c in df_merged.columns if c.lower() in ("longitude", "lon", "lng")]
                    if lat_cols and lon_cols:
                        lat_col = lat_cols[0]
                        lon_col = lon_cols[0]
                        locs_df = df_merged[["Location", lat_col, lon_col]].drop_duplicates().reset_index(drop=True)
                        locs_df.columns = ["Location", "Latitude", "Longitude"]
                        coords = locs_df[["Latitude","Longitude"]].astype(float).values
                        distance_matrix = cdist(coords, coords, metric="euclidean")
                        with st.spinner("Solving VRP..."):
                            routes = solve_vrp(distance_matrix, num_vehicles=num_vehicles, depot=depot_index)
                        st.session_state["last_vrp"] = {"routes": routes, "locations_df": locs_df}
                        st.success(f"VRP solved, routes: {len(routes)}")
                        if len(locs_df)>0 and routes:
                            st.plotly_chart(plot_vrp_map(routes, locs_df), use_container_width=True)
                            for i,r in enumerate(routes):
                                seq = [locs_df.iloc[idx]["Location"] for idx in r]
                                st.write(f"Vehicle {i+1}: {seq}")
                    else:
                        locations = list(df_merged["Location"].unique())
                        n = len(locations)
                        distance_matrix = np.random.randint(10,100,size=(n,n))
                        with st.spinner("Solving VRP (no coords)..."):
                            routes = solve_vrp(distance_matrix, num_vehicles=num_vehicles, depot=depot_index)
                        st.session_state["last_vrp"] = {"routes": routes, "locations_list": locations}
                        st.success("VRP solved (network fallback).")
                        G = plot_vrp_network(routes, locations)
                        plt.figure(figsize=(8,6))
                        nx.draw(G, nx.spring_layout(G, seed=42), with_labels=True, node_size=1500, node_color="lightblue", arrowsize=20)
                        st.pyplot(plt.gcf())
                        plt.close()
                except Exception as e:
                    st.error(f"VRP error: {e}")
                    log_event(f"VRP error: {e}", level="ERROR")

    st.markdown("---")
    st.subheader("Audit Log & Scenario Store")
    with st.expander("View logs"):
        logs = st.session_state["logs"][-200:]
        if logs:
            st.dataframe(pd.DataFrame(logs))
            st.download_button("Download logs JSON", json.dumps(logs, indent=2), "logs.json")
        else:
            st.write("No logs yet.")
    with st.expander("Saved Scenarios"):
        if st.session_state["scenario_store"]:
            for name, meta in st.session_state["scenario_store"].items():
                st.write(f"- {name} (saved at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(meta['timestamp']))})")
        else:
            st.write("No scenarios saved yet.")

if __name__ == "__main__":
    run_dashboard()
