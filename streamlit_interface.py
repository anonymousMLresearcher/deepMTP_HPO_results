import streamlit as st
import os
import pandas as pd
import pickle
import plotly.graph_objects as go
from scipy.interpolate import interp1d
from scipy.stats import rankdata
import numpy as np


def get_ranking(
    hyperopt_runtime_arr_mean,
    hyperopt_performance_arr_mean,
    hyperopt_performance_arr_min,
    hyperopt_performance_arr_max,
    hyperopt_performance_arr_std,
):
    interpolated_per_HPO_method = {}
    interpolator_per_HPO_method = {}
    data_per_HPO_method = {}

    data_per_HPO_method[hp_name] = [
        hyperopt_runtime_arr_mean,
        hyperopt_performance_arr_mean,
        hyperopt_performance_arr_min,
        hyperopt_performance_arr_max,
        hyperopt_performance_arr_std,
    ]
    interpolator_per_HPO_method[hp_name] = interp1d(
        hyperopt_runtime_arr_mean, hyperopt_performance_arr_mean, bounds_error=False
    )


@st.experimental_memo
def get_data_to_plot(temp_dir):
    print("accessing plot data")
    hyperopt_runtime_arr_mean = pickle.load(
        open(temp_dir + "/runtime_arr_mean.pkl", "rb",)
    )

    hyperopt_performance_arr_mean = pickle.load(
        open(temp_dir + "/performance_arr_mean.pkl", "rb",)
    )
    hyperopt_performance_arr_min = pickle.load(
        open(temp_dir + "/performance_arr_min.pkl", "rb",)
    )
    hyperopt_performance_arr_max = pickle.load(
        open(temp_dir + "/performance_arr_max.pkl", "rb",)
    )
    hyperopt_performance_arr_std = pickle.load(
        open(temp_dir + "/performance_arr_std.pkl", "rb",)
    )
    return {
        "hyperopt_runtime_arr_mean": hyperopt_runtime_arr_mean,
        "hyperopt_performance_arr_mean": hyperopt_performance_arr_mean,
        "hyperopt_performance_arr_min": hyperopt_performance_arr_min,
        "hyperopt_performance_arr_max": hyperopt_performance_arr_max,
        "hyperopt_performance_arr_std": hyperopt_performance_arr_std,
    }


@st.experimental_memo
def get_dataset_info(temp_dir):
    return pd.read_csv(temp_dir, sep=", ")


def get_file_structure(base_path):
    dir_names_per_depth = [x[0] for x in os.walk(base_path)]

    dir_structure_dict = {}
    for dir in dir_names_per_depth[1:]:
        broken_dir = dir.split("/")[1:]
        if not broken_dir[0] in dir_structure_dict:
            dir_structure_dict[broken_dir[0]] = {}
        if len(broken_dir) > 1:
            if not broken_dir[1] in dir_structure_dict[broken_dir[0]]:
                dir_structure_dict[broken_dir[0]][broken_dir[1]] = {}
            if len(broken_dir) > 2:
                if (
                    not broken_dir[2]
                    in dir_structure_dict[broken_dir[0]][broken_dir[1]]
                ):
                    dir_structure_dict[broken_dir[0]][broken_dir[1]][
                        broken_dir[2]
                    ] = set()
                if len(broken_dir) > 3:
                    dir_structure_dict[broken_dir[0]][broken_dir[1]][broken_dir[2]].add(
                        broken_dir[3]
                    )
    return dir_structure_dict


def show_performance_plots(plot_cols, data_dict):
    idx = 0
    for mode, hp_data in data_dict.items():

        fig = go.Figure()

        # fig.update_layout(
        #    autosize=False, width=2000, height=1000, paper_bgcolor="LightSteelBlue",
        # )

        for hp_name, hp_dict in hp_data.items():
            hyperopt_runtime_arr_mean = hp_dict["raw_data"]["hyperopt_runtime_arr_mean"]
            hyperopt_performance_arr_mean = hp_dict["raw_data"][
                "hyperopt_performance_arr_mean"
            ]
            hyperopt_performance_arr_min = hp_dict["raw_data"][
                "hyperopt_performance_arr_min"
            ]
            hyperopt_performance_arr_max = hp_dict["raw_data"][
                "hyperopt_performance_arr_max"
            ]
            hyperopt_performance_arr_std = hp_dict["raw_data"][
                "hyperopt_performance_arr_std"
            ]

            if visualize_error == "area":
                fig.add_trace(
                    go.Scatter(
                        x=hyperopt_runtime_arr_mean,
                        y=hyperopt_performance_arr_mean,
                        mode="lines+markers",
                        name=hp_name,
                    )
                )
                if error_type == "min_max":
                    fig.add_trace(
                        go.Scatter(
                            x=hyperopt_runtime_arr_mean,
                            y=hyperopt_performance_arr_min,
                            mode="lines",
                            showlegend=False,
                            line=dict(width=0),
                        )
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=hyperopt_runtime_arr_mean,
                            y=hyperopt_performance_arr_max,
                            mode="lines",
                            name=hp_name + "_area",
                            fill="tonexty",
                            showlegend=True,
                            line=dict(width=0),
                        )
                    )

                elif error_type == "std":
                    fig.add_trace(
                        go.Scatter(
                            x=hyperopt_runtime_arr_mean,
                            y=hyperopt_performance_arr_mean
                            - hyperopt_performance_arr_std,
                            mode="lines",
                            showlegend=False,
                            line=dict(width=0),
                        )
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=hyperopt_runtime_arr_mean,
                            y=hyperopt_performance_arr_mean
                            + hyperopt_performance_arr_std,
                            mode="lines",
                            name=hp_name + "_area",
                            fill="tonexty",
                            showlegend=True,
                            line=dict(width=0),
                        )
                    )

        if log_x_axis:
            fig.update_xaxes(
                exponentformat="power", type="log", dtick="D0", tickfont=dict(size=20)
            )

        if log_y_axis:
            fig.update_yaxes(exponentformat="power", type="log")

        fig.update_layout(
            xaxis=dict(title="Wall clock time (seconds)"),
            yaxis=dict(title=metric_option + "_" + averaging_option),
        )

        plot_cols[idx].plotly_chart(fig, use_container_width=True)
        idx += 1


def show_ranking_plots(rank_cols, data_dict, resolution):
    idx = 0
    for mode, hp_data in data_dict.items():
        interpolated_per_HPO_method = {}
        interpolator_per_HPO_method = {}

        for hp_name, hp_dict in hp_data.items():
            hyperopt_runtime_arr_mean = hp_dict["raw_data"]["hyperopt_runtime_arr_mean"]
            hyperopt_performance_arr_mean = hp_dict["raw_data"][
                "hyperopt_performance_arr_mean"
            ]

            interpolator_per_HPO_method[hp_name] = interp1d(
                hyperopt_runtime_arr_mean,
                hyperopt_performance_arr_mean,
                bounds_error=False,
                fill_value=(np.nan, hyperopt_performance_arr_mean[-1]),
            )

        # calculate the min and max time points across all HPO methods
        min_x = min(
            [
                data["raw_data"]["hyperopt_runtime_arr_mean"][0]
                for data in hp_data.values()
            ]
        )
        max_x = max(
            [
                data["raw_data"]["hyperopt_runtime_arr_mean"][-1]
                for data in hp_data.values()
            ]
        )
        global_x = np.linspace(min_x, max_x, resolution)

        for hp_name in hp_data.keys():
            interpolated_per_HPO_method[hp_name] = interpolator_per_HPO_method[hp_name](
                global_x
            )

        raw_ratings_arr = np.array([d for d in interpolated_per_HPO_method.values()])
        if metric_option in [
            "auroc",
            "aupr",
            "f1_score",
            "recall",
            "precision",
            "accuracy",
        ]:
            rankings_arr = rankdata(-1 * raw_ratings_arr, axis=0, method="min")
        else:
            rankings_arr = rankdata(raw_ratings_arr, axis=0, method="min")

        rankings_arr = rankings_arr.astype(float)
        rankings_arr[np.isnan(raw_ratings_arr)] = np.nan

        fig = go.Figure()

        fig.update_layout(
            xaxis=dict(title="Wall clock time (seconds)"), yaxis=dict(title="ranking"),
        )

        for i in range(rankings_arr.shape[0]):
            nan_ids = np.isnan(rankings_arr[i, :])
            fig.add_trace(
                go.Scatter(
                    x=global_x[~nan_ids],
                    y=rankings_arr[i, ~nan_ids],
                    mode="lines+markers",
                    name=list(hp_data.keys())[i],
                )
            )
        rank_cols[idx].plotly_chart(fig, use_container_width=True)
        idx += 1


base_path = "streamlit_cached_data/"
visualize_error = "area"
error_type = "std"
st.set_page_config(layout="wide")

# get a dictionary of dictionaries with the file structure of all the data to be plotted
dir_structure_dict = get_file_structure(base_path)

st.sidebar.title("HPO benchmarks for DeepMTP")

viz_option = st.sidebar.radio("Visualization mode:", ("Individual", "Aggregated"))

if viz_option == "Individual":

    # dataset selector
    dataset_option = st.sidebar.selectbox(
        "Select a dataset", list(dir_structure_dict.keys())
    )

    st.header("Basic Dataset info")
    dataset_info_df = get_dataset_info("dataset_info.csv")

    st.dataframe(dataset_info_df[dataset_info_df["dataset_name"] == dataset_option])

    st.markdown("""---""")

    # performance metric selector, conditioned on the selected dataset
    metric_option = st.sidebar.selectbox(
        "Select a metric", list(dir_structure_dict[dataset_option].keys())
    )

    # performance averaging selector, conditioned on the selected dataset and performance metric
    averaging_option = st.sidebar.selectbox(
        "Averaging method option",
        list(dir_structure_dict[dataset_option][metric_option].keys()),
    )

    # checkboxes for dynamically applying log scaling the two axis
    log_x_axis = st.sidebar.checkbox("Scale x-axis")
    log_y_axis = st.sidebar.checkbox("Scale y-axis")
    # add_x_axis_slider = st.checkbox("Enable custom range on x-axis")
    header_cols = [c for c in st.columns(6)]
    header_cols[1].header("Validation " + metric_option + "_" + averaging_option)
    header_cols[4].header("Test " + metric_option + "_" + averaging_option)
    plot_cols = [c for c in st.columns(2)]

    data_dict = {}
    for idx, mode in enumerate(["val", "test"]):
        data_dict[mode] = {}
        for hp_name in list(
            dir_structure_dict[dataset_option][metric_option][averaging_option]
        ):
            temp_dir = (
                base_path
                + dataset_option
                + "/"
                + metric_option
                + "/"
                + averaging_option
                + "/"
                + hp_name
                + "/"
                + mode
            )

            data_dict[mode][hp_name] = {
                "raw_data": get_data_to_plot(temp_dir),
                "dataset_option": dataset_option,
                "metric_option": metric_option,
                "averaging_option": averaging_option,
            }

    show_performance_plots(plot_cols, data_dict)

    resolution = st.slider("Define the sampling resolution: ", 10, 1000, 100)

    rank_cols = [c for c in st.columns(2)]

    show_ranking_plots(rank_cols, data_dict, resolution)


else:
    st.header("Will be implemented soon...")

