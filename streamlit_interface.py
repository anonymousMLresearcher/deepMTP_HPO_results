import streamlit as st
import os
import pandas as pd
import pickle
import plotly.graph_objects as go
from scipy.interpolate import interp1d
from scipy.stats import rankdata
import numpy as np


def get_list_of_settings(aggregate_option):
    if aggregate_option == "Classification":
        return ["MLC", "MTL", "DP"]
    elif aggregate_option == "Regression":
        return ["MTR", "MC"]
    else:
        return aggregate_option


def get_ranking(
    base_path="streamlit_cached_data/",
    dataset_option="bibtex",
    metric_option="auroc",
    averaging_option="macro",
    resolution=None,
):
    HPO_names = ["hyperband", "random_search", "2_x_random_search", "SMAC", "BOHB"]

    results_per_dataset = {}
    for mode in ["val", "test"]:
        interpolated_per_HPO_method = {}
        interpolator_per_HPO_method = {}
        data_per_HPO_method = {}
        for hp_name in HPO_names:

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
            temp_data_dict = get_data_to_plot(temp_dir)

            data_per_HPO_method[hp_name] = [
                temp_data_dict["hyperopt_runtime_arr_mean"],
                temp_data_dict["hyperopt_performance_arr_mean"],
                temp_data_dict["hyperopt_performance_arr_min"],
                temp_data_dict["hyperopt_performance_arr_max"],
                temp_data_dict["hyperopt_performance_arr_std"],
            ]

            interpolator_per_HPO_method[hp_name] = interp1d(
                temp_data_dict["hyperopt_runtime_arr_mean"],
                temp_data_dict["hyperopt_performance_arr_mean"],
                bounds_error=False,
                fill_value=(
                    np.nan,
                    temp_data_dict["hyperopt_performance_arr_mean"][-1],
                ),
            )

        # calculate the min and max time points across all HPO methods
        min_x = min([data[0][0] for HPO_name, data in data_per_HPO_method.items()])
        max_x = max([data[0][-1] for HPO_name, data in data_per_HPO_method.items()])
        if resolution is None:
            resolution = int(max_x - min_x)
            print("Calculating maximum resolution: " + str(resolution))
        global_x = np.linspace(min_x, max_x, resolution)

        for hp_name in HPO_names:
            interpolated_per_HPO_method[hp_name] = interpolator_per_HPO_method[hp_name](
                global_x
            )

        raw_ratings_arr = np.array([d for d in interpolated_per_HPO_method.values()])
        if metric_option in ["RRMSE", "RMSE", "MSE", "MAE"]:
            rankings_arr = rankdata(raw_ratings_arr, axis=0, method="min")
        else:
            rankings_arr = rankdata(-1 * raw_ratings_arr, axis=0, method="min")

        rankings_arr = rankings_arr.astype(float)
        rankings_arr[np.isnan(raw_ratings_arr)] = np.nan

        results_per_dataset[mode] = {
            "rankings": rankings_arr,
            "raw_ratings": raw_ratings_arr,
        }

    return results_per_dataset


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

viz_option = st.sidebar.radio(
    "Visualization mode:", ("Individual performance plots", "Aggregated ranking plots")
)

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
    base_path = "streamlit_cached_data/"

    HPO_names = ["hyperband", "random_search", "2_x_random_search", "SMAC", "BOHB"]
    info_per_dataset = [
        {
            "MTP_setting": "MLC",
            "dataset_name": "bibtex",
            "metric_name": "aupr",
            "metric_average": "macro",
        },
        {
            "MTP_setting": "MLC",
            "dataset_name": "Corel5k",
            "metric_name": "aupr",
            "metric_average": "macro",
        },
        {
            "MTP_setting": "MTR",
            "dataset_name": "rf2",
            "metric_name": "RRMSE",
            "metric_average": "macro",
        },
        {
            "MTP_setting": "MTR",
            "dataset_name": "scm1d",
            "metric_name": "RRMSE",
            "metric_average": "macro",
        },
        {
            "MTP_setting": "MTL",
            "dataset_name": "dog",
            "metric_name": "aupr",
            "metric_average": "macro",
        },
        {
            "MTP_setting": "MTL",
            "dataset_name": "bird",
            "metric_name": "aupr",
            "metric_average": "macro",
        },
        {
            "MTP_setting": "MC",
            "dataset_name": "movielens_100k",
            "metric_name": "RMSE",
            "metric_average": "micro",
        },
        {
            "MTP_setting": "MC",
            "dataset_name": "movielens_1M",
            "metric_name": "RMSE",
            "metric_average": "micro",
        },
        {
            "MTP_setting": "DP",
            "dataset_name": "ern",
            "metric_name": "aupr",
            "metric_average": "micro",
        },
        {
            "MTP_setting": "DP",
            "dataset_name": "srn",
            "metric_name": "aupr",
            "metric_average": "micro",
        },
    ]

    type_aggregation_option = st.sidebar.radio(
        "Select a type of aggregation", ("Predefined", "Manual")
    )
    list_of_datasets = []
    if type_aggregation_option == "Predefined":

        # aggregate selector
        aggregate_option = st.sidebar.selectbox(
            "Select a aggregation mode",
            ["Classification", "Regression", "MLC", "MTR", "MTL", "MC", "DP"],
        )

        list_of_settings = get_list_of_settings(aggregate_option)

        list_of_datasets = [
            d for d in info_per_dataset if d["MTP_setting"] in list_of_settings
        ]

    else:

        dataset_options = st.multiselect(
            "Select one or more of the datasets",
            [d_info["dataset_name"] for d_info in info_per_dataset],
            ["bibtex", "Corel5k"],
        )

        list_of_datasets = [
            d for d in info_per_dataset if d["dataset_name"] in dataset_options
        ]

    if len(list_of_datasets) != 0:
        result_per_dataset = []
        # get the ranking matrices from every dataset-metric combo
        for dataset in list_of_datasets:

            result_per_dataset.append(
                get_ranking(
                    base_path=base_path,
                    dataset_option=dataset["dataset_name"],
                    metric_option=dataset["metric_name"],
                    averaging_option=dataset["metric_average"],
                    resolution=100,
                )
            )

        # calculate the average ranking (separately for the validation and test sets)
        for mode in ["val", "test"]:
            # this is the idea of the padding option. It doesn't make sense as you lose temporal information. Sampling works better

            ranking_per_dataset = np.array(
                [d[mode]["rankings"] for d in result_per_dataset]
            )
            print("Averaging over " + str(len(ranking_per_dataset)) + " datasets")
            average_ranking = np.nanmean(ranking_per_dataset, axis=0)
            fig = go.Figure()
            fig.update_layout(
                xaxis=dict(title="% of runtime"), yaxis=dict(title="ranking"),
            )

            for i in range(average_ranking.shape[0]):
                nan_ids = np.isnan(average_ranking[i, :])
                fig.add_trace(
                    go.Scatter(
                        x=np.arange(average_ranking.shape[1])[~nan_ids],
                        y=average_ranking[i, ~nan_ids],
                        mode="lines+markers",
                        name=HPO_names[i],
                    )
                )
            st.title(mode)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("""---""")

    else:
        st.warning("No datasets or MTP settings selected")

