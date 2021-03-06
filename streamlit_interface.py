import streamlit as st
import os
import pandas as pd
import pickle
import plotly.graph_objects as go
from scipy.interpolate import interp1d
from scipy.stats import rankdata
import numpy as np
import matplotlib.pyplot as plt


def get_list_of_settings(aggregate_option):
    if aggregate_option == "Classification":
        return ["MLC", "MTL", "DP"]
    elif aggregate_option == "Regression":
        return ["MTR", "MC"]
    elif aggregate_option == "All datasets":
        return ["MLC", "MTR", "MTL", "MC", "DP"]
    else:
        return aggregate_option


@st.experimental_memo
def get_config_from_txt(MTP_setting_name):
    file = open("config_spaces/" + MTP_setting_name + ".txt", "r")
    config = file.read()
    file.close()
    return config


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
        interpolated_performance_per_HPO_method = {}
        interpolator_per_HPO_method = {}
        data_per_HPO_method = {}
        end_point_per_HPO_method = {}
        raw_performance_arr = None
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
            interpolated_performance_per_HPO_method[
                hp_name
            ] = interpolator_per_HPO_method[hp_name](global_x)
            end_point_per_HPO_method[hp_name] = (
                np.where(data_per_HPO_method[hp_name][0][-1] > global_x)[0][-1] + 1
            )

        raw_performance_arr = np.array(
            [interpolated_performance_per_HPO_method[hp_name] for hp_name in HPO_names]
        )

        if metric_option in ["RRMSE", "RMSE", "MSE", "MAE"]:
            rankings_arr = rankdata(raw_performance_arr, axis=0, method="min")
        else:
            rankings_arr = rankdata(-1 * raw_performance_arr, axis=0, method="min")

        rankings_arr = rankings_arr.astype(float)
        rankings_arr[np.isnan(raw_performance_arr)] = np.nan

        results_per_dataset[mode] = {
            "rankings": rankings_arr,
            "raw_performance_arr": raw_performance_arr,
            "end_points": end_point_per_HPO_method,
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
    return pd.read_csv(temp_dir, sep=", ", engine="python")


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


def show_performance_plots(plot_cols, data_dict, color_per_HPO_method):
    idx = 0
    for mode, hp_data in data_dict.items():

        fig = go.Figure()

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
                        line_color=color_per_HPO_method[hp_name],
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


def show_ranking_plots(rank_cols, data_dict, resolution, color_per_HPO_method):
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
                    line_color=color_per_HPO_method[list(hp_data.keys())[i]],
                )
            )
        rank_cols[idx].plotly_chart(fig, use_container_width=True)
        idx += 1


color_per_HPO_method = {
    "hyperband": "#636EFA",
    "random_search": "#EF553B",
    "2_x_random_search": "#00CC96",
    "SMAC": "#AB63FA",
    "BOHB": "#FFA15A",
}

base_path = "streamlit_cached_data/"
visualize_error = "area"
error_type = "std"
st.set_page_config(layout="wide")

# get a dictionary of dictionaries with the file structure of all the data to be plotted
dir_structure_dict = get_file_structure(base_path)

st.sidebar.title("HPO benchmarks for DeepMTP")

viz_option = st.sidebar.radio(
    "Visualization mode:",
    ("Individual performance plots", "Aggregated ranking plots", "other plots"),
)

if viz_option == "Individual performance plots":

    # dataset selector
    dataset_option = st.sidebar.selectbox(
        "Select a dataset", list(dir_structure_dict.keys())
    )
    col1, col2, col3 = st.columns([4, 1, 3])

    col1.header("Basic Dataset info")
    dataset_info_df = get_dataset_info("dataset_info.csv")
    col1.dataframe(dataset_info_df[dataset_info_df["dataset_name"] == dataset_option])

    col3.header("Hyperparameter space details")
    experiment_config = get_config_from_txt(dataset_option)
    col3.text(experiment_config)
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

    show_performance_plots(plot_cols, data_dict, color_per_HPO_method)

    resolution = st.slider("Define the sampling resolution: ", 10, 1000, 100)

    rank_cols = [c for c in st.columns(2)]

    show_ranking_plots(rank_cols, data_dict, resolution, color_per_HPO_method)


elif viz_option == "Aggregated ranking plots":
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
            [
                "All datasets",
                "Classification",
                "Regression",
                "MLC",
                "MTR",
                "MTL",
                "MC",
                "DP",
            ],
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
                        line_color=color_per_HPO_method[HPO_names[i]],
                    )
                )
                average_end_point = np.mean(
                    [d[mode]["end_points"][HPO_names[i]] for d in result_per_dataset]
                )
                fig.add_vline(
                    x=average_end_point,
                    line_width=3,
                    line_dash="dash",
                    line_color=color_per_HPO_method[HPO_names[i]],
                )
            st.title(mode)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("""---""")

    else:
        st.warning("No datasets or MTP settings selected")


elif viz_option == "other plots":
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
            "metric_name": "RMSE",
            "metric_average": "micro",
        },
        {
            "MTP_setting": "MTR",
            "dataset_name": "scm1d",
            "metric_name": "RMSE",
            "metric_average": "micro",
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
            "metric_average": "macro",
        },
        {
            "MTP_setting": "DP",
            "dataset_name": "srn",
            "metric_name": "aupr",
            "metric_average": "macro",
        },
    ]

    # aggregate selector
    aggregate_option = st.sidebar.selectbox(
        "Select a aggregation mode", ["Classification", "Regression",],
    )

    list_of_settings = get_list_of_settings(aggregate_option)

    list_of_datasets = [
        d for d in info_per_dataset if d["MTP_setting"] in list_of_settings
    ]

    metric_used = set([dataset["metric_name"] for dataset in list_of_datasets])
    averaging_used = set([dataset["metric_average"] for dataset in list_of_datasets])

    st.header(
        "Calculating the difference on "
        + list(averaging_used)[0]
        + "-"
        + list(metric_used)[0]
    )

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

    averaged_average_ranking_per_mode = {}
    # calculate the average ranking (separately for the validation and test sets)
    for mode in ["val", "test"]:
        averaged_average_ranking_per_hpo_method = []
        # this is the idea of the padding option. It doesn't make sense as you lose temporal information. Sampling works better

        ranking_per_dataset = np.array(
            [d[mode]["rankings"] for d in result_per_dataset]
        )
        # st.write("Averaging over " + str(len(ranking_per_dataset)) + " datasets")
        average_ranking = np.nanmean(ranking_per_dataset, axis=0)
        # fig = go.Figure()
        # fig.update_layout(
        #     xaxis=dict(title="% of runtime"), yaxis=dict(title="ranking"),
        # )

        for i in range(average_ranking.shape[0]):
            nan_ids = np.isnan(average_ranking[i, :])
            # fig.add_trace(
            #     go.Scatter(
            #         x=np.arange(average_ranking.shape[1])[~nan_ids],
            #         y=average_ranking[i, ~nan_ids],
            #         mode="lines+markers",
            #         name=HPO_names[i],
            #         line_color=color_per_HPO_method[HPO_names[i]],
            #     )
            # )
            averaged_average_ranking_per_hpo_method.append(
                np.mean(average_ranking[i, ~nan_ids])
            )
            average_end_point = np.mean(
                [d[mode]["end_points"][HPO_names[i]] for d in result_per_dataset]
            )
            # fig.add_vline(
            #     x=average_end_point,
            #     line_width=3,
            #     line_dash="dash",
            #     line_color=color_per_HPO_method[HPO_names[i]],
            # )

        averaged_average_ranking_per_mode[
            mode
        ] = averaged_average_ranking_per_hpo_method

        # st.title(mode)
        # st.plotly_chart(fig, use_container_width=True)
        # st.markdown("""---""")

    for mode in ["val", "test"]:
        # isolate the raw values. The raw_ratings variable will have the following shape (num_datasets, num_HPO_methods, sampling resolution)
        raw_performances = np.array(
            [d[mode]["raw_performance_arr"] for d in result_per_dataset]
        )

        # calculate the mean of the raw_ratings_variable on the first axis-num_datasets. This will give the average rating values of every HPO method for across datasets (important!! use metrics that make sense for all datasets)
        average_performances = np.nanmean(raw_performances, axis=0)
        for i, val in enumerate(averaged_average_ranking_per_mode[mode]):
            # print(HPO_names[i] + ": " + str(val))
            pass

        # isolate the names of the two top performing HPO methods (in terms of the average of average ranks)
        top_HPO_method_names = [
            HPO_names[i]
            for i in np.argsort(averaged_average_ranking_per_mode[mode])[:2]
        ]

        # isolate the average ratings for only two top performing HPO methods
        average_performances = average_performances[
            np.argsort(averaged_average_ranking_per_mode[mode])[:2], :
        ]

        # replace any possible nan values with zeros to eliminate any errors during plotting
        average_performances = np.nan_to_num(average_performances)
        fig = plt.figure(figsize=(25, 10))

        ax = plt.gca()
        ax.set_title(mode, fontsize=30)

        plt.rc("xtick", labelsize=35)
        plt.rc("ytick", labelsize=25)

        # calculate the score difference between the two methods
        st.subheader(
            "visualizing diff: "
            + top_HPO_method_names[0]
            + " - "
            + top_HPO_method_names[1]
        )
        diff = average_performances[0] - average_performances[1]

        # define the upper and lower limit of the y-axis
        ax.set_ylim([-max(diff), max(diff)])
        ax.set_xlim([0, 100])

        # color the two quadrants
        plt.fill_between(
            [0, 100],
            0,
            max(diff),
            alpha=0.2,
            color=color_per_HPO_method[top_HPO_method_names[0]],
        )
        plt.fill_between(
            [0, 100],
            0,
            -max(diff),
            alpha=0.2,
            color=color_per_HPO_method[top_HPO_method_names[1]],
        )

        # st.write(top_HPO_method_names[0] + ": " + str(average_performances[0]))
        # st.write(top_HPO_method_names[1] + ": " + str(average_performances[1]))

        ax.text(
            0.5,
            0.75,
            top_HPO_method_names[0].capitalize(),
            transform=ax.transAxes,
            fontsize=60,
            color="gray",
            alpha=0.5,
            ha="center",
            va="center",
            rotation=0,
        )

        ax.text(
            0.5,
            0.25,
            top_HPO_method_names[1].capitalize(),
            transform=ax.transAxes,
            fontsize=60,
            color="gray",
            alpha=0.5,
            ha="center",
            va="center",
            rotation=0,
        )
        if list(metric_used)[0] in ["hamming_loss", "RRMSE", "RMSE", "MAE", "MSE"]:
            # st.write("Loss detected. Flipping values")
            plt.plot(
                np.arange(average_performances.shape[1]),
                -diff,
                linewidth=3,
                color="black",
            )
        else:
            plt.plot(
                np.arange(average_performances.shape[1]),
                diff,
                linewidth=3,
                color="black",
            )
        st.pyplot(fig)
