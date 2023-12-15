import numpy as np
import pandas as pd
from curve_based import curve_model_s
from itertools import chain
from tqdm.notebook import tqdm, trange
from landmarks import DlibLandmarksDetector
from pathlib import Path
from settings.expresions import FACIAL_EXPRESSION_NAMES

def curve_measures(videos_path):
    curve_measures_list = []
    landmarks_detector = DlibLandmarksDetector()
    for video_path in tqdm(videos_path, desc="Computing similar measures"):
        video = np.load(video_path)
        landmarks = landmarks_detector.preprocess_and_detect_landmarks_numpy(video[0:1])
        measure = curve_model_s(video[0], landmarks[0])
        flatten_measure = tuple(chain.from_iterable(measure))
        curve_measures_list.append({
            "video_path": video_path,
            "measure": np.array(flatten_measure)
        })
    curve_measures_df = pd.DataFrame(curve_measures_list)
    return curve_measures_df

def video_subject(video_path):
    return Path(video_path).parents[-5].name

def remove_query_subjects(curve_measures_df, query_video_path):
    query_subject = video_subject(query_video_path)
    curve_measures_df = curve_measures_df[curve_measures_df["video_path"].str.contains(query_subject) == False]
    return curve_measures_df

def select_facial_expression(videos_path, expression_name):
    expression_videos_path = [video_path for video_path in videos_path if expression_name in video_path]
    return expression_videos_path


def similar_index(query_paths, search_paths, output_folder):
    similar_index_list = []
    query_curve_measures_df = curve_measures(query_paths)
    for expression_name in FACIAL_EXPRESSION_NAMES:
        search_expression_search_paths = select_facial_expression(search_paths, expression_name)
        search_curve_measures_df = curve_measures(search_expression_search_paths)
        for i in trange(len(query_curve_measures_df), desc=f"Generating {expression_name} similar videos index"):
            query_metric = np.array(query_curve_measures_df.iloc[i]["measure"])
            query_path = query_curve_measures_df.iloc[i]["video_path"]
            search_df = remove_query_subjects(search_curve_measures_df, query_path)
            curve_measures_array = np.array(search_df["measure"].values.tolist())
            most_curve_measures = np.sum(np.abs(curve_measures_array - query_metric), axis=1)
            most_similar_videos_ids = np.argsort(most_curve_measures, axis=0)
            most_similar_video_path = search_df.iloc[most_similar_videos_ids[1]]["video_path"]
            similar_index_list.append({
                "video_path": query_path,
                "similar_video_path": most_similar_video_path,
            })
        similar_index_df = pd.DataFrame(similar_index_list)
        similar_index_df.set_index('video_path')
        output_path = str(output_folder / f"{expression_name}.csv")
        similar_index_df.to_csv(output_path)
        print(f"Similar index for {expression_name} saved in {output_path}")
    

def compute_similar_measures(videos_path, landmark_detector):
    measures_similar = []
    for i in trange(len(videos_path), desc="Computing similar measures"):
        video = np.load(videos_path[i])
        landmarks = landmark_detector.preprocess_and_detect_landmarks_numpy(video[0:1])
        measure = curve_model_s(video[0], landmarks[0])
        flatten_measure = tuple(chain.from_iterable(measure))
        measures_similar.append({
            "video_path": videos_path[i],
            "measure": np.array(flatten_measure)
        })
    measure_df = measures_similar = pd.DataFrame(measures_similar)
    measure_df = measure_df.set_index('video_path')
    return measure_df


def compute_similar_measures_with_query(query_paths, search_paths, remove_query_subject=False):
    similar_index_list = []
    similar_index_expression_dict = {}
    query_curve_measures_df = curve_measures(query_paths)
    for expression_name in FACIAL_EXPRESSION_NAMES:
        search_expression_search_paths = select_facial_expression(search_paths, expression_name)
        search_curve_measures_df = curve_measures(search_expression_search_paths)
        for i in trange(len(query_curve_measures_df), desc=f"Generating {expression_name} similar videos index"):
            query_metric = np.array(query_curve_measures_df.iloc[i]["measure"])
            query_path = query_curve_measures_df.iloc[i]["video_path"]
            if remove_query_subject:
                search_df = remove_query_subjects(search_curve_measures_df, query_path)
            else:
                search_df = search_curve_measures_df
            curve_measures_array = np.array(search_df["measure"].values.tolist())
            most_curve_measures = np.sum(np.abs(curve_measures_array - query_metric), axis=1)
            most_similar_videos_ids = np.argsort(most_curve_measures, axis=0)
            most_similar_video_path = search_df.iloc[most_similar_videos_ids[0]]["video_path"]
            similar_index_list.append({
                "video_path": query_path,
                "similar_video_path": most_similar_video_path,
            })
        similar_index_df = pd.DataFrame(similar_index_list)
        similar_index_df = similar_index_df.set_index('video_path')
        similar_index_expression_dict[expression_name] = similar_index_df
    return similar_index_expression_dict