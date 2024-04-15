import numpy as np
import pandas as pd
from curve_based import curve_model_s
from itertools import chain
from tqdm.notebook import tqdm, trange
from landmarks import DlibLandmarksDetector
from pathlib import Path
from settings.expresions import FACIAL_EXPRESSION_NAMES

def curve_measures(videos_path):
    """
    Compute curve measures for a list of videos.

    Args:
        videos_path (list): A list of file paths to the videos.

    Returns:
        pandas.DataFrame: A DataFrame containing the computed curve measures for each video.

    """
    curve_measures_list = []
    landmarks_detector = DlibLandmarksDetector()

    # Iterate over each video path
    for video_path in tqdm(videos_path, desc="Computing similar measures"):
        video = np.load(video_path)
        landmarks = landmarks_detector.preprocess_and_detect_landmarks_numpy(video[0:1])

        # Compute curve measure for the first frame of the video
        measure = curve_model_s(video[0], landmarks[0])

        # Flatten the measure and store it in a list
        flatten_measure = tuple(chain.from_iterable(measure))
        curve_measures_list.append({
            "video_path": video_path,
            "measure": np.array(flatten_measure)
        })

    # Create a DataFrame from the list of curve measures
    curve_measures_df = pd.DataFrame(curve_measures_list)
    return curve_measures_df

def video_subject(video_path):
    """
    Extracts the subject name from a video path.

    Parameters:
    video_path (str): The path of the video file.

    Returns:
    str: The name of the subject.

    Example:
    >>> video_subject('/home/rafa/videos/subject1/video.mp4')
    'subject1'
    """

    # Convert the video path to a Path object
    video_path = Path(video_path)

    # Extract the name of the subject from the video path
    subject_name = video_path.parents[-5].name

    return subject_name

def remove_query_subjects(curve_measures_df, query_video_path):
    """
    Removes rows from the curve_measures_df DataFrame that contain the subject of the query video.

    Parameters:
    curve_measures_df (DataFrame): The DataFrame containing curve measures.
    query_video_path (str): The path of the query video.

    Returns:
    DataFrame: The updated curve_measures_df DataFrame without rows containing the subject of the query video.
    """
    # Get the subject of the query video
    query_subject = video_subject(query_video_path)

    # Filter out rows that contain the subject of the query video
    curve_measures_df = curve_measures_df[curve_measures_df["video_path"].str.contains(query_subject) == False]

    return curve_measures_df

def select_facial_expression(videos_path, expression_name):
    """
    Selects facial expression videos from a given list of video paths based on the expression name.

    Parameters:
    - videos_path (list): A list of video paths.
    - expression_name (str): The name of the facial expression to select.

    Returns:
    - expression_videos_path (list): A list of video paths that match the given expression name.
    """
    # Filter the video paths based on the expression name
    expression_videos_path = [video_path for video_path in videos_path if expression_name in video_path]

    return expression_videos_path


def similar_index(query_paths, search_paths, output_folder):
    """
    This function finds videos with similar facial expressions to a given query video. 
    It does this by comparing geometric features extracted from facial landmarks using the `curve_model_s` function.

    Args:
        query_paths (list): A list of file paths to the query videos.
        search_paths (list): A list of file paths to the search videos.
        output_folder (str): The path to the folder where similarity results will be saved.

    Returns:
        None - This function saves the similarity results as CSV files for each facial expression.
    """

    # Calculate features for query videos 
    query_curve_measures_df = curve_measures(query_paths)

    similar_index_list = []  # Initialize a list to store similarity results

    # Process each facial expression type
    for expression_name in FACIAL_EXPRESSION_NAMES:
        # Filter search videos for matching expression
        search_expression_search_paths = select_facial_expression(search_paths, expression_name)
        search_curve_measures_df = curve_measures(search_expression_search_paths)

        # Compare each query video with relevant search videos
        for i in trange(len(query_curve_measures_df), desc=f"Generating {expression_name} similar videos index"):
            query_metric = np.array(query_curve_measures_df.iloc[i]["measure"])
            query_path = query_curve_measures_df.iloc[i]["video_path"]

            # Avoid comparing a video with itself 
            search_df = remove_query_subjects(search_curve_measures_df, query_path)

            # Calculate similarity score with remaining videos
            curve_measures_array = np.array(search_df["measure"].values.tolist())
            most_curve_measures = np.sum(np.abs(curve_measures_array - query_metric), axis=1)
            most_similar_videos_ids = np.argsort(most_curve_measures, axis=0)

            # Get the most similar video (excluding the query video itself, which would be at index 0)
            most_similar_video_path = search_df.iloc[most_similar_videos_ids[1]]["video_path"]

            # Store the result
            similar_index_list.append({
                "video_path": query_path,
                "similar_video_path": most_similar_video_path,
            })

        # Create a DataFrame and save similarity results for the current expression
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