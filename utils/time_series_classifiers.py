import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay
from datetime import datetime
import pytz

import tsai
from tsai.all import *

def create_windowed_data(df, sensor_cols, window_size, step_size, variables_first, all_windows=False, label_list=None, constant_label=None):
    """
    df - pandas dataframe with index corresponding to epoch time in milliseconds
    sensor_cols - columns that should be included in the output sliding windows
    window_size - number of samples (df rows) to use in each window
    step_size - number of rows to step forward before creating the next window
    variables_first - If True, return windows as a 3D array with shape (# samples, # variables, window_size)
                      If False, return windows as a 3D array with shape (# samples, window_size, # variables)
    all_windows - If False then only return windows (and labels and intervals) for labeled intervals (i.e., determined by 
                      label_list or constant_label). If True, then you will also get windows with a corresponding 
                      label of "" for unlabeled data and data that overlapped label boundaries
    label_list - List of format [[StartTimestamp, EndTimestamp, LabelString, Subject], ] that will be used to assign labels 
                      to windows (e.g., used when loading from a file that has multiple intervals of activities with 
                      different classes)
    constant_label - A single label string that should be applied to all windows (e.g., used when loading from a file 
                      that only contains a single class)
    """
    assert step_size <= window_size
    assert step_size > 0

    max_gap = 100 #max gap allowed in sensor data in milliseconds

    windowed_data = []
    intervals = []
    labels = []
    subjects = [] 

    current_window = []
    current_interval = []
    current_labels = []
    last_timestamp = None

    if label_list is not None:
        label_list.sort()
        label_idx = 0

    for row in df.itertuples():
        row_values = [getattr(row, col) for col in sensor_cols ]
        timestamp = getattr(row, "Index")

        if last_timestamp is None:
            last_timestamp = timestamp

        if constant_label is not None:
            label = constant_label
            subject = None
        elif label_list is not None:
            label = ""
            subject = None
            for i in range(label_idx, len(label_list)):
                if timestamp > label_list[i][1]:
                    label_idx += 1
                elif timestamp < label_list[i][0]:
                    break
                elif timestamp >= label_list[i][0] and timestamp <= label_list[i][1]:
                    label = label_list[i][2]
                    if len(label_list[i]) > 3:
                        subject = label_list[i][3]
                    break
        else:
            label = ""
            subject = None

        if not all_windows and len(current_labels) > 0 and label != current_labels[-1]:
            current_window = []
            current_interval = []
            current_labels = []

        if timestamp - last_timestamp > max_gap:
            current_window = []
            current_interval = []
            current_labels = []
        last_timestamp = timestamp

        current_window.append(row_values)
        current_interval.append(timestamp)
        current_labels.append(label)

        if len(current_window) == window_size:
            if all_windows or label != "":
                if variables_first:
                    windowed_data.append(np.transpose(current_window))
                else:
                    windowed_data.append(np.array(current_window))

                intervals.append((current_interval[0], current_interval[-1]))

                if len(set(current_labels)) > 1:
                    labels.append("")
                else:
                    labels.append(label)

                subjects.append(subject)

            for i in range(step_size):
                current_window.pop(0)
                current_interval.pop(0)
                current_labels.pop(0)

    return windowed_data, labels, intervals, subjects

def datetime_merging(primary_df, secondary_df_list):
    """
    primary_df - this df will be used to decide the timestamps used for the combined_df
    secondary_df_list - list of dfs that should be joined into primary_df.
                        Any row with a timestamp that matches a timestamp in primary_df will have its values matched to that primary_df row
                        Any rows in primary_df that don't have a match in the secondary df will have values backfilled to fill the gap.
    """
    combined_df = primary_df.copy(deep=True)

    for df2 in secondary_df_list:
        combined_df = combined_df.join(df2, how="left")
    
    combined_df.fillna(method='bfill', inplace=True)
    combined_df.dropna(inplace=True)
    return combined_df

def create_confusion_matrix(gt_ints, pred_ints, class_list, filename, title_text=None, debug_mode=False):
    gt_set = set(gt_ints)
    pred_set = set(pred_ints)

    try:
        cm = confusion_matrix(gt_ints, pred_ints)

        combined_set = gt_set | pred_set
        if len(combined_set) != len(class_list):
            ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=None).plot(include_values=False, xticks_rotation='vertical', cmap="magma_r")
        else:
            ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_list).plot(include_values=False, xticks_rotation='vertical', cmap="magma_r")
        if title_text is not None:
            plt.title(title_text)
        plt.savefig(filename)

    except ValueError as err:
        print(err)

        if debug_mode:
            print("DEBUG MODE: Adding extra items so that all classes are represented in gt values")

            #DEBUGGING only
            #adding an extra bad prediction for each gt class that wasn't present in this test data so we can display the confusion matrix
            #when checking that the classification seems to work well. This part should not be used when reporting results
            for pred in pred_set:
                if pred not in gt_set:
                    bad_pred = [p for p in pred_set if p != pred][0]
                    gt_ints.append(pred)
                    pred_ints.append(bad_pred)
            #END DEBUG

            cm = confusion_matrix(gt_ints, pred_ints)

            combined_set = gt_set | pred_set
            if len(combined_set) != len(class_list):
                ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=None).plot(include_values=False, xticks_rotation='vertical', cmap="magma_r")
            else:
                ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_list).plot(include_values=False, xticks_rotation='vertical', cmap="magma_r")
            if title_text is not None:
                plt.title(title_text)
            plt.savefig(filename)

def extract_date_string_from_filename(full_path):
    filename = os.path.basename(full_path)
    if "\\" in filename:
        filename = filename.split("\\")[-1]

    filename_list = filename.split("_")
    return filename_list[0]

def load_csv_file_list(csv_file_path_list, cols, prefix=None, sample_rate=None):
    """
    csv_file_path_list - list of csv paths that should all be combined into a single df
    cols - list of columns that should be used when creating the df
    prefix - adds a prefix to sensor column names, useful when multiple sensors have a column with the same name (e.g., X)
    sample_rate - Sample rate (in Hz) to use for the data. If None, it will be estimated from the data
    """

    combined_df = None

    for csv_file_path in csv_file_path_list:
        try:
            df = pd.read_csv(csv_file_path, usecols=cols)
        except ValueError:
            df = pd.read_csv(csv_file_path, names=cols)

        if len(df) == 0:
            print("Empty csv file found: " + str(csv_file_path), flush=True)
            continue

        if prefix is not None:
            rename_cols = {c: prefix + "_" + c for c in cols if c != "time"}
            df.rename(columns=rename_cols, inplace=True)
        
        if "." not in df.at[0, "time"]:
            df.at[0, "time"] = df.at[0, "time"] + ".000"

        #https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#from-timestamps-to-epoch
        stt_time = pd.to_datetime(df.at[0, "time"], format="%Y%m%d_%H:%M:%S.%f").tz_localize('Asia/Tokyo').tz_convert("UTC")
        stt_time = (stt_time - pd.Timestamp("1970-01-01", tz="UTC")) / pd.Timedelta("1s")
        stt_time = round(stt_time, 2) #we're only dealing with up to 100 Hz, this should be updated if higher samping rates are used

        if sample_rate is None:
            if "." not in df.at[len(df["time"])-1, "time"]:
                df.at[len(df["time"])-1, "time"] = df.at[len(df["time"])-1, "time"] + ".000"
            end_time = pd.to_datetime(df.at[len(df["time"])-1, "time"], format="%Y%m%d_%H:%M:%S.%f").tz_localize('Asia/Tokyo').tz_convert("UTC")
            end_time = (end_time - pd.Timestamp("1970-01-01", tz="UTC")) / pd.Timedelta("1s")
            end_time = round(end_time, 2) #we're only dealing with up to 100 Hz, this should be updated if higher samping rates are used
            sample_rate = round(len(df["time"]) / (end_time - stt_time))

        #set time values as milliseconds since epoch
        time_values = np.array([ 1000 * (stt_time + (i/sample_rate)) for i in range(len(df["time"])) ], dtype=np.int64)
        df["time"] = time_values

        if combined_df is None:
            combined_df = df
        else:
            combined_df = pd.concat([combined_df, df])

    if combined_df is None:
        return None
    else:
        combined_df.set_index("time", inplace=True)
        combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
        return combined_df

def trimmed_data_example(trimmed_dir, use_sensors):
    print("trimmed cross scene example")

    trimmed_dir = os.path.join(trimmed_dir, "sensors")

    sensor_dir_list = ["acc_phone_clip", "acc_watch_clip", "gyro_clip", "orientation_clip",]
    subject_list = ["subject" + str(i) for i in range(1, 21)]
    scene_list = ["scene" + str(i) for i in range(1, 5)]
    session_list = ["session" + str(i) for i in range(1, 6)]

    filename_case_map = {} #allows us to catch filenames that have capitalized letters

    class_list = set()
    for sensor_dir in sensor_dir_list:
        for subject in subject_list:
            for scene in scene_list:
                for session in session_list:
                    d = os.path.join(trimmed_dir, sensor_dir, subject, scene, session)
                    if os.path.exists(d):
                        for filename in os.listdir(d):
                            if filename.endswith(".csv"):
                                c = filename.split(".")[0].strip()
                                c_lower = c.lower()
                                if c_lower not in filename_case_map:
                                    filename_case_map[c_lower] = [c_lower]
                                if c not in filename_case_map[c_lower]:              
                                    filename_case_map[c_lower].append(c)
                                class_list.add(c_lower)
    class_list = list(class_list)
    class_list.sort()

    train_scene_list = ["scene1", "scene2","scene3"] #training scene from the challenge
    test_scene_list = ["scene4"] #val scene from the challenge
    for s in test_scene_list:
        assert s not in train_scene_list
    for s in train_scene_list:
        assert s not in test_scene_list

    train_X = []
    train_y = []
    train_intervals = []
    test_X = []
    test_y = []
    test_intervals = []

    for subject in subject_list:
        print(subject, flush=True)
        for scene in scene_list:
            for session in session_list:
                for label in class_list:
                    acc_w_df = None
                    acc_s_df = None
                    gyro_df = None
                    orientation_df = None                

                    if "acc_phone" in use_sensors: 
                        for label_ in filename_case_map[label]:
                            acc_s_path = os.path.join(trimmed_dir, "acc_phone_clip", subject, scene, session, label_ + ".csv")
                            if os.path.exists(acc_s_path):
                                acc_s_df = load_csv_file_list(csv_file_path_list=[acc_s_path], cols=["time", "x", "y", "z"], prefix="s_acc", sample_rate=100)
                        if acc_s_df is None:
                            continue
                        else:
                            acc_s_df = acc_s_df / 9.8 #converting values from m/s^2 to g's

                    if "acc_watch" in use_sensors: 
                        for label_ in filename_case_map[label]:
                            acc_w_path = os.path.join(trimmed_dir, "acc_watch_clip", subject, scene, session, label_ + ".csv")
                            if os.path.exists(acc_w_path):
                                acc_w_df = load_csv_file_list(csv_file_path_list=[acc_w_path], cols=["time", "x", "y", "z"], prefix="w_acc", sample_rate=100)
                        if acc_w_df is None:
                            continue
                        else:
                            acc_w_df = acc_w_df / 9.8 #converting values from m/s^2 to g's

                    if "gyro" in use_sensors: 
                        for label_ in filename_case_map[label]:
                            gyro_path = os.path.join(trimmed_dir, "gyro_clip", subject, scene, session, label_ + ".csv")
                            if os.path.exists(gyro_path):
                                gyro_df = load_csv_file_list(csv_file_path_list=[gyro_path], cols=["time", "x", "y", "z"], prefix="gyro", sample_rate=50)
                        if gyro_df is None:
                            continue

                    if "orientation" in use_sensors: 
                        for label_ in filename_case_map[label]:
                            orientation_path = os.path.join(trimmed_dir, "orientation_clip", subject, scene, session, label_ + ".csv")
                            if os.path.exists(orientation_path):
                                orientation_df = load_csv_file_list(csv_file_path_list=[orientation_path], cols=["time", "azimuth", "pitch", "roll"], sample_rate=None)
                        if orientation_df is None:
                            continue

                    if acc_s_df is not None:
                        primary_df = acc_s_df
                        secondary_df_list=[acc_w_df, gyro_df, orientation_df]
                        secondary_df_list = [item for item in secondary_df_list if item is not None]

                    elif acc_w_df is not None:
                        primary_df = acc_w_df
                        secondary_df_list=[acc_s_df, gyro_df, orientation_df]
                        secondary_df_list = [item for item in secondary_df_list if item is not None]

                    elif gyro_df is not None:
                        primary_df = gyro_df
                        secondary_df_list=[acc_s_df, acc_w_df, orientation_df]
                        secondary_df_list = [item for item in secondary_df_list if item is not None]

                    elif orientation_df is not None:
                        primary_df = orientation_df
                        secondary_df_list=[acc_s_df, acc_w_df, gyro_df]
                        secondary_df_list = [item for item in secondary_df_list if item is not None]

                    else:
                        raise ValueError()

                    combined_df = datetime_merging(primary_df=primary_df, secondary_df_list=secondary_df_list)

                    all_windows = False
                    variables_first = True
                    sensor_cols = [col for col in combined_df.columns if col != "time"]
                    window_size = 100 #1 second window of 100 Hz data
                    step_size = 25 #75% overlap between windows

                    X, y, intervals, _ = create_windowed_data(df=combined_df, sensor_cols=sensor_cols, window_size=window_size, step_size=step_size, variables_first=variables_first, all_windows=all_windows, label_list=None, constant_label=label.lower())

                    if scene in train_scene_list:
                        train_X.extend(X)
                        train_y.extend(y)
                        train_intervals.extend(intervals)

                    elif scene in test_scene_list:
                        test_X.extend(X)
                        test_y.extend(y)
                        test_intervals.extend(intervals)

    train_val_split = int(0.9*len(train_X))
    temp_train_X = train_X[:train_val_split]
    temp_train_y = train_y[:train_val_split]
    temp_train_intervals = train_intervals[:train_val_split]
    val_X = train_X[train_val_split:]
    val_y = train_y[train_val_split:]
    val_intervals = train_intervals[train_val_split:]
    train_X = temp_train_X
    train_y = temp_train_y
    train_intervals = temp_train_intervals

    train_X = np.array(train_X)
    train_y = np.array(train_y)
    val_X = np.array(val_X)
    val_y = np.array(val_y)
    test_X = np.array(test_X)
    test_y = np.array(test_y)

    print(train_X.shape)
    print(train_y.shape)
    print(len(train_intervals))
    print(val_X.shape)
    print(val_y.shape)
    print(len(val_intervals))
    print(test_X.shape)
    print(test_y.shape)
    print(len(test_intervals))

    X, y, splits = combine_split_data([train_X, val_X], [train_y, val_y])
    tfms  = [None, [Categorize()]]
    dsets = TSDatasets(X, y, tfms=tfms, splits=splits, inplace=True)
    dls = TSDataLoaders.from_dsets(dsets.train, dsets.valid, bs=[1024, 1024], shuffle=True, batch_tfms=[], num_workers=0)

    model = InceptionTime(dls.vars, dls.c)
    learn = Learner(dls, model, wd=0.01, path="nn", model_dir="checkpoints", metrics=accuracy, loss_func=LabelSmoothingCrossEntropy())
    _, lr_steep = learn.lr_find()

    learn.fit_one_cycle(15, lr_max=lr_steep)

    valid_dl = dls.valid
    test_ds = valid_dl.dataset.add_test(test_X)
    test_dl = valid_dl.new(test_ds)

    test_probas, _, test_preds = learn.get_preds(dl=test_dl, with_decoded=True, save_preds=None, save_targs=None)

    vocab = list(dls.vocab)
    test_y_ints, test_pred_ints = [], []
    for i in range(len(test_y)):
        test_y_ints.append(vocab.index(test_y[i]))
        test_pred_ints.append(int(test_preds[i]))

    print("Example of how to covert timestamps in intervals list to datetime objects:")
    print("test_intervals[0] corresponds to the start and stop times of the sliding window used to extract the data for test_X[0]")
    print("raw values in test_intervals[0]:", test_intervals[0])
    stt = datetime.fromtimestamp(test_intervals[0][0] / 1000.0, tz=pytz.timezone("Asia/Tokyo"))
    end = datetime.fromtimestamp(test_intervals[0][1] / 1000.0, tz=pytz.timezone("Asia/Tokyo"))
    print("datetime values in test_intervals[0]:", stt.strftime('%Y-%m-%d %H:%M:%S.%f'), end.strftime('%Y-%m-%d %H:%M:%S.%f'))

    f1 = f1_score(test_y_ints, test_pred_ints, average="macro")
    print(f'macro f-measure: {f1:10.6f}')

    title_text = 'Trimmed cross scene example, macro f-measure: ' + str(round(f1, 3))
    create_confusion_matrix(gt_ints=test_y_ints, pred_ints=test_pred_ints, class_list=vocab, filename="trimmed_cross_scene_example_cm.svg", title_text=title_text, debug_mode=True)

def untrimmed_data_example(untrimmed_dir, use_sensors):
    print("untrimmed cross session example")

    subject_list = ["subject" + str(i) for i in range(1, 21)]
    scene_list = ["scene" + str(i) for i in range(1, 5)]
    session_list = ["session" + str(i) for i in range(1, 6)]

    class_list = set()
    label_dict = {}
    for subject in subject_list:
        label_dict[subject] = {}
        for scene in scene_list:
            label_dict[subject][scene] = {}
            for session in session_list:
                label_dict[subject][scene][session] = []
                label_path = os.path.join(untrimmed_dir, "sessions", scene, subject, scene + "_" + subject + "_" + session + ".txt")
                if os.path.exists(label_path):
                    with open(label_path, "r") as label_file:
                        for line in label_file:
                            line_list = line.strip().split("-")
                            stt_time = pd.to_datetime(line_list[0], format="%Y/%m/%d %H:%M:%S.%f").tz_localize('Asia/Tokyo').tz_convert("UTC")
                            stt_time = int((stt_time - pd.Timestamp("1970-01-01", tz="UTC")) / pd.Timedelta("1ms"))
                            end_time = pd.to_datetime(line_list[1], format="%Y/%m/%d %H:%M:%S.%f").tz_localize('Asia/Tokyo').tz_convert("UTC")
                            end_time = int((end_time - pd.Timestamp("1970-01-01", tz="UTC")) / pd.Timedelta("1ms"))
                            label = line_list[2].lower().strip()
                            class_list.add(label)
                            label_dict[subject][scene][session].append([stt_time, end_time, label])
    class_list = list(class_list)
    class_list.sort()

    train_session_list = ["session1", "session2", "session3"] #training sessions from the challenge
    test_session_list = ["session4"] #val session from the challenge
    for s in test_session_list:
        assert s not in train_session_list
    for s in train_session_list:
        assert s not in test_session_list

    train_X = []
    train_y = []
    train_intervals = []
    test_X = []
    test_y = []
    test_intervals = []

    base_dir = os.path.join(untrimmed_dir, "sensors")
    for subject in subject_list:
        print(subject, flush=True)
        for scene in scene_list:
            for session in session_list:
                acc_w_df = None
                acc_s_df = None
                gyro_df = None
                orientation_df = None                

                if "acc_phone" in use_sensors: 
                    acc_s_dir = os.path.join(base_dir, "acc_phone_clip", subject, scene, session)
                    acc_s_files = []
                    if os.path.exists(acc_s_dir):
                        for filename in os.listdir(acc_s_dir):
                            if filename.endswith(".csv"):
                                acc_s_files.append(os.path.join(acc_s_dir, filename))
                    acc_s_files.sort()
                    acc_s_df = load_csv_file_list(csv_file_path_list=acc_s_files, cols=["time", "x", "y", "z"], prefix="s_acc", sample_rate=100)
                    if acc_s_df is None:
                        print("Missing acc_phone data for " + subject + " " + scene + " " + session)
                        continue
                    else:
                        acc_s_df = acc_s_df / 9.8 #converting values from m/s^2 to g's

                if "acc_watch" in use_sensors: 
                    acc_w_dir = os.path.join(base_dir, "acc_watch_clip", subject, scene, session)
                    acc_w_files = []
                    if os.path.exists(acc_w_dir):
                        for filename in os.listdir(acc_w_dir):
                            if filename.endswith(".csv"):
                                acc_w_files.append(os.path.join(acc_w_dir, filename))
                    acc_w_files.sort()
                    acc_w_df = load_csv_file_list(csv_file_path_list=acc_w_files, cols=["time", "x", "y", "z"], prefix="w_acc", sample_rate=100)
                    if acc_w_df is None:
                        print("Missing acc_watch data for " + subject + " " + scene + " " + session)
                        continue
                    else:
                        acc_w_df = acc_w_df / 9.8 #converting values from m/s^2 to g's

                if "gyro" in use_sensors: 
                    gyro_dir = os.path.join(base_dir, "gyro_clip", subject, scene, session)
                    gyro_files = []
                    if os.path.exists(gyro_dir):
                        for filename in os.listdir(gyro_dir):
                            if filename.endswith(".csv"):
                                gyro_files.append(os.path.join(gyro_dir, filename))
                    gyro_files.sort()
                    gyro_df = load_csv_file_list(csv_file_path_list=gyro_files, cols=["time", "x", "y", "z"], prefix="gyro", sample_rate=50)
                    if gyro_df is None:
                        print("Missing gyro data for " + subject + " " + scene + " " + session)
                        continue

                if "orientation" in use_sensors: 
                    orientation_dir = os.path.join(base_dir, "orientation_clip", subject, scene, session)
                    orientation_files = []
                    if os.path.exists(orientation_dir):
                        for filename in os.listdir(orientation_dir):
                            if filename.endswith(".csv"):
                                orientation_files.append(os.path.join(orientation_dir, filename))
                    orientation_files.sort()
                    orientation_df = load_csv_file_list(csv_file_path_list=orientation_files, cols=["time", "azimuth", "pitch", "roll"], sample_rate=None)
                    if orientation_df is None:
                        print("Missing orientation data for " + subject + " " + scene + " " + session)
                        continue

                if acc_s_df is not None:
                    primary_df = acc_s_df
                    secondary_df_list=[acc_w_df, gyro_df, orientation_df]
                    secondary_df_list = [item for item in secondary_df_list if item is not None]

                elif acc_w_df is not None:
                    primary_df = acc_w_df
                    secondary_df_list=[acc_s_df, gyro_df, orientation_df]
                    secondary_df_list = [item for item in secondary_df_list if item is not None]

                elif gyro_df is not None:
                    primary_df = gyro_df
                    secondary_df_list=[acc_s_df, acc_w_df, orientation_df]
                    secondary_df_list = [item for item in secondary_df_list if item is not None]

                elif orientation_df is not None:
                    primary_df = orientation_df
                    secondary_df_list=[acc_s_df, acc_w_df, gyro_df]
                    secondary_df_list = [item for item in secondary_df_list if item is not None]

                else:
                    raise ValueError()

                combined_df = datetime_merging(primary_df=primary_df, secondary_df_list=secondary_df_list)

                all_windows = False
                variables_first = True
                sensor_cols = [col for col in combined_df.columns if col != "time"]
                window_size = 100 #1 second window of 100 Hz data
                step_size = 25 #75% overlap between windows

                X, y, intervals, _ = create_windowed_data(df=combined_df, sensor_cols=sensor_cols, window_size=window_size, step_size=step_size, variables_first=variables_first, all_windows=all_windows, label_list=label_dict[subject][scene][session], constant_label=None)

                if session in train_session_list:
                    train_X.extend(X)
                    train_y.extend(y)
                    train_intervals.extend(intervals)

                elif session in test_session_list:
                    test_X.extend(X)
                    test_y.extend(y)
                    test_intervals.extend(intervals)

    train_val_split = int(0.9*len(train_X))
    temp_train_X = train_X[:train_val_split]
    temp_train_y = train_y[:train_val_split]
    temp_train_intervals = train_intervals[:train_val_split]
    val_X = train_X[train_val_split:]
    val_y = train_y[train_val_split:]
    val_intervals = train_intervals[train_val_split:]
    train_X = temp_train_X
    train_y = temp_train_y
    train_intervals = temp_train_intervals

    train_X = np.array(train_X)
    train_y = np.array(train_y)
    val_X = np.array(val_X)
    val_y = np.array(val_y)
    test_X = np.array(test_X)
    test_y = np.array(test_y)

    print(train_X.shape)
    print(train_y.shape)
    print(len(train_intervals))
    print(val_X.shape)
    print(val_y.shape)
    print(len(val_intervals))
    print(test_X.shape)
    print(test_y.shape)
    print(len(test_intervals))

    X, y, splits = combine_split_data([train_X, val_X], [train_y, val_y])
    tfms  = [None, [Categorize()]]
    dsets = TSDatasets(X, y, tfms=tfms, splits=splits, inplace=True)
    dls = TSDataLoaders.from_dsets(dsets.train, dsets.valid, bs=[1024, 1024], shuffle=True, batch_tfms=[], num_workers=0)

    model = InceptionTime(dls.vars, dls.c)
    learn = Learner(dls, model, wd=0.01, path="nn", model_dir="checkpoints", metrics=accuracy, loss_func=LabelSmoothingCrossEntropy())
    _, lr_steep = learn.lr_find()

    learn.fit_one_cycle(15, lr_max=lr_steep)

    valid_dl = dls.valid
    test_ds = valid_dl.dataset.add_test(test_X)
    test_dl = valid_dl.new(test_ds)

    test_probas, _, test_preds = learn.get_preds(dl=test_dl, with_decoded=True, save_preds=None, save_targs=None)

    vocab = list(dls.vocab)
    test_y_ints, test_pred_ints = [], []
    for i in range(len(test_y)):
        test_y_ints.append(vocab.index(test_y[i]))
        test_pred_ints.append(int(test_preds[i]))

    print("Example of how to covert timestamps in intervals list to datetime objects:")
    print("test_intervals[0] corresponds to the start and stop times of the sliding window used to extract the data for test_X[0]")
    print("raw values in test_intervals[0]:", test_intervals[0])
    stt = datetime.fromtimestamp(test_intervals[0][0] / 1000.0, tz=pytz.timezone("Asia/Tokyo"))
    end = datetime.fromtimestamp(test_intervals[0][1] / 1000.0, tz=pytz.timezone("Asia/Tokyo"))
    print("datetime values in test_intervals[0]:", stt.strftime('%Y-%m-%d %H:%M:%S.%f'), end.strftime('%Y-%m-%d %H:%M:%S.%f'))

    f1 = f1_score(test_y_ints, test_pred_ints, average="macro")
    print(f'macro f-measure: {f1:10.6f}')

    title_text = 'Untrimmed cross session example, macro f-measure: ' + str(round(f1, 3))
    create_confusion_matrix(gt_ints=test_y_ints, pred_ints=test_pred_ints, class_list=vocab, filename="untrimmed_cross_session_example_cm.svg", title_text=title_text, debug_mode=True)

if __name__ == "__main__":

    use_sensors = ["acc_phone", "acc_watch", "gyro",] #["acc_phone", "acc_watch", "gyro", "orientation",]

    trimmed_dir = "trimmed"

    #Trimmed cross-scene example
    trimmed_data_example(trimmed_dir=trimmed_dir, use_sensors=use_sensors)
 
    untrimmed_dir = "untrimmed"
 
    #Untrimmed cross-session example
    untrimmed_data_example(untrimmed_dir=untrimmed_dir, use_sensors=use_sensors)
