# MMAct Challenge
## <a name="aeol"></a>MMAct Challenge 2021 with ActivityNet @ CVPR'21

The [MMAct Challenge 2021](file:///Users/kong/Project/Dataset/challenge/index.html#downloads) will be hosted in the [CVPR'21 International Challenge on Activity Recognition (ActivityNet) Workshop](http://activity-net.org/challenges/2021/index.html).</a>
This challenge asks participants to propose cross-modal video action recognition/localization approaches for addressing shortcomings in visual only approaches using [MMAct Dataset](https://mmact19.github.io/2019/).

### Dataset structure
After the extraction of the dataset is over, you will see the dataset structure as follows.

For videos, video data will be stored under each session folders. Example of untrimmed videos under `untrimmed/video` folder:
```
trainval/
├── cam1
│   ├── subject1
│   │   ├── scene1
│   │   │   ├── session1
│   │   │   │   └── 20181016-133948.mp4
│   │   │   ├── session2
│   │   │   │   └── 20181016-134634.mp4
│   │   │   └── session3
│   │   │       └── 20181016-140459.mp4
...
```
For sensors, sensor data (.csv) will be stored under each session folders according to each type of sensor, `acc_phone_clip, gyro_clip, orientation_clip`:acceleration, gyroscope, orientation from smartphone in the right pockets of pants, `acc_watch_clip`:acceleration from smartwatch worn on the right hand. Example of untrimmed sensor data under `untrimmed/sensor`. Notice that sensor has NO view definition.
```
sensor/
├── acc_phone_clip
│   ├── subject1
│   │   ├── scene1
│   │   │   ├── session1
│   │   │   ├── session2
│   │   │   ├── session3
...
```
For trimmed data, the annotation will be the file name itself. For untrimmed data, the annotation will be stored under `untrimmed/annotation` folder, the file name of the untrimmed video is the end time stamp. The split of `train` and `val` for each sub-task will be stored in `trimmed/splits` and `untrimmed/splits` respectively. Camera views with the same session index as `subjectXX/sceneYY/sessionZZ` share the same annotation in `annotation/trainval/subjectXX/sceneYY/sessionZZ/`. The folder structure under `untrimmed/annotation` is:
```
trainval/
├── subject1
│   ├── scene1
│   │   ├── session1
│   │   │   └── subject1_scene1_session1.txt
│   │   ├── session2
│   │   │   └── subject1_scene1_session2.txt
│   │   └── session3
│   │       └── subject1_scene1_session3.txt
...
```
In the untrimmed video annotation file, each column means `[start timestamp]-[end timestamp]-[action_name]` such as:
```
2018/10/16 13:33:45.170-2018/10/16 13:33:49.891-standing
2018/10/16 13:33:55.362-2018/10/16 13:34:00.323-crouching
2018/10/16 13:34:06.132-2018/10/16 13:34:14.522-walking
2018/10/16 13:34:19.402-2018/10/16 13:34:25.114-running
2018/10/16 13:34:33.226-2018/10/16 13:34:38.762-checking_time
2018/10/16 13:34:46.450-2018/10/16 13:34:51.698-waving_hand
2018/10/16 13:34:57.226-2018/10/16 13:35:04.075-using_phone
...
```

### Pose keypoints related frame extraction
Human pose keypoints data are stored under `trimmed/pose` folder. Keypoints are provide with a json format by using [openpifpaf](https://github.com/vita-epfl/openpifpaf) with a manually check. All coordinates are in pixel coordinates. The `keypoints` entry is in COCO format with triples of `(x, y, c)` (`c` for confidence) for every joint as listed under `coco-person-keypoints`. To get the same frame index number stored in json that corresponded with the extracted keypoints, below is a sample script for using:
```
python utils/pose_frame_extraction.py
```

### Toolkit for sensor processing
For the entry of sensor data processing, we provide a complete example to show the way of creating time-series sliding window data, dealing with datetime merging across different type of sensor with different sampling rate if an early fusion needed, and use the pre-processed data for training/testing a time-series classifier. When `MMAct trimmed cross-scene dataset` and `MMAct untrimmed cross-session dataset` prepared over, to run the classifier ([InceptionTime](https://github.com/hfawaz/InceptionTime)) training and testing example as:
```
python utils/time_series_classifiers.py
```

### Evaluation
To evaluate Task1 Action Recognition with validation set, run:
```
python evaluation/eval_mmact_trimmed.py --gt ground_truth_file --pred prediction_file
```
Eamaple:
```
python evaluation/eval_mmact_trimmed.py --gt trimmed_val_view_gt.json --pred trimmed_val_view_sample_submission.json
```

To evaluate Task2 Temporal Localization with validation set, run:
```
python evaluation/eval_mmact_untrimmed.py --gt ground_truth_file --pred prediction_file
```
Eamaple:
```
python evaluation/eval_mmact_untrimmed.py --gt untrimmed_val_gt.json --pred untrimmed_val_sample_submission.json
```

### Sample submission
For Task1 Action Recognition, user needs to submit two results on `cross-view` and `cross-scene`,respectively.
Both of the two splits submission files are the same format as follows, 
```
{
  "results": {
    "nljxzmeshydtlonl": [
      {
        "label": "walking", #one prediction per video is required
        "score": 0.5
      }
    ],
    "hvuapypvzwsjutrf": [
      {
        "label": "talking",
        "score": 0.5
      }
    ],
    "hiukqqolgmtcnisi": [
      {
        "label": "throwing",
        "score": 0.5
      }
    ]
  }
}
```

For Task2 Temporal Localization with validation set,the format example is:
```
{
  "results": {
    "mynbiqpmzjplsgqe": [{
        "label": "standing",
        "score": 0.40685554496254395,
        "segment": [
          62.03, #start seconds, 0.0 is the starting time of the given video.
          66.32  #end seconds
        ]
      },
      {
        "label": "crouching",
        "score": 0.5805843080181547,
        "segment": [
          70.58,
          75.12
        ]
      }
    ]
  }
}
```

### Reference
Please cite the following paper if you use the code or dataset.
```
@InProceedings{Kong_2019_ICCV,
          author = {Kong, Quan and Wu, Ziming and Deng, Ziwei and Klinkigt, Martin and Tong, Bin and Murakami, Tomokazu},
          title = {MMAct: A Large-Scale Dataset for Cross Modal Human Action Understanding},
          booktitle = {The IEEE International Conference on Computer Vision (ICCV)},
          month = {October},
          year = {2019}
        }
```
