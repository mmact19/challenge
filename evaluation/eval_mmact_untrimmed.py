"""
MMAct Temporal Localization evaluation code.
This code is based on ActivityNet Temporal Action Localization evaluation code.
"""
import numpy as np
import os
import pandas as pd
import json
import argparse
from joblib import Parallel, delayed

def interpolated_prec_rec(prec, rec):
    """Interpolated AP - VOCdevkit from VOC 2011.
    """
    mprec = np.hstack([[0], prec, [0]])
    mrec = np.hstack([[0], rec, [1]])
    for i in range(len(mprec) - 1)[::-1]:
        mprec[i] = max(mprec[i], mprec[i + 1])
    idx = np.where(mrec[1::] != mrec[0:-1])[0] + 1
    ap = np.sum((mrec[idx] - mrec[idx - 1]) * mprec[idx])
    return ap

def segment_iou(target_segment, candidate_segments):
    """Compute the temporal intersection over union between a
    target segment and all the test segments.
    Parameters
    ----------
    target_segment : 1d array
        Temporal target segment containing [starting, ending] times.
    candidate_segments : 2d array
        Temporal candidate segments containing N x [starting, ending] times.
    Outputs
    -------
    tiou : 1d array
        Temporal intersection over union score of the N's candidate segments.
    """
    tt1 = np.maximum(target_segment[0], candidate_segments[:, 0])
    tt2 = np.minimum(target_segment[1], candidate_segments[:, 1])
    # Intersection including Non-negative overlap score.
    segments_intersection = (tt2 - tt1).clip(0)
    # Segment union.
    segments_union = (candidate_segments[:, 1] - candidate_segments[:, 0]) \
      + (target_segment[1] - target_segment[0]) - segments_intersection
    # Compute overlap as the ratio of the intersection
    # over union of two segments.
    tIoU = segments_intersection.astype(float) / segments_union
    return tIoU

def import_ground_truth(ground_truth_filename):
        """Reads ground truth file, checks if it is well formatted, and returns
           the ground truth instances and the activity classes.
        Parameters
        ----------
        ground_truth_filename : str
            Full path to the ground truth json file.
        Outputs
        -------
        ground_truth : df
            Data frame containing the ground truth instances.
        activity_index : dict
            Dictionary containing class index.
        """
        with open(ground_truth_filename, 'r') as fobj:
            data = json.load(fobj)
        # Checking format
        #if not all([field in data.keys() for field in self.gt_fields]):
        #    raise IOError('Please input a valid ground truth file.')

        # Read ground truth data.
        activity_index, cidx = {}, 0
        video_lst, t_start_lst, t_end_lst, label_lst = [], [], [], []
        annotations = data['annotations']
        videos = annotations.keys()
        for video_id in videos:
            for ann in annotations[video_id]:
                    if ann['label'] not in activity_index:
                        activity_index[ann['label']] = cidx
                        cidx += 1
                    video_lst.append(video_id)
                    t_start_lst.append(float(ann['segment'][0]))
                    t_end_lst.append(float(ann['segment'][1]))
                    label_lst.append(activity_index[ann['label']])

        ground_truth = pd.DataFrame({'video-id': video_lst,
                                     't-start': t_start_lst,
                                     't-end': t_end_lst,
                                     'label': label_lst})
        return ground_truth, activity_index

def import_prediction(prediction_filename, activity_index):
        """Reads prediction file, checks if it is well formatted, and returns
           the prediction instances.
        Parameters
        ----------
        prediction_filename : str
            Full path to the prediction json file.
        Outputs
        -------
        prediction : df
            Data frame containing the prediction instances.
        """
        with open(prediction_filename, 'r') as fobj:
            data = json.load(fobj)
        # Checking format...
        #if not all([field in data.keys() for field in self.pred_fields]):
        #    raise IOError('Please input a valid prediction file.')

        # Read predictions.
        video_lst, t_start_lst, t_end_lst, label_lst, score_lst = [], [], [], [], []
        predictions = data['results']
        videos = predictions.keys()
        for video_id in videos:
            for ann in predictions[video_id]:
                    #if ann['label'] not in activity_index:
                    #    activity_index[ann['label']] = cidx
                    #    cidx += 1
                    video_lst.append(video_id)
                    t_start_lst.append(float(ann['segment'][0]))
                    t_end_lst.append(float(ann['segment'][1]))
                    score_lst.append(float(ann['score']))
                    label_lst.append(activity_index[ann['label']])
        prediction = pd.DataFrame({'video-id': video_lst,
                                   't-start': t_start_lst,
                                   't-end': t_end_lst,
                                   'label': label_lst,
                                   'score': score_lst})
        return prediction

def get_predictions_with_label(prediction_by_label, label_name, cidx):
        """Get all predicitons of the given label. Return empty DataFrame if there
        is no predcitions with the given label.
        """
        try:
            return prediction_by_label.get_group(cidx).reset_index(drop=True)
        except:
            print('Warning: No predictions of label \'%s\' were provdied.' % label_name)
            return pd.DataFrame()


def compute_average_precision_detection(ground_truth, prediction, tiou_thresholds=np.linspace(0.5, 0.95, 10)):
    """Compute average precision (detection task) between ground truth and
    predictions data frames. If multiple predictions occurs for the same
    predicted segment, only the one with highest score is matches as
    true positive. This code is greatly inspired by Pascal VOC devkit.
    Parameters
    ----------
    ground_truth : df
        Data frame containing the ground truth instances.
        Required fields: ['video-id', 't-start', 't-end']
    prediction : df
        Data frame containing the prediction instances.
        Required fields: ['video-id, 't-start', 't-end', 'score']
    tiou_thresholds : 1darray, optional
        Temporal intersection over union threshold.
    Outputs
    -------
    ap : float
        Average precision score.
    """
    ap = np.zeros(len(tiou_thresholds))
    if prediction.empty:
        return ap

    npos = float(len(ground_truth))
    lock_gt = np.ones((len(tiou_thresholds),len(ground_truth))) * -1
    # Sort predictions by decreasing score order.
    sort_idx = prediction['score'].values.argsort()[::-1]
    prediction = prediction.loc[sort_idx].reset_index(drop=True)

    # Initialize true positive and false positive vectors.
    tp = np.zeros((len(tiou_thresholds), len(prediction)))
    fp = np.zeros((len(tiou_thresholds), len(prediction)))

    # Adaptation to query faster
    ground_truth_gbvn = ground_truth.groupby('video-id')

    # Assigning true positive to truly grount truth instances.
    for idx, this_pred in prediction.iterrows():

        try:
            # Check if there is at least one ground truth in the video associated.
            ground_truth_videoid = ground_truth_gbvn.get_group(this_pred['video-id'])
        except Exception as e:
            fp[:, idx] = 1
            continue

        this_gt = ground_truth_videoid.reset_index()
        tiou_arr = segment_iou(this_pred[['t-start', 't-end']].values,
                               this_gt[['t-start', 't-end']].values)
        # We would like to retrieve the predictions with highest tiou score.
        tiou_sorted_idx = tiou_arr.argsort()[::-1]
        for tidx, tiou_thr in enumerate(tiou_thresholds):
            for jdx in tiou_sorted_idx:
                if tiou_arr[jdx] < tiou_thr:
                    fp[tidx, idx] = 1
                    break
                if lock_gt[tidx, this_gt.loc[jdx]['index']] >= 0:
                    continue
                # Assign as true positive after the filters above.
                tp[tidx, idx] = 1
                lock_gt[tidx, this_gt.loc[jdx]['index']] = idx
                break

            if fp[tidx, idx] == 0 and tp[tidx, idx] == 0:
                fp[tidx, idx] = 1

    tp_cumsum = np.cumsum(tp, axis=1).astype(np.float)
    fp_cumsum = np.cumsum(fp, axis=1).astype(np.float)
    recall_cumsum = tp_cumsum / npos

    precision_cumsum = tp_cumsum / (tp_cumsum + fp_cumsum)

    for tidx in range(len(tiou_thresholds)):
        ap[tidx] = interpolated_prec_rec(precision_cumsum[tidx,:], recall_cumsum[tidx,:])

    return ap


def main():
    parser = argparse.ArgumentParser(description='Take input of ground truth file and submission file.')
    parser.add_argument('--gt', metavar='gt', type=str, help='ground truth json file path')
    parser.add_argument('--pred', metavar='pred', type=str,  help='prediction file path')
    args = parser.parse_args()
    tiou_thresholds = np.linspace(0.5, 0.95, 10)
    #print(args.ground_truth)
    gt, activity_idx = import_ground_truth(args.gt)
    pred = import_prediction(args.pred, activity_idx)
    gt_by_label = gt.groupby('label')
    pred_by_label = pred.groupby('label')
    ap = np.zeros((len(tiou_thresholds), len(activity_idx)))
    results = Parallel(n_jobs=len(activity_idx))(
                    delayed(compute_average_precision_detection)(
                        ground_truth=gt_by_label.get_group(cidx).reset_index(drop=True),
                        prediction=get_predictions_with_label(pred_by_label, label_name, cidx),
                        tiou_thresholds=tiou_thresholds,
                    ) for label_name, cidx in activity_idx.items())
    for i, cidx in enumerate(activity_idx.values()):
        ap[:,cidx] = results[i]
    #print(ap)
    mAP = ap.mean(axis=1)
    #print(mAP)
    average_mAP = mAP.mean()
    print('Performance Evaluation')
    print('\tAverage-mAP: {}'.format(average_mAP))


if __name__ == '__main__':
    main()
