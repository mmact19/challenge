"""
MMAct Trimmed Video Action Recognitiono evaluation code.
This code is based on ActivityNet Trimmed Activity Recognition Task evaluation code.
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

################################################################################
# Metrics
################################################################################

def compute_average_precision_classification(ground_truth, prediction):
    """Compute average precision (classification task) between ground truth and
    predictions data frames. If multiple predictions occurs for the same
    predicted segment, only the one with highest score is matched as
    true positive. This code is greatly inspired by Pascal VOC devkit.
    Parameters
    ----------
    ground_truth : df
        Data frame containing the ground truth instances.
        Required fields: ['video-id']
    prediction : df
        Data frame containing the prediction instances.
        Required fields: ['video-id, 'score']
    Outputs
    -------
    ap : float
        Average precision score.
    """
    npos = float(len(ground_truth))
    lock_gt = np.ones(len(ground_truth)) * -1
    # Sort predictions by decreasing score order.
    sort_idx = prediction['score'].values.argsort()[::-1]
    prediction = prediction.loc[sort_idx].reset_index(drop=True)

    # Initialize true positive and false positive vectors.
    tp = np.zeros(len(prediction))
    fp = np.zeros(len(prediction))

    # Assigning true positive to truly grount truth instances.
    for idx in range(len(prediction)):
        this_pred = prediction.loc[idx]
        gt_idx = ground_truth['video-id'] == this_pred['video-id']
        # Check if there is at least one ground truth in the video associated.
        if not gt_idx.any():
            fp[idx] = 1
            continue
        this_gt = ground_truth.loc[gt_idx].reset_index()
        if lock_gt[this_gt['index']] >= 0:
            fp[idx] = 1
        else:
            tp[idx] = 1
            lock_gt[this_gt['index']] = idx

    # Computing prec-rec
    tp = np.cumsum(tp).astype(np.float)
    fp = np.cumsum(fp).astype(np.float)
    rec = tp / npos
    prec = tp / (tp + fp)
    return interpolated_prec_rec(prec, rec)

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

        # Initialize data frame
        activity_index, cidx = {}, 0
        video_lst, label_lst = [], []

        for videoid, v in data['annotations'].items():
            for ann in v:
                if ann['label'] not in activity_index:
                    activity_index[ann['label']] = cidx
                    cidx += 1
                video_lst.append(videoid)
                label_lst.append(activity_index[ann['label']])
        ground_truth = pd.DataFrame({'video-id': video_lst,
                                     'label': label_lst})
        ground_truth = ground_truth.drop_duplicates().reset_index(drop=True)
        return ground_truth, activity_index

def import_prediction(ground_truth_filename, prediction_filename):
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
        ground_truth, activity_index = import_ground_truth(ground_truth_filename)

        with open(prediction_filename, 'r') as fobj:
            data = json.load(fobj)

        # Initialize data frame
        video_lst, label_lst, score_lst = [], [], []
        for videoid, v in data['results'].items():
            for result in v:
                label = activity_index[result['label']]
                video_lst.append(videoid)
                label_lst.append(label)
                score_lst.append(result['score'])
        prediction = pd.DataFrame({'video-id': video_lst,
                                   'label': label_lst,
                                   'score': score_lst})
        return prediction

def wrapper_compute_average_precision(ground_truth,prediction,activity_index):
    """Computes average precision for each class in the subset.
    """
    ap = np.zeros(len(activity_index.items()))
    for activity, cidx in activity_index.items():
        gt_idx = ground_truth['label'] == cidx
        pred_idx = prediction['label'] == cidx
        ap[cidx] = compute_average_precision_classification(
            ground_truth.loc[gt_idx].reset_index(drop=True),
            prediction.loc[pred_idx].reset_index(drop=True))
    return ap

def main():
    parser = argparse.ArgumentParser(description='Take input of ground truth file and submission file.')
    parser.add_argument('--gt', metavar='gt', type=str, help='ground truth json file path')
    parser.add_argument('--pred', metavar='pred', type=str,  help='prediction file path')
    args = parser.parse_args()

    ground_truth, activity_index = import_ground_truth(args.gt)
    prediction = import_prediction(args.gt,args.pred)

    ap = wrapper_compute_average_precision(ground_truth,prediction,activity_index)

    print ('[RESULTS] Performance on MMAct trimmed video '
           'action recognition task.')
    print('\tMean Average Precision: {}'.format(ap.mean()))

if __name__ == '__main__':
    main()
