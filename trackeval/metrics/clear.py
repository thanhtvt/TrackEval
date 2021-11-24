import os
import numpy as np
from scipy.optimize import linear_sum_assignment
from ._base_metric import _BaseMetric
from .. import _timing
from .. import utils


# Global variables
fn_dataset = False
fp_dataset = False
idsw = False


class CLEAR(_BaseMetric):
    """Class which implements the CLEAR metrics"""

    @staticmethod
    def get_default_config():
        """Default class config values"""
        default_config = {
            'THRESHOLD': 0.5,  # Similarity score threshold required for a TP match. Default 0.5.
            'PRINT_CONFIG': True,  # Whether to print the config information on init. Default: False.
        }
        return default_config

    def __init__(self, config=None):
        super().__init__()
        main_integer_fields = ['CLR_TP', 'CLR_FN', 'CLR_FP', 'IDSW', 'MT', 'PT', 'ML', 'Frag']
        extra_integer_fields = ['CLR_Frames']
        self.integer_fields = main_integer_fields + extra_integer_fields
        main_float_fields = ['MOTA', 'MOTP', 'MODA', 'CLR_Re', 'CLR_Pr', 'MTR', 'PTR', 'MLR', 'sMOTA']
        extra_float_fields = ['CLR_F1', 'FP_per_frame', 'MOTAL', 'MOTP_sum']
        self.float_fields = main_float_fields + extra_float_fields
        self.fields = self.float_fields + self.integer_fields
        self.summed_fields = self.integer_fields + ['MOTP_sum']
        self.summary_fields = main_float_fields + main_integer_fields

        # Configuration options:
        self.config = utils.init_config(config, self.get_default_config(), self.get_name())
        self.threshold = float(self.config['THRESHOLD'])

    @_timing.time
    def eval_sequence(self, data, seq, tracker):
        """Calculates CLEAR metrics for one sequence"""
        # Initialise results
        res = {}
        for field in self.fields:
            res[field] = 0

        # Return result quickly if tracker or gt sequence is empty
        if data['num_tracker_dets'] == 0:
            res['CLR_FN'] = data['num_gt_dets']
            res['ML'] = data['num_gt_ids']
            res['MLR'] = 1.0
            return res
        if data['num_gt_dets'] == 0:
            res['CLR_FP'] = data['num_tracker_dets']
            res['MLR'] = 1.0
            return res

        # Variables counting global association
        num_gt_ids = data['num_gt_ids']
        gt_id_count = np.zeros(num_gt_ids)  # For MT/ML/PT
        gt_matched_count = np.zeros(num_gt_ids)  # For MT/ML/PT
        gt_frag_count = np.zeros(num_gt_ids)  # For Frag

        # Note that IDSWs are counted based on the last time each gt_id was present (any number of frames previously),
        # but are only used in matching to continue current tracks based on the gt_id in the single previous timestep.
        prev_tracker_id = np.nan * np.zeros(num_gt_ids)  # For scoring IDSW
        prev_timestep_tracker_id = np.nan * np.zeros(num_gt_ids)  # For matching IDSW
        num_frame_id_disappear = np.zeros(num_gt_ids)  # For counting no. frames since id's last appearance

        # Get path/to/root_dir
        code_path = utils.get_code_path()

        if fp_dataset:
            fp_path = r'boxdetails/{}/{}'.format(tracker, seq)
            fp_path = os.path.join(code_path, fp_path)
            os.makedirs(fp_path, exist_ok=True)
            filepath = os.path.join(fp_path, 'fp.txt')
            if os.path.isfile(filepath):
                open(filepath, 'r+').truncate(0)
            if os.path.isdir(filepath):
                print(filepath)
            fp_frames_file = open(filepath, 'a')

        if fn_dataset:
            fn_path = r'boxdetails/{}/{}'.format(tracker, seq)
            fn_path = os.path.join(code_path, fn_path)
            os.makedirs(fn_path, exist_ok=True)
            filepath = os.path.join(fn_path, 'fn.txt')
            if os.path.isfile(filepath):
                open(filepath, 'r+').truncate(0)
            fn_frames_file = open(filepath, 'a')

        if idsw:
            idsw_path = 'boxdetails/{}/{}'.format(tracker, seq)
            idsw_path = os.path.join(code_path, idsw_path)
            os.makedirs(idsw_path, exist_ok=True)
            filepath = os.path.join(idsw_path, 'idsw.txt')
            if os.path.isfile(filepath):
                open(filepath, 'r+').truncate(0)
            idsw_file = open(filepath, 'a')

        # Define variable storing previous t value
        old_t = 0
        # Create pred_id.txt (similar format with prediction.txt but with TrackEval's ids)
        pred_id_path = f'boxdetails/{tracker}/{seq}'
        pred_id_path = os.path.join(code_path, pred_id_path)
        os.makedirs(pred_id_path, exist_ok=True)
        filepath = os.path.join(pred_id_path, 'pred_id.txt')
        if os.path.isfile(filepath):
            open(filepath, 'r+').truncate(0)
        if os.path.isdir(filepath):
            print(filepath)
        pred_id_file = open(filepath, 'a')
        
        # Calculate scores for each timestep
        for t, (gt_ids_t, tracker_ids_t) in enumerate(zip(data['gt_ids'], data['tracker_ids'])):
            # Deal with the case that there are no gt_det/tracker_det in a timestep.
            if len(gt_ids_t) == 0:
                res['CLR_FP'] += len(tracker_ids_t)
                # Write file
                if fp_dataset and len(tracker_ids_t) > 0:
                    fp_frames_file.write(str(t + 1))
                    for elem in data['tracker_dets'][t].flatten():
                        fp_frames_file.write(' ' + str(elem))
                    fp_frames_file.write('\n')
                continue
            if len(tracker_ids_t) == 0:
                res['CLR_FN'] += len(gt_ids_t)
                # Write file
                if fn_dataset and len(gt_ids_t) > 0:
                    fn_frames_file.write(str(t + 1))
                    for elem in data['gt_dets'][t].flatten():
                        fn_frames_file.write(' ' + str(elem))
                    fn_frames_file.write('\n')
                gt_id_count[gt_ids_t] += 1
                continue
            
            for i in range(len(tracker_ids_t)):
                pred_id_file.write(str(t + 1) + ' ' + str(tracker_ids_t[i]))
                for elem in data['tracker_dets'][t][i]:
                    pred_id_file.write(' ' + str(elem))
                pred_id_file.write('\n')
                
            # Calc score matrix to first minimise IDSWs from previous frame, and then maximise MOTP secondarily
            similarity = data['similarity_scores'][t]
            score_mat = (tracker_ids_t[np.newaxis, :] == prev_timestep_tracker_id[gt_ids_t[:, np.newaxis]])
            score_mat = 1000 * score_mat + similarity
            score_mat[similarity < self.threshold - np.finfo('float').eps] = 0

            # Hungarian algorithm to find best matches
            match_rows, match_cols = linear_sum_assignment(-score_mat)
            actually_matched_mask = score_mat[match_rows, match_cols] > 0 + np.finfo('float').eps
            match_rows = match_rows[actually_matched_mask]
            match_cols = match_cols[actually_matched_mask]

            matched_gt_ids = gt_ids_t[match_rows]
            matched_tracker_ids = tracker_ids_t[match_cols]

            # Calc IDSW for MOTA
            prev_matched_tracker_ids = prev_tracker_id[matched_gt_ids]
            is_idsw = (np.logical_not(np.isnan(prev_matched_tracker_ids))) & (
                np.not_equal(matched_tracker_ids, prev_matched_tracker_ids))
            res['IDSW'] += np.sum(is_idsw)

            # Update counters for MT/ML/PT/Frag and record for IDSW/Frag for next timestep
            gt_id_count[gt_ids_t] += 1
            gt_matched_count[matched_gt_ids] += 1
            not_previously_tracked = np.isnan(prev_timestep_tracker_id)
            prev_tracker_id[matched_gt_ids] = matched_tracker_ids
            prev_timestep_tracker_id[:] = np.nan
            prev_timestep_tracker_id[matched_gt_ids] = matched_tracker_ids
            currently_tracked = np.logical_not(np.isnan(prev_timestep_tracker_id))
            gt_frag_count += np.logical_and(not_previously_tracked, currently_tracked)
            if t - old_t > 1:
                num_frame_id_disappear += (t - old_t - 1)   # As at frame t, pedestrian already appear

            # Write id switch frames
            if idsw and np.sum(is_idsw) > 0:
                # Get id of pedestrian before and after switching
                idsw_tracker_ids = tuple(np.where(is_idsw == 1))        # pos of switched id in is_idsw
                gt_idsw = matched_gt_ids[idsw_tracker_ids]              # id of human in groundtruth
                prev_idsw = prev_matched_tracker_ids[idsw_tracker_ids]  # id before being switched
                after_idsw = matched_tracker_ids[idsw_tracker_ids]      # id after being switched

                # Get index of id that being switched in prev_tracker_id
                index_matched_idsw = list(np.array(matched_gt_ids)[idsw_tracker_ids])

                # Get disappear frames count
                disappear_count = []
                for elem in index_matched_idsw:
                    disappear_count.append(int(num_frame_id_disappear[elem]) + 2)

                curr_id_to_prev_info = {}

                # Start writing file
                # Write current frame first
                idsw_file.write(str(t + 1))
                for i in range(len(after_idsw)):

                    ids = after_idsw[i]
                    gt_ids = gt_idsw[i]
                    idsw_file.write(' ' + str(gt_ids) + ' ' + str(ids))

                    # Update dictionary
                    curr_id_to_prev_info[ids] = list()
                    curr_id_to_prev_info[ids].append(int(prev_idsw[i]))
                    curr_id_to_prev_info[ids].append(disappear_count[i])

                    idsw_tracker_to_tracker_id = np.where(data['tracker_ids'][t] == int(ids))
                    idsw_tracker_dets = data['tracker_dets'][t][idsw_tracker_to_tracker_id].flatten()
                    for elem in idsw_tracker_dets:
                        idsw_file.write(' ' + str(int(elem)))
                idsw_file.write('\n')

                # Write frame that pedestrian's last appearance
                for id_after_switch in curr_id_to_prev_info.keys():
                    id_before_switch = curr_id_to_prev_info.get(id_after_switch)[0]
                    if curr_id_to_prev_info.get(id_after_switch)[1] == 2:
                        prev_frame = 1  # Handling case idsw happens at consecutive frames
                    else:
                        prev_frame = curr_id_to_prev_info.get(id_after_switch)[1]
                    pos = list(curr_id_to_prev_info.keys()).index(id_after_switch)
                    idsw_file.write(str(t + 1 - prev_frame) + ' ' + str(gt_idsw[pos]) + ' ' + str(id_before_switch))
                    idsw_tracker_to_tracker_id = np.where(data['tracker_ids'][t - prev_frame] == id_before_switch)
                    idsw_tracker_dets = data['tracker_dets'][t - prev_frame][idsw_tracker_to_tracker_id].flatten()
                    for elem in idsw_tracker_dets:
                        idsw_file.write(' ' + str(int(elem)))
                    idsw_file.write('\n')

            if t - old_t == 1:
                num_frame_id_disappear += not_previously_tracked
                num_frame_id_disappear -= (num_frame_id_disappear * currently_tracked)
            old_t = t

            # Calculate and accumulate basic statistics
            num_matches = len(matched_gt_ids)
            res['CLR_TP'] += num_matches
            res['CLR_FN'] += len(gt_ids_t) - num_matches
            res['CLR_FP'] += len(tracker_ids_t) - num_matches

            # Starting to write FN boxes
            if fn_dataset and len(gt_ids_t) != num_matches:
                tracker_conv = {}
                fn_tracker_ids = []
                for idx, val in enumerate(gt_ids_t):
                    tracker_conv[val] = idx
                for val in matched_gt_ids:
                    tracker_conv.pop(val)
                for key in tracker_conv.keys():
                    fn_tracker_ids.append(tracker_conv.get(key))

                fn_tracker_dets_t = data['gt_dets'][t][fn_tracker_ids].flatten()
                fn_frames_file.write(str(t + 1))
                for elem in fn_tracker_dets_t:
                    fn_frames_file.write(' ' + str(elem))
                fn_frames_file.write('\n')

            # Starting to write fp boxes
            if fp_dataset and len(tracker_ids_t) != num_matches:
                tracker_conv = {}
                fp_tracker_ids = []
                for idx, val in enumerate(tracker_ids_t):
                    tracker_conv[val] = idx
                for val in matched_tracker_ids:
                    tracker_conv.pop(val)
                for key in tracker_conv.keys():
                    fp_tracker_ids.append(tracker_conv.get(key))

                fp_tracker_dets_t = data['tracker_dets'][t][fp_tracker_ids].flatten()
                fp_frames_file.write(str(t + 1))
                for elem in fp_tracker_dets_t:
                    fp_frames_file.write(' ' + str(elem))
                fp_frames_file.write('\n')

            if num_matches > 0:
                res['MOTP_sum'] += sum(similarity[match_rows, match_cols])

        # Close file
        if fp_dataset:
            fp_frames_file.close()
        if fn_dataset:
            fn_frames_file.close()
        if idsw:
            idsw_file.close()

        # Calculate MT/ML/PT/Frag/MOTP
        tracked_ratio = gt_matched_count[gt_id_count > 0] / gt_id_count[gt_id_count > 0]
        res['MT'] = np.sum(np.greater(tracked_ratio, 0.8))
        res['PT'] = np.sum(np.greater_equal(tracked_ratio, 0.2)) - res['MT']
        res['ML'] = num_gt_ids - res['MT'] - res['PT']
        res['Frag'] = np.sum(np.subtract(gt_frag_count[gt_frag_count > 0], 1))
        res['MOTP'] = res['MOTP_sum'] / np.maximum(1.0, res['CLR_TP'])

        res['CLR_Frames'] = data['num_timesteps']

        # Calculate final CLEAR scores
        res = self._compute_final_fields(res)
        return res

    def combine_sequences(self, all_res):
        """Combines metrics across all sequences"""
        res = {}
        for field in self.summed_fields:
            res[field] = self._combine_sum(all_res, field)
        res = self._compute_final_fields(res)
        return res

    def combine_classes_det_averaged(self, all_res):
        """Combines metrics across all classes by averaging over the detection values"""
        res = {}
        for field in self.summed_fields:
            res[field] = self._combine_sum(all_res, field)
        res = self._compute_final_fields(res)
        return res

    def combine_classes_class_averaged(self, all_res, ignore_empty_classes=False):
        """Combines metrics across all classes by averaging over the class values.
        If 'ignore_empty_classes' is True, then it only sums over classes with at least one gt or predicted detection.
        """
        res = {}
        for field in self.integer_fields:
            if ignore_empty_classes:
                res[field] = self._combine_sum(
                    {k: v for k, v in all_res.items() if v['CLR_TP'] + v['CLR_FN'] + v['CLR_FP'] > 0}, field)
            else:
                res[field] = self._combine_sum({k: v for k, v in all_res.items()}, field)
        for field in self.float_fields:
            if ignore_empty_classes:
                res[field] = np.mean(
                    [v[field] for v in all_res.values() if v['CLR_TP'] + v['CLR_FN'] + v['CLR_FP'] > 0], axis=0)
            else:
                res[field] = np.mean([v[field] for v in all_res.values()], axis=0)
        return res

    @staticmethod
    def _compute_final_fields(res):
        """Calculate sub-metric ('field') values which only depend on other sub-metric values.
        This function is used both for both per-sequence calculation, and in combining values across sequences.
        """
        num_gt_ids = res['MT'] + res['ML'] + res['PT']
        res['MTR'] = res['MT'] / np.maximum(1.0, num_gt_ids)
        res['MLR'] = res['ML'] / np.maximum(1.0, num_gt_ids)
        res['PTR'] = res['PT'] / np.maximum(1.0, num_gt_ids)
        res['CLR_Re'] = res['CLR_TP'] / np.maximum(1.0, res['CLR_TP'] + res['CLR_FN'])
        res['CLR_Pr'] = res['CLR_TP'] / np.maximum(1.0, res['CLR_TP'] + res['CLR_FP'])
        res['MODA'] = (res['CLR_TP'] - res['CLR_FP']) / np.maximum(1.0, res['CLR_TP'] + res['CLR_FN'])
        res['MOTA'] = (res['CLR_TP'] - res['CLR_FP'] - res['IDSW']) / np.maximum(1.0, res['CLR_TP'] + res['CLR_FN'])
        res['MOTP'] = res['MOTP_sum'] / np.maximum(1.0, res['CLR_TP'])
        res['sMOTA'] = (res['MOTP_sum'] - res['CLR_FP'] - res['IDSW']) / np.maximum(1.0, res['CLR_TP'] + res['CLR_FN'])

        res['CLR_F1'] = res['CLR_TP'] / np.maximum(1.0, res['CLR_TP'] + 0.5 * res['CLR_FN'] + 0.5 * res['CLR_FP'])
        res['FP_per_frame'] = res['CLR_FP'] / np.maximum(1.0, res['CLR_Frames'])
        safe_log_idsw = np.log10(res['IDSW']) if res['IDSW'] > 0 else res['IDSW']
        res['MOTAL'] = (res['CLR_TP'] - res['CLR_FP'] - safe_log_idsw) / np.maximum(1.0, res['CLR_TP'] + res['CLR_FN'])
        return res
