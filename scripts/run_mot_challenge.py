""" run_mot_challenge.py

Run example:
run_mot_challenge.py --USE_PARALLEL False --METRICS Hota --TRACKERS_TO_EVAL Lif_T

Command Line Arguments: Defaults, # Comments
    Eval arguments:
        'USE_PARALLEL': False,
        'NUM_PARALLEL_CORES': 8,
        'BREAK_ON_ERROR': True,
        'PRINT_RESULTS': True,
        'PRINT_ONLY_COMBINED': False,
        'PRINT_CONFIG': True,
        'TIME_PROGRESS': True,
        'OUTPUT_SUMMARY': True,
        'OUTPUT_DETAILED': True,
        'PLOT_CURVES': True,
    Dataset arguments:
        'GT_FOLDER': os.path.join(code_path, 'data/gt/mot_challenge/'),  # Location of GT data
        'TRACKERS_FOLDER': os.path.join(code_path, 'data/trackers/mot_challenge/'),  # Trackers location
        'OUTPUT_FOLDER': None,  # Where to save eval results (if None, same as TRACKERS_FOLDER)
        'TRACKERS_TO_EVAL': None,  # Filenames of trackers to eval (if None, all in folder)
        'CLASSES_TO_EVAL': ['pedestrian'],  # Valid: ['pedestrian']
        'BENCHMARK': 'MOT17',  # Valid: 'MOT17', 'MOT16', 'MOT20', 'MOT15'
        'SPLIT_TO_EVAL': 'train',  # Valid: 'train', 'test', 'all'
        'INPUT_AS_ZIP': False,  # Whether tracker input files are zipped
        'PRINT_CONFIG': True,  # Whether to print current config
        'DO_PREPROC': True,  # Whether to perform preprocessing (never done for 2D_MOT_2015)
        'TRACKER_SUB_FOLDER': 'data',  # Tracker files are in TRACKER_FOLDER/tracker_name/TRACKER_SUB_FOLDER
        'OUTPUT_SUB_FOLDER': '',  # Output files are saved in OUTPUT_FOLDER/tracker_name/OUTPUT_SUB_FOLDER
    Metric arguments:
        'METRICS': ['HOTA', 'CLEAR', 'Identity', 'VACE']
"""

import sys
import os
import argparse
from multiprocessing import freeze_support
from time import perf_counter

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import trackeval  # noqa: E402
from trackeval import extract_frame  # noqa: E402

if __name__ == '__main__':
    freeze_support()

    # Command line interface:
    default_eval_config = trackeval.Evaluator.get_default_eval_config()
    default_eval_config['DISPLAY_LESS_PROGRESS'] = False
    default_dataset_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    default_metrics_config = {'METRICS': ['HOTA', 'CLEAR', 'Identity'], 'THRESHOLD': 0.5}
    default_extractor_config = extract_frame.get_default_extractor_config()

    # Merge default configs
    config = {**default_eval_config, **default_dataset_config, **default_metrics_config, **default_extractor_config}
    parser = argparse.ArgumentParser()
    for setting in config.keys():
        if type(config[setting]) == list or type(config[setting]) == type(None):
            parser.add_argument("--" + setting, nargs='+')
        else:
            parser.add_argument("--" + setting)

    args = parser.parse_args().__dict__
    for setting in args.keys():
        if args[setting] is not None:
            if type(config[setting]) == type(True):
                if args[setting] == 'True':
                    x = True
                elif args[setting] == 'False':
                    x = False
                else:
                    raise Exception('Command line parameter ' + setting + 'must be True or False')
            elif type(config[setting]) == type(1):
                x = int(args[setting])
            elif type(args[setting]) == type(None):
                x = None
            elif setting == 'SEQ_INFO':
                x = dict(zip(args[setting], [None] * len(args[setting])))
            else:
                x = args[setting]
            config[setting] = x
    eval_config = {k: v for k, v in config.items() if k in default_eval_config.keys()}
    dataset_config = {k: v for k, v in config.items() if k in default_dataset_config.keys()}
    metrics_config = {k: v for k, v in config.items() if k in default_metrics_config.keys()}
    extractor_config = {k: v for k, v in config.items() if k in default_extractor_config.keys()}

    # Run code
    evaluator = trackeval.Evaluator(eval_config)
    dataset_list = [trackeval.datasets.MotChallenge2DBox(dataset_config)]
    metrics_list = []
    for metric in [trackeval.metrics.HOTA, trackeval.metrics.CLEAR, trackeval.metrics.Identity, trackeval.metrics.VACE]:
        if metric.get_name() in metrics_config['METRICS']:
            metrics_list.append(metric(metrics_config))
    if len(metrics_list) == 0:
        raise Exception('No metrics selected for evaluation')

    # Prepare for extractor information
    extr_bool = [False, False]
    if len(extractor_config['EXTRACTOR']) > 0:
        for elem in extractor_config['EXTRACTOR']:
            if elem == 'FP':
                trackeval.metrics.clear.fp_dataset = True
                extr_bool[0] = True
            else:
                trackeval.metrics.clear.fn_dataset = True
                extr_bool[1] = True

    # Prepare for heatmap information
    heatmap_bool = [False, False, False, False, False]
    if len(extractor_config['HEATMAP']) > 0:
        for elem in extractor_config['HEATMAP']:
            if elem == 'FP':
                trackeval.metrics.clear.fp_dataset = True
                heatmap_bool[0] = True
            elif elem == 'FN':
                trackeval.metrics.clear.fn_dataset = True
                heatmap_bool[1] = True
            elif elem == 'PRED':
                heatmap_bool[2] = True
            elif elem == 'IDSW':  # Son add this
                heatmap_bool[3] = True
            else:
                heatmap_bool[4] = True

    if extractor_config['ID_SWITCH']:
        trackeval.metrics.clear.idsw = True

    evaluator.evaluate(dataset_list, metrics_list)

    for gt_filepath, tracker_filepath, tracker_name, seq_name in dataset_list[0].get_files_loc_and_names():
        # Update filepath
        start = perf_counter()
        for key in extract_frame.filepath.keys():
            if key == 'GT_FILE':
                extract_frame.filepath[key] = gt_filepath
                continue
            elif key == 'TRACKER_FILE':
                extract_frame.filepath[key] = tracker_filepath
                continue
            extract_frame.filepath[key] = extract_frame.filepath[key].format(tracker_name, seq_name)

        # Update global vars in extract_frame.py
        extract_frame.tracker_name = tracker_name
        extract_frame.seq_name = seq_name
        extract_frame.start_pt = len('boxdetails') + len(tracker_name) + len(seq_name) + len('/') * 3

        print(extract_frame.filepath)
        # frame_storage = extract_frame.read_video()
        extract_frame.read_video()
        # Get frames
        extract_frame.get_square_frame(extr_bool)
        # Get heatmap
        extract_frame.get_heatmap(heatmap_bool)
        # Get idsw
        extract_frame.get_idsw_frame(trackeval.metrics.clear.idsw, tracker_filepath)

        # Return to initial dict
        extract_frame.filepath = extract_frame.copy_filepath.copy()
        print("Elapsed time: ", perf_counter() - start)