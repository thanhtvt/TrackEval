import os
from re import split
import cv2
import numpy as np
from .utils import get_code_path
from collections import defaultdict
import imageio
from ._timing import time_recorder

"""----General utils----"""

# Global variables
tracker_name = ""
seq_name = ""
start_pt = 0

# File path storage
code_path = get_code_path()
filepath = {'GT_FILE': '',
            'TRACKER_FILE': '',
            'RAW_VIDEO': 'video/{}/{}.mp4',
            'FP_DETAILS': 'boxdetails/{}/{}/fp.txt',
            'FN_DETAILS': 'boxdetails/{}/{}/fn.txt',
            'IDSW_DETAILS': 'boxdetails/{}/{}/idsw.txt',
            'IDSW_HEAT': 'boxdetails/{}/{}/idsw_heatmap.txt',
            'GT_DETAILS': 'boxdetails/{}/{}/gt.txt',
            'PRED_DETAILS': 'boxdetails/{}/{}/pred.txt',
            'EXTRACTOR_OUTPUT': 'output/{}/{}/square_images/',
            'HEATMAP_OUTPUT': 'output/{}/{}/heatmap/',
            'IDSW_OUTPUT': 'output/{}/{}/idsw/',
            'IDSW_GIF': 'output/{}/{}/gif/',
            'IDSW_BBOX_OUTPUT': 'output/{}/{}/idsw/bbox_idsw/',
            'IDSW_ATTACH_OUTPUT': 'output/{}/{}/idsw/attach/'}
copy_filepath = filepath.copy()

@time_recorder
def read_video():
    """Read video with opencv 

    Returns:
        [dict]: Dictionay -> {"frame_idx": frame_array}
    """
    cap = cv2.VideoCapture(filepath['RAW_VIDEO'])
    # frame_storage = {}
    curr_frame = 0 
    
    while True:
        ret, frame = cap.read()
        curr_frame += 1
        # frame_storage[curr_frame] = frame
        if not ret:
            break   
         
    cap.release()
    # return frame_storage

    
    
def get_default_extractor_config():
    """Default frames extractor config"""

    default_config = {
        'EXTRACTOR': [],  # Valid: ['FN', 'FP']
        'HEATMAP': [],  # Valid: ['PRED', 'GT', 'FN', 'FP']
        'ID_SWITCH': False,  # Valid: [True, False]
    }
    return default_config


def put_text(frame, text):
    """Put text on frame

    Input:
        - frame: frame that being written on
        - text: appear on frame
    Output: frame after putting text"""

    # Set up params
    org = (40, 40)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.5
    color = (0, 0, 0)
    thickness = 2
    line_type = cv2.LINE_4

    cv2.putText(frame, text, org, font, font_scale, color, thickness, line_type)

    return frame


def convert_file_format(org_file, destination_file):
    """Convert file format from:

    <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z> to:

    <frame> <bb1_left> <bb1_top> <bb1_width> <bb1_height> <bb2_left> <bb2_top> <bb2_width> <bb2_height> ..."""

    # Get needed infos for file rewriting
    file = open(org_file, 'r').read()
    frame_to_boxes = {}
    for line in list(file.split('\n')):
        if len(line) < 2:
            continue
        box = [int(float(elem)) for elem in line.split(',')[2:6]]
        idx = int(line.split(',')[0])
        if idx not in frame_to_boxes.keys():
            frame_to_boxes[idx] = box
            continue
        frame_to_boxes[idx].extend(box)

    # Sort dictionary
    sorted_frame_to_boxes = dict(sorted(frame_to_boxes.items()))

    # Create file with new format
    if os.path.isfile(destination_file):
        open(destination_file, 'r+').truncate(0)
    dest_file = open(destination_file, 'a')

    # Write file
    for key, val in sorted_frame_to_boxes.items():
        dest_file.write(str(key))
        for elem in val:
            dest_file.write(' ' + str(elem))
        dest_file.write('\n')


def delete_images(directory):
    if not os.path.isdir(directory):
        return
    for file in os.listdir(directory):
        if file.endswith('.jpg'):
            os.remove(os.path.join(directory, file))

def save_fig(directory, image, filename):
    os.makedirs(directory, exist_ok=True)

    addon = 0
    prefix_name = filename.split('.')[0]
    temp_name = filename
    # Check if file has already existed
    while True:
        if os.path.isfile(temp_name):
            temp_name = prefix_name + '_' + str(addon) + '.jpg'
            addon += 1
            continue
        else:
            filename = temp_name
            break
        
    image = cv2.resize(image, (0, 0), fx = 0.5, fy = 0.5)
    cv2.imwrite(filename, image)


def get_bounding_box(image, ids_boxes, frame_no):
    for i in range(len(ids_boxes)):
        if i % 6 == 5:
            bbox_id_gt = ids_boxes[i - 5]
            bbox_id = ids_boxes[i - 4]
            bbox_left = ids_boxes[i - 3] if ids_boxes[i - 3] >= 0 else 0
            bbox_top = ids_boxes[i - 2] if ids_boxes[i - 2] >= 0 else 0
            bbox_right = ids_boxes[i - 1]
            bbox_bottom = ids_boxes[i]

            # Cut bounding box
            bounding_box = image[bbox_top:bbox_bottom, bbox_left:bbox_right]

            directory = filepath['IDSW_BBOX_OUTPUT']
            filename = '{}/{}_{}_{}.jpg'.format(directory, str(bbox_id_gt).zfill(2), str(frame_no).zfill(3),
                                                str(bbox_id).zfill(2))
            save_fig(directory, bounding_box, filename)

@time_recorder
def attach_images(images_dir, output_dir, dim):
    delete_images(output_dir)

    images_path = []

    for img_path in os.listdir(images_dir):
        if img_path.endswith('.jpg'):
            images_path.append(os.path.join(images_dir, img_path))

    images_path.sort()

    for i in range(0, len(images_path) - 1, 2):
        img1 = cv2.resize(cv2.imread(images_path[i]), dim)
        img2 = cv2.resize(cv2.imread(images_path[i + 1]), dim)
        new_img = cv2.vconcat([img1, img2])
        img_name = split(r'[_/.\\]', images_path[i])[4] + '_' + split(r'[_/.\\]', images_path[i])[5] + '_' + \
                   split(r'[_/.\\]', images_path[i + 1])[5] + '.jpg'
        filename = os.path.join(output_dir, img_name)
        save_fig(output_dir, new_img, filename)


"""----Functions for creating square boxes----"""

@time_recorder
def convert_bbox_info(f_frame_len, bbox_info):
    """Convert bbox old information: <bb_left>, <bb_top>, <bb_width>, <bb_height>
    to new form to fit cv2.rectangle() inputs: <bb_left>, <bb_top>, <bb_right>, <bb_bottom>"""

    total_length = 0
    bbox = list(bbox_info)
    for key in f_frame_len.keys():
        total_length += f_frame_len.get(key)
    for i in range(total_length):
        if i % 4 == 2 or i % 4 == 3:
            bbox[i] = bbox[i - 2] + bbox[i]

    return bbox


def read_file(path):
    """This function read file with given path.

    Output:
        - f_frame_len: A dictionary whose key is a frame index, value is a length of box info
        - bbox_info: A list containing left coordinate, top coordinate, width and height of box"""
    
    f = open(path, 'r').read()
    f_frame_len = {}
    bbox_info = []
    for line in f.split('\n'):
        first = True
        if len(line) > 0:
            for elem in line.split():
                if first:
                    f_frame_len[int(elem)] = len(line.split()) - 1
                    first = False
                    continue
                bbox_info.append(float(elem))

    return f_frame_len, bbox_info


def draw_rectangle(image, length, bbox, bbox_idx, color=(0, 0, 0)):
    """Draw a rectangle with given bbox info.

    Input:
        - image: Frame to draw on
        - length: Number of info (4 info/box)
        - bbox: A list containing rectangles' info to draw
        - bbox_idx: Just a idx that smaller than len(bbox)
    Output: Frame that has been drawn on"""

    for temp_idx in range(length):
        if temp_idx % 4 == 3:
            bbox_left = int(round(bbox[bbox_idx + temp_idx - 3]))
            bbox_top = int(round(bbox[bbox_idx + temp_idx - 2]))
            bbox_right = int(round(bbox[bbox_idx + temp_idx - 1]))
            bbox_bottom = int(round(bbox[bbox_idx + temp_idx]))

            # Set up params
            left_top_pt = (bbox_left, bbox_top)
            right_bottom_pt = (bbox_right, bbox_bottom)
            thickness = 5

            image = cv2.rectangle(image, left_top_pt, right_bottom_pt, color, thickness)
    return image

@time_recorder
def get_square_frame_utils(path_to_read):
    """Get frames utils"""
    video_path = filepath['RAW_VIDEO']
    cap = cv2.VideoCapture(video_path)
    curr_frame = 0
    frame_idx = 0
    bbox_idx = 0
    pred_bbox_idx = 0

    f_frame_len, bbox_info = read_file(path_to_read)
    bbox = convert_bbox_info(f_frame_len, bbox_info)

    f_frame = list(f_frame_len)
    # Total number of FP/FN frames
    size = len(f_frame_len)

    directory = filepath['EXTRACTOR_OUTPUT'] + path_to_read[start_pt:-4] + '/'
    delete_images(directory)

    # Read prediction file
    if not os.path.exists(filepath['PRED_DETAILS']):
        convert_file_format(filepath['TRACKER_FILE'], filepath['PRED_DETAILS'])
    pred_frame_len, pred_bbox = read_file(filepath['PRED_DETAILS'])
    pred_bbox = convert_bbox_info(pred_frame_len, pred_bbox)

    
    while True:
        ret, frame = cap.read()
        curr_frame += 1
        if not ret:
            break

        pred_length = pred_frame_len.get(curr_frame)
        if frame_idx < size and curr_frame == f_frame[frame_idx]:
            length = f_frame_len.get(curr_frame)

            # Draw and write frames
            frame = draw_rectangle(frame, pred_length, pred_bbox, pred_bbox_idx)        # draw pred
            frame = draw_rectangle(frame, length, bbox, bbox_idx, color=(0, 0, 255))    # draw FN/FP
            frame = put_text(frame, path_to_read[11:-4].upper())
            

            filename = directory + str(curr_frame) + '.jpg'
            save_fig(directory, frame, filename)

            # Update params
            frame_idx += 1
            bbox_idx += length
        pred_bbox_idx += pred_length

    cap.release()

@time_recorder
def get_square_frame(detect):
    """Draw a rectangle on and write frames that contain FP boxes to chosen folder"""

    # Change current working directory to parent dir
    if os.getcwd() != code_path:
        os.chdir(code_path)

    if detect[0]:
        print('\nDetecting FP boxes of {}/{}...'.format(tracker_name, seq_name))
        get_square_frame_utils(filepath['FP_DETAILS'])
        print('Finished!!')

    if detect[1]:
        print('\nDetecting FN boxes of {}/{}...'.format(tracker_name, seq_name))
        get_square_frame_utils(filepath['FN_DETAILS'])
        print('Finished!!')


"""-----Functions for creating heatmap----"""


def create_heatmap(frame, bbox):
    """Create heatmap with given input:
        - frame: considered frame
        - length: Number of info (4 info/box)
        - bbox: A list containing rectangles' info to draw
        - bbox_idx: Just a idx that smaller than len(bbox)
    Output: frame after being drawn on"""

    # Create overlay
    overlay_img = np.full(frame.shape, 255, dtype=np.uint8)
    
    frame = cv2.addWeighted(overlay_img, 0.4, frame, 0.6, 0)

    for idx in range(len(bbox)):
        if idx % 4 == 3:
            bbox_x_center = int(round(bbox[idx - 3] + bbox[idx - 1] / 2))
            bbox_y_center = int(round(bbox[idx - 2] + bbox[idx] / 2))

            # Set up params
            pt = (bbox_x_center, bbox_y_center)
            radius = 0
            color = (0, 0, 0)
            thickness = 4

            frame = cv2.circle(frame, pt, radius, color, thickness)

    return frame


def get_heatmap_utils(path_to_read):
    """Utils of get_heatmap function"""

    cap = cv2.VideoCapture(filepath['RAW_VIDEO'])
    running = True
    _, bbox = read_file(path_to_read)
    directory = filepath['HEATMAP_OUTPUT']
    os.makedirs(directory, exist_ok=True)

    while running:
        ret, frame = cap.read()

        # Draw and write frames
        frame = create_heatmap(frame, bbox)
        # cv2.imshow(path_to_read, frame)

        frame = put_text(frame, path_to_read[11:-4].upper())

        filename = directory + path_to_read[start_pt:-4] + '.jpg'
        print("filename: ", os.path.abspath(filename))
        cv2.imwrite(filename, frame)

        running = False

        # if cv2.waitKey(0) & 0xFF == ord('q'):
        #     break

        if not ret:
            break

    # cv2.destroyAllWindows()
    cap.release()

@time_recorder
def get_heatmap(heat):
    """Call this function to get heatmap of wanted type(s)"""

    # Change current working directory to parent dir
    if os.getcwd() != code_path:
        os.chdir(code_path)

    if heat[0]:
        print('\nGetting heatmap of {}/{}/FP...'.format(tracker_name, seq_name))
        get_heatmap_utils(filepath['FP_DETAILS'])
        print('Finished!!')

    if heat[1]:
        print('\nGetting heatmap of {}/{}/FN...')
        get_heatmap_utils(filepath['FN_DETAILS'])
        print('Finished!!')

    if heat[2]:
        print('\nGetting heatmap of {}/{}/Prediction...'.format(tracker_name, seq_name))
        convert_file_format(filepath['TRACKER_FILE'], filepath['PRED_DETAILS'])
        get_heatmap_utils(filepath['PRED_DETAILS'])
        print('Finished!!')

    if heat[3]:  # son add this
        print('\nGetting heatmap of {}/{}/IDSW...'.format(tracker_name, seq_name))
        convert_idsw_to_heatmap_format(filepath['IDSW_DETAILS'], filepath['IDSW_HEAT'])
        get_heatmap_utils(filepath['IDSW_HEAT'])
        print('Finished!!')

    if heat[4]:
        print('\nGetting heatmap of {}-{}-Ground truth...'.format(tracker_name, seq_name))
        convert_file_format(filepath['GT_FILE'], filepath['GT_DETAILS'])
        get_heatmap_utils(filepath['GT_DETAILS'])
        print('Finished!!')


"""Functions for getting id-switch frames"""


def read_idsw_file(filepath):
    """Similar use of read_file() function"""

    frame_to_ids_boxes = {}
    f = open(filepath, 'r').read()
    for line in f.split('\n'):
        if len(line) < 2:
            continue
        frame = 0
        first = True
        for num_str in line.split():
            num = int(num_str)
            if first:
                frame = num
                if frame not in frame_to_ids_boxes.keys():
                    frame_to_ids_boxes[frame] = []
                first = False
                continue
            frame_to_ids_boxes[frame].append(num)

    frame_to_ids_boxes = dict(sorted(frame_to_ids_boxes.items()))
    return frame_to_ids_boxes


"""
author: Son
modify: 7/9/2021
purpose: add heatmap for idsw
"""


def convert_idsw_to_heatmap_format(filepath, dest_file):
    """Convert idsw format to heatmap formats
    idsw: <frame> <id1_gt> <id1> <bb1_left> <bb1_top> <bb1_width> <bb1_height> <id2_gt> <id2> <bb2_left> <bb2_top> <bb2_width> ...
    heatmap:  <frame> <bb1_left> <bb1_top> <bb1_width> <bb1_height> <bb2_left> <bb2_top> <bb2_width> <bb2_height> ...

    Args:
        filepath ([str]): idsw path
        dest_file : save file path
    """
    from collections import defaultdict

    ids_group = defaultdict(list)
    obj_infos = []

    with open(filepath, "r") as f:
        for line in f:
            # <frame> <id1_gt> <id1> <bb1_left> <bb1_top> <bb1_width> <bb1_height> <id2_gt> <id2> <bb2_left> <bb2_top> <bb2_width>
            p = line.rstrip().split(" ")
            p = list(map(int, p))
            # get number of objects in current frame 
            num_obj = int((len(p) - 1) / 6)
            for idx in range(num_obj):
                # <frame> <id1_gt> <id1> <bb1_left> <bb1_top> <bb1_width> <bb1_height>\n <frame> <id2_gt> <id2> <bb2_left> <bb2_top> <bb2_width>
                obj_infos.append([p[0]] + p[1 + 6 * idx: 7 + 6 * idx])

    for obj in obj_infos:
        ids_group[obj[1]].append(obj)

    # print(ids_group)
    with open(dest_file, 'w') as f:
        for _, objs in ids_group.items():
            objs = sorted(objs, key=lambda x: x[0])
            if len(objs) % 2 == 0:
                for i in range(1, len(objs), 2):
                    tmp = list(map(str, objs[i]))
                    # print(tmp)
                    line = tmp[0] + " " + tmp[3] + " " + tmp[4] + " " + tmp[5] + " " + tmp[6] + "\n"
                    f.write(line)

    return 1


# ------------------ end ------------------


def convert_idsw_bbox_info(frame_to_ids_boxes):
    """Similar use of convert_bbox_info() function"""

    copy_frame = {}
    for frame in frame_to_ids_boxes.keys():
        copy_frame[frame] = []
        ids_and_boxes = frame_to_ids_boxes.get(frame)
        for idx, elem in enumerate(ids_and_boxes):
            if idx % 6 == 4 or idx % 6 == 5:
                ids_and_boxes[idx] = ids_and_boxes[idx] + ids_and_boxes[idx - 2]
        copy_frame[frame].extend(ids_and_boxes)

    return copy_frame


def draw_idsw_rectangle(image, ids_boxes, frame_no):
    """Draw boxes and label id for each box"""

    # General params
    color = (0, 0, 255)
    thickness_box = 6
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    thicknes_id = 2
    line_type = cv2.LINE_4

    for i in range(len(ids_boxes)):
        if i % 6 == 5:
            bbox_id_gt = ids_boxes[i - 5]
            bbox_id = ids_boxes[i - 4]
            bbox_left = ids_boxes[i - 3]
            bbox_top = ids_boxes[i - 2]
            bbox_right = ids_boxes[i - 1]
            bbox_bottom = ids_boxes[i]

            # Params for box
            left_top_pt = (bbox_left, bbox_top)
            right_bottom_pt = (bbox_right, bbox_bottom)

            # Params for id
            org = (bbox_left, bbox_top - 5)

            # Create copy
            image_copy = np.copy(image)

            # Draw
            image_copy = cv2.rectangle(image_copy, left_top_pt, right_bottom_pt, color, thickness_box)
            cv2.putText(image_copy, str(bbox_id), org, font, font_scale, color, thicknes_id, line_type)
            put_text(image_copy, str(frame_no))

            directory = filepath['IDSW_OUTPUT']
            filename = '{}/{}_{}_{}.jpg'.format(directory, str(bbox_id_gt).zfill(2), str(frame_no).zfill(3),
                                                str(bbox_id).zfill(2))
            save_fig(directory, image_copy, filename)

"""
author: Son
modify: 28/9/2021
purpose: add gif for idsw
"""    

def read_tracker_file(file_path):
    """Convert mot file to frame group format and mapping pred id to trackEval id

    Args:
        file_path ([type]): [description]

    Returns:
        [type]: [description]
    """
    lines = []
    spliter = None
    with open(file_path, 'r') as f:
        for line in f:
            if(spliter is None):
                spliter = "," if "," in line else " " 
            line = line.rstrip().split(spliter)
            line = list(map(float, line)) # convert str to float 
            line = list(map(int, line)) # convert to int for opencv visualine
            lines.append(line)
    
    # group obj in frames: 
    frame_groups = defaultdict(list)
    ids_pred = set()
    for line in lines:
        ids_pred.add(line[1])
        frame_groups[line[0]].append(line[:6]) 
    
    ids_pred = sorted(list(ids_pred))
    idsw_mapper = {k:v for k, v in zip(ids_pred, range(len(ids_pred)))} # map from id in prediction file to id in trackeval cuz id in track eval start from 1 
    
    # print("IDSW MAPPER: ", idsw_mapper)
    # map to new id 
    for k, v in frame_groups.items():
        obj_in_frames = [] 
        for obj in v:
            obj[1] = idsw_mapper[obj[1]] # id index 1 
            obj_in_frames.append(obj)
        frame_groups[k] = obj_in_frames
        
    return frame_groups

def group_obj_idsw_gt(frame_to_ids_boxes):
    idsw_gt_groups = defaultdict(list)
    
    for k, v in frame_to_ids_boxes.items():
        for i in range(0, len(v), 6):
            id_gt = v[i]
            idsw_gt_groups[id_gt].append([k, v[i+1]]) # [frame, id_pred] 
    return idsw_gt_groups
        
def is_in_frame_range(frame_idx, key):
    start_frame, end_frame = list(map(int, key.split("_")))
    if(frame_idx >= start_frame and frame_idx <= end_frame):
        return True 
    return False  

def draw_gif_frame(image, bbox, frame_no):
    """Draw a rectangle with given bbox info.

    Input:
        - image: Frame to draw on
        - length: Number of info (4 info/box)
        - bbox: A list containing rectangles' info to draw -> frame id x y w h 
    Output: Frame that has been drawn on"""

    obj_id = bbox[1]
    bbox_left = int(bbox[2])
    bbox_top = int(bbox[3])
    bbox_right = bbox_left + int(bbox[4])
    bbox_bottom = bbox_top + int(bbox[5])

    # Set up params
    left_top_pt = (bbox_left, bbox_top)
    right_bottom_pt = (bbox_right, bbox_bottom)
    color = (255, 0, 0)
    thickness = 8
    org = (bbox_left, bbox_top - 5)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    thicknes_id = 3
    line_type = cv2.LINE_4

    cv2.rectangle(image, left_top_pt, right_bottom_pt, color, thickness)
    cv2.putText(image, str(obj_id), org, font, font_scale, color, thicknes_id, line_type)
    put_text(image, str(frame_no))
    # test
    # cv2.imshow("img", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows() 
    
    return image

def draw_gif_all_frames(frames, start_frame, frame_tracker_groups, idsw_val):
    """Draw idsw box for all frame in gif

    Args:
        frames ([type]): [description]
        start_frame ([type]): [description]
        frame_tracker_groups ([type]): [description]
        idsw_val ([type]): [description]

    Returns:
        [list]: Frames have been drawn
    """
    drawn_frames = []
    for idx in range(len(frames)):
        cur_objs = frame_tracker_groups[start_frame + idx] 
        for obj in cur_objs:
            if((obj[1] == idsw_val[0] and idx != len(frames)-1) or (obj[1] == idsw_val[1] and idx == len(frames)-1)):
                new_frame = draw_gif_frame(frames[idx], obj, start_frame + idx)
                drawn_frames.append(new_frame)
    
    return drawn_frames
        
@time_recorder
def get_idsw_gif(idsw_gt_groups, frame_range, frame_tracker_groups):
    """Get gif images for all idsw case

    Args:
        idsw_gt_groups ([type]): {"id_gt": [[frame, id_pred], ...]}
        frame_range ([type]): {"{start_frame}_{end_frame}": []}
        frame_tracker_groups ([type]): {"frame": [bbox1, bbox2, ...]}
    """
    gif_save_path = filepath['IDSW_GIF']
    
    for id_gt, v in idsw_gt_groups.items():
        for idx in range(0, len(v), 2):
            frame_range_key = "{start_frame}_{end_frame}".format(start_frame=v[idx][0], end_frame=v[idx+1][0])
            idsw_val = [v[idx][1], v[idx+1][1]]
            drawn_frames = draw_gif_all_frames(frame_range[frame_range_key], v[idx][0], frame_tracker_groups, idsw_val)
            gif_name = f"{v[idx][0]}_{v[idx][1]}to{v[idx+1][0]}_{v[idx+1][1]}.gif"
            print(f"Number of frame {len(drawn_frames)} in {gif_name}, which is {len(frame_range[frame_range_key])}")
            imageio.mimsave(os.path.join(gif_save_path, gif_name), drawn_frames, fps=2)
            
# -------- end ---------   
            
def get_idsw_frames_utils(path_to_read, tracker_filepath):
    """Utils of get_idsw_frame function"""

    cap = cv2.VideoCapture(filepath['RAW_VIDEO'])
    curr_frame = 0
    idx = 0

    frame_to_ids_boxes_raw = read_idsw_file(path_to_read)
    frame_to_ids_boxes = convert_idsw_bbox_info(frame_to_ids_boxes_raw)
    
    # group idsw grouthtruth 
    idsw_gt_groups = group_obj_idsw_gt(frame_to_ids_boxes) # {"id_gt": [[frame, id_pred], ...]}
    # construct frame range to get image frame
    frame_range = {}
    frame_range_pattern = "{start_frame}_{end_frame}"
    for k, v in idsw_gt_groups.items():
        for idx in range(0, len(v) - 1, 2):
            frame_range[frame_range_pattern.format(start_frame=v[idx][0], end_frame=v[idx+1][0])] = []            
    # group tracker frames
    frame_tracker_groups = read_tracker_file(tracker_filepath) # {"frame": [bbox1, bbox2, ...]}
    # print("frame_tracker_groups: ", frame_tracker_groups)
    # print("idsw_gt_groups: ", idsw_gt_groups)
    # print("frame_range: ", frame_range)
    
    size = len(frame_to_ids_boxes)
    counter = 0
    while True:
        ret, frame = cap.read()
        curr_frame += 1
        if not ret:
            break
        
        for key in frame_range.keys():
            if is_in_frame_range(curr_frame, key):
                # print(f"{curr_frame} frame in range {key}")
                # frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # frame_range[key].append(frame_gray)  # tran` RAM o day`
    
        if idx < size and curr_frame == list(frame_to_ids_boxes)[idx]:
            get_bounding_box(frame, frame_to_ids_boxes[curr_frame], curr_frame)
            draw_idsw_rectangle(frame, frame_to_ids_boxes[curr_frame], curr_frame)
            idx += 1
        # if(counter == 40):
        #     break
            

    # get_idsw_gif(idsw_gt_groups, frame_range,frame_tracker_groups)
    attach_images(filepath['IDSW_OUTPUT'], filepath['IDSW_ATTACH_OUTPUT'], (1280, 720))
    cap.release()

@time_recorder
def get_idsw_frame(idsw, tracker_filepath):
    """Call this function to get frames of switched ids"""

    # Change current working directory to parent dir
    if os.getcwd() != code_path:
        os.chdir(code_path)

    # Delete existed images
    delete_images(filepath['IDSW_OUTPUT'])
    delete_images(filepath['IDSW_BBOX_OUTPUT'])
    delete_images(filepath['IDSW_GIF'])
    # Create dirs
    os.makedirs(filepath['IDSW_OUTPUT'], exist_ok=True)
    os.makedirs(filepath['IDSW_ATTACH_OUTPUT'], exist_ok=True)
    os.makedirs(filepath['IDSW_BBOX_OUTPUT'], exist_ok=True)
    os.makedirs(filepath['IDSW_GIF'], exist_ok=True)


    if idsw:
        print('\nGetting ID switched frames of {}/{}...'.format(tracker_name, seq_name))
        get_idsw_frames_utils(filepath['IDSW_DETAILS'], tracker_filepath)
        print('Finished!!')
