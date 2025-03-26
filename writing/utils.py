import cv2
import numpy as np
from scipy.spatial import distance as dist
from screeninfo import get_monitors
from difflib import SequenceMatcher
import csv
from ultralytics import YOLO
import torch

#-------------Initialize variables
algorithm_name = "Yolo"
words = ["WALID"  ,  "CADY"  , "PATEK"  ,  "SHOBAKI" , "UNIVERSITY"]
metrics_headers = ["Algorithm name", "User", "Actual word", "Written word", "Precision", "Recall", "F1", "Similarity", "Total_inference_time"]
file_name = "../../results.csv"
model = YOLO('best.pt')
camera_ID = 0  # select webcam
width_keyboard = 650
height_keyboard = 350
offset_keyboard = (140, 90)  # pixel offset (x, y) of keyboard coordinates
resize_eye_frame = 5  # scaling factor for window's size
resize_frame = 0.3  # scaling factor for window's size
#-------------------------------------------------------------------------------------------------------------------
# -----   Initialize camera
def init_camera(camera_ID):
    camera = cv2.VideoCapture(0)
    return camera
# --------------------------------------------------
# ----- Make black page [3 channels]
def make_black_page(size):
    page = (np.zeros((int(size[0]), int(size[1]), 3))).astype('uint8')
    return page
# --------------------------------------------------
# ----- Make black page [3 channels]
def get_screen_size():
    for monitor in get_monitors():
        return (monitor.height, monitor.width)
# --------------------------------------------------

# ----- Make white page [3 channels]
def make_white_page(size):
    page = (np.zeros((int(size[0]), int(size[1]), 3)) + 255).astype('uint8')
    return page
# --------------------------------------------------

# -----   Rotate / flip / everything else (NB: depends on camera conf)
def adjust_frame(frame):
    # frame = cv2.rotate(frame, cv2.ROTATE_180)
    frame = cv2.flip(frame, 1)
    return frame
# --------------------------------------------------

# ----- Shut camera / windows off
def shut_off(camera):
    camera.release() # When everything done, release the capture
    cv2.destroyAllWindows()
# --------------------------------------------------
# ----- Show a window
def show_window(title_window, window):
    cv2.namedWindow(title_window)
    cv2.imshow(title_window,window)
# --------------------------------------------------
# ----- find the limits of frame-cut around the calibrated box
def find_cut_limits_(calibration_cut):
    # Convert the list of lists to a numpy array with shape (N, 4)
    calibration_cut_array = np.array(calibration_cut)
    x_cut_min = calibration_cut_array[:, 0].min()
    x_cut_max = calibration_cut_array[:, 2].max()
    y_cut_min = calibration_cut_array[:, 1].min()
    y_cut_max = calibration_cut_array[:, 3].max()
    return x_cut_min, x_cut_max, y_cut_min, y_cut_max
# --------------------------------------------------
# ----- find projection on page
def project_on_page_(src_frame, dst_frame, src_point):
    scale_x = dst_frame.shape[1] / src_frame.shape[1]
    scale_y = dst_frame.shape[0] / src_frame.shape[0]
    projected_x = int(src_point[0] * scale_x)
    projected_y = int(src_point[1] * scale_y)
    return projected_x, projected_y
# --------------------------------------------------
# -----   Calculate the center of the bounding box for both pupils
def get_pupil_center_(bbox):
    if len(bbox) == 2:
        # Assuming bbox format (center_x, center_y)
        return bbox
    x1, y1, x2, y2 = bbox
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2
    return center_x, center_y
# --------------------------------------------------
# -----   Check if the pupil center is within the keyboard bounds
def is_pupil_center_valid_(center, bounds):
    x, y = center
    x_min, x_max, y_min, y_max = bounds
    
    # Print statements for debugging
    print(f"Center coordinates: ({x}, {y})")
    print(f"Bounds: x_min={x_min}, x_max={x_max}, y_min={y_min}, y_max={y_max}")
    
    # Check if the center coordinates are within bounds
    valid_x = x_min <= x <= x_max
    valid_y = y_min <= y <= y_max
    
    # Print intermediate results for debugging
    print(f"x validity: {valid_x}, y validity: {valid_y}")
    
    # Return True if both x and y are within bounds, otherwise False
    return valid_x and valid_y
# --------------------------------------------------
# -----   display keyboard
def display_keyboard_(img, keys):
    color_board = (255, 250, 100)
    for key in keys:
        x1, x2, x3, x4 = key
        x2 = [round(float(i)) for i in x2]
        x3 = [round(float(i)) for i in x3]
        x4 = [round(float(i)) for i in x4]
        cv2.putText(img, x1, x2, cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 100), thickness=3)
        cv2.rectangle(img, x3, x4, color_board, thickness=4)
# --------------------------------------------------
#-----   get keyboard
def get_keyboard(width_keyboard, height_keyboard, offset_keyboard):
    """
    Draw a keyboard qwerty 10 x 5
    offset_keyboard = (int, int) is the spatial offset on x, y of the keyboard
    """
    column = np.arange(0, width_keyboard, width_keyboard / 10, dtype=int) + offset_keyboard[0]
    row = np.arange(0, height_keyboard, height_keyboard / 5, dtype=int) + offset_keyboard[1]

    box = int(width_keyboard / 10)
    color_board = (250, 0, 100)

    key_points = []
                    # key   center               upper-left                      bottom-right
    key_points.append(['1', (column[0], row[0]), (column[0]-box/2, row[0]-box/2), (column[0]+box/2, row[0]+box/2)])
    key_points.append(['2', (column[1], row[0]), (column[1]-box/2, row[0]-box/2), (column[1]+box/2, row[0]+box/2)])
    key_points.append(['3', (column[2], row[0]), (column[2]-box/2, row[0]-box/2), (column[2]+box/2, row[0]+box/2)])
    key_points.append(['4', (column[3], row[0]), (column[3]-box/2, row[0]-box/2), (column[3]+box/2, row[0]+box/2)])
    key_points.append(['5', (column[4], row[0]), (column[4]-box/2, row[0]-box/2), (column[4]+box/2, row[0]+box/2)])
    key_points.append(['6', (column[5], row[0]), (column[5]-box/2, row[0]-box/2), (column[5]+box/2, row[0]+box/2)])
    key_points.append(['7', (column[6], row[0]), (column[6]-box/2, row[0]-box/2), (column[6]+box/2, row[0]+box/2)])
    key_points.append(['8', (column[7], row[0]), (column[7]-box/2, row[0]-box/2), (column[7]+box/2, row[0]+box/2)])
    key_points.append(['9', (column[8], row[0]), (column[8]-box/2, row[0]-box/2), (column[8]+box/2, row[0]+box/2)])
    key_points.append(['0', (column[9], row[0]), (column[9]-box/2, row[0]-box/2), (column[9]+box/2, row[0]+box/2)])

    key_points.append(['Q', (column[0], row[1]), (column[0]-box/2, row[1]-box/2), (column[0]+box/2, row[1]+box/2)])
    key_points.append(['W', (column[1], row[1]), (column[1]-box/2, row[1]-box/2), (column[1]+box/2, row[1]+box/2)])
    key_points.append(['E', (column[2], row[1]), (column[2]-box/2, row[1]-box/2), (column[2]+box/2, row[1]+box/2)])
    key_points.append(['R', (column[3], row[1]), (column[3]-box/2, row[1]-box/2), (column[3]+box/2, row[1]+box/2)])
    key_points.append(['T', (column[4], row[1]), (column[4]-box/2, row[1]-box/2), (column[4]+box/2, row[1]+box/2)])
    key_points.append(['Y', (column[5], row[1]), (column[5]-box/2, row[1]-box/2), (column[5]+box/2, row[1]+box/2)])
    key_points.append(['U', (column[6], row[1]), (column[6]-box/2, row[1]-box/2), (column[6]+box/2, row[1]+box/2)])
    key_points.append(['I', (column[7], row[1]), (column[7]-box/2, row[1]-box/2), (column[7]+box/2, row[1]+box/2)])
    key_points.append(['O', (column[8], row[1]), (column[8]-box/2, row[1]-box/2), (column[8]+box/2, row[1]+box/2)])
    key_points.append(['P', (column[9], row[1]), (column[9]-box/2, row[1]-box/2), (column[9]+box/2, row[1]+box/2)])

    key_points.append(['A', (column[0]+ box/3, row[2]), (column[0]+ box/3-box/2, row[2]-box/2), (column[0]+ box/3+box/2, row[2]+box/2)])
    key_points.append(['S', (column[1]+ box/3, row[2]), (column[1]+ box/3-box/2, row[2]-box/2), (column[1]+ box/3+box/2, row[2]+box/2)])
    key_points.append(['D', (column[2]+ box/3, row[2]), (column[2]+ box/3-box/2, row[2]-box/2), (column[2]+ box/3+box/2, row[2]+box/2)])
    key_points.append(['F', (column[3]+ box/3, row[2]), (column[3]+ box/3-box/2, row[2]-box/2), (column[3]+ box/3+box/2, row[2]+box/2)])
    key_points.append(['G', (column[4]+ box/3, row[2]), (column[4]+ box/3-box/2, row[2]-box/2), (column[4]+ box/3+box/2, row[2]+box/2)])
    key_points.append(['H', (column[5]+ box/3, row[2]), (column[5]+ box/3-box/2, row[2]-box/2), (column[5]+ box/3+box/2, row[2]+box/2)])
    key_points.append(['J', (column[6]+ box/3, row[2]), (column[6]+ box/3-box/2, row[2]-box/2), (column[6]+ box/3+box/2, row[2]+box/2)])
    key_points.append(['K', (column[7]+ box/3, row[2]), (column[7]+ box/3-box/2, row[2]-box/2), (column[7]+ box/3+box/2, row[2]+box/2)])
    key_points.append(['L', (column[8]+ box/3, row[2]), (column[8]+ box/3-box/2, row[2]-box/2), (column[8]+ box/3+box/2, row[2]+box/2)])

    key_points.append(['Z', (column[0]+ box*2/3, row[3]), (column[0]+ box*2/3-box/2, row[3]-box/2), (column[0]+ box*2/3+box/2, row[3]+box/2)])
    key_points.append(['X', (column[1]+ box*2/3, row[3]), (column[1]+ box*2/3-box/2, row[3]-box/2), (column[1]+ box*2/3+box/2, row[3]+box/2)])
    key_points.append(['C', (column[2]+ box*2/3, row[3]), (column[2]+ box*2/3-box/2, row[3]-box/2), (column[2]+ box*2/3+box/2, row[3]+box/2)])
    key_points.append(['V', (column[3]+ box*2/3, row[3]), (column[3]+ box*2/3-box/2, row[3]-box/2), (column[3]+ box*2/3+box/2, row[3]+box/2)])
    key_points.append(['B', (column[4]+ box*2/3, row[3]), (column[4]+ box*2/3-box/2, row[3]-box/2), (column[4]+ box*2/3+box/2, row[3]+box/2)])
    key_points.append(['N', (column[5]+ box*2/3, row[3]), (column[5]+ box*2/3-box/2, row[3]-box/2), (column[5]+ box*2/3+box/2, row[3]+box/2)])
    key_points.append(['M', (column[6]+ box*2/3, row[3]), (column[6]+ box*2/3-box/2, row[3]-box/2), (column[6]+ box*2/3+box/2, row[3]+box/2)])

    key_points.append(['.', (column[8], row[3]), (column[8]-box/2, row[3]-box/2), (column[8]+box/2, row[3]+box/2)])
    key_points.append(["'", (column[9], row[3]), (column[9]-box/2, row[3]-box/2), (column[9]+box/2, row[3]+box/2)])

    key_points.append(['del', (column[0], row[4]), (column[0] - (box*2) / 3 +18, row[4] - box / 2), (column[0] + (box*2) / 3 +18, row[4] + box / 2)])    
    key_points.append([' ', (column[4], row[4]), (column[3]-box/2, row[4]-box/2), (column[6]+box/2, row[4]+box/2)])
    key_points.append(['?', (column[8], row[4]), (column[8]-box/2, row[4]-box/2), (column[8]+box/2, row[4]+box/2)])
    key_points.append(['!', (column[9], row[4]), (column[9]-box/2, row[4]-box/2), (column[9]+box/2, row[4]+box/2)])

    return key_points
# --------------------------------------------------
# -----   check key on keyboard and take input
def identify_key(key_points, coordinate_X, coordinate_Y):
    pressed_key = None

    for key in range(0, len(key_points)):
        condition_1 = np.mean(np.array([coordinate_Y, coordinate_X]) > np.array(key_points[key][2]))
        condition_2 = np.mean(np.array([coordinate_Y, coordinate_X]) < np.array(key_points[key][3]))

        if (int(condition_1 + condition_2) == 2):
            pressed_key = key_points[key][0]
            break
    return pressed_key

# --------------------------------------------------
# -----   # Function to update pupil centers from bounding box detections
def update_pupil_centers_(boxes):
    pupil_centers = []
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        pupil_centers.append((center_x, center_y))
    return pupil_centers
# --------------------------------------------------
# -----   # Function to calculate the radius
def take_radius_eye_(pupil_coordinates):
    x1, y1, x2, y2 = pupil_coordinates
    radius = ((x2 - x1) + (y2 - y1)) // 4
    return radius
# --------------------------------------------------
# -----   # Function to compute EAR
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear
# --------------------------------------------------
# --------------------------Function to detect blinking
def check_blink(frame, detector, predictor):
    # Constants
    EYE_AR_THRESH = 0.25
    # Define the landmark indices for the left and right eyes
    (lStart, lEnd) = (42, 48)
    (rStart, rEnd) = (36, 42)
    # Initialize frame counters and total blinks
    COUNTER = 0
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)

    for rect in rects:
        shape = predictor(gray, rect)
        shape = np.array([[p.x, p.y] for p in shape.parts()])

        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]

        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0

        if ear < EYE_AR_THRESH:
            return True
# --------------------------------------------------
# -----   calculate accuracy of writing
def evaluate_results(predicted, actual):
    pred_tokens = predicted.split()
    actual_tokens = actual.split()
    actual_chars = set(actual)
    predicted_chars = set(predicted)

    true_positive = len(actual_chars.intersection(predicted_chars))
    false_positive = len(predicted_chars - actual_chars)
    false_negative = len(actual_chars - predicted_chars)

    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive > 0 else 0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0



    similarity = SequenceMatcher(None, predicted, actual).ratio()

    return [precision, recall, f1, similarity]
# --------------------------------------------------
# -----   storing metrics of measurement
def write_to_csv(filename, field_names, data):
    # Check if the file exists
    file_exists = False
    try:
        with open(filename, 'r') as file:   
            file_exists = True
    except FileNotFoundError:
        file_exists = False

    # Open the CSV file in the appropriate mode
    mode = 'a' if file_exists else 'w'
    with open(filename, mode, newline='') as file:
        writer = csv.writer(file)

        # Write a new line if the file is empty
        if not file_exists:
            writer.writerow(field_names)  # Example column headers

        # Write the data to the file
        writer.writerow(data)
# --------------------------------------------------