import time
import cv2
import numpy as np
from utils import *
import datetime
import dlib

# ------------------------------------------------------------------- INITIALIZATION
# Load the face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
# Initialize the camera
camera = init_camera(camera_ID=camera_ID)
# Get the screen size
size_screen = get_screen_size()
# make a page (2D frame) to write & project
keyboard_page = make_black_page(size=(size_screen[0]//2,size_screen[1]))
calibration_page = make_black_page(size=size_screen)
# Initialize keyboard
key_points = get_keyboard(width_keyboard=width_keyboard, height_keyboard=height_keyboard, offset_keyboard=offset_keyboard)
# Initialize corner index
corner = 0
pressed_key = True
# key_on_screen = " "
string_to_write = "text: "
# ------------------------------------------------------------------- CALIBRATION
corners = [(offset_keyboard),(width_keyboard + offset_keyboard[0], height_keyboard + offset_keyboard[1]),
           (width_keyboard + offset_keyboard[0], offset_keyboard[1]), (offset_keyboard[0], height_keyboard + offset_keyboard[1])]
calibration_cut_right = []
calibration_cut_left = []
# ----------------------------------------------------------Initialize lists for storing pupil centers and their history
pupil_centers = []
pupil_history = []
# Initialize variables for tracking pupil movement
historical_pupil_positions = []
# Main loop for calibration and pupil detection
while True:
    if len(calibration_cut_right) >= 4:# and len(calibration_cut2) >= 4:  # Calibration of 4 corners
        calibration_cut_right = calibration_cut_right[:4]
        break
    ret, frame = camera.read()  # Capture frame
    frame = adjust_frame(frame)  # Rotate / flip
    # Messages for calibration
    cv2.putText(calibration_page, 'Move your face left and right focusing on circle', tuple((np.array(size_screen) / 7).astype('int')), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
    cv2.circle(calibration_page, corners[corner], 40, (0, 255, 0), -1)
    # Run YOLO model for detection
    results = model(frame)
    boxes = results[0].boxes.xyxy  # Bounding boxes
    labels = results[0].boxes.cls  # Class labels
    confidences = results[0].boxes.conf  # Confidence scores
    names = results[0].names  # Class names dictionary
    pupil_boxes = []
    # Iterate through detected objects
    for box, label, confidence in zip(boxes, labels, confidences):
        # Detecting two pupils bound box at once
        if label == 1:  # Adjust confidence threshold and label as needed
            pupil_boxes.append(box)
    # Update pupil centers if pupil boxes are detected
    if pupil_boxes:
        pupil_centers = update_pupil_centers_(pupil_boxes)
        # Check if pupil coordinates are detected
        if pupil_centers:
            for pupil_box in pupil_boxes:
                x_min, y_min, x_max, y_max = map(int, pupil_box)
                width = x_max - x_min
                half_width = width // 2
                right_eye_box = (x_min + half_width, y_min, x_max, y_max)
                left_eye_box = (x_min , y_min, x_max - half_width, y_max)
                calibration_cut_right.append(right_eye_box)
                calibration_cut_left.append(left_eye_box)
            # Draw "OK" text at the current corner's position
            cv2.putText(calibration_page, 'OK', tuple(np.array(corners[corner]) - 5), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 5)
            # Move to the next corner for calibration
            if corner < 3:
                corner += 1
    # Display the calibration page
    show_window('projection', calibration_page)
    # Display the frame for visual feedback (optional)
    show_window('frame', cv2.resize(frame, (480, 640)))
    # Exit condition (for debugging)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()
# Ensure the list of pupil coordinates is consistent
if len(calibration_cut_right) >= 4:# and len(calibration_cut2) >= 4:
    calibration_cut_right = calibration_cut_right[:4]
    x_min_r, x_max_r, y_min_r, y_max_r = find_cut_limits_(calibration_cut_right)
    x_min_l, x_max_l, y_min_l, y_max_l = find_cut_limits_(calibration_cut_left)
    # Messages for user
    cv2.putText(calibration_page, 'calibration done. please wait for the keyboard...', tuple((np.array(size_screen) / 5).astype('int')), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3)
    show_window('projection', calibration_page)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    string_to_write = ""
    written_words = []
    times_taken = []
    for i in range(len(words)):
        start_time = datetime.datetime.now()
        end_time = start_time + datetime.timedelta(minutes=1)
        input_start_time = datetime.datetime.now()
        while datetime.datetime.now() < end_time:
            ret, frame = camera.read()  # Capture frame
            frame = adjust_frame(frame)  # Rotate / flip
            text = words[i]
            height, width = frame.shape[:2]
    
            # Get the size of the text
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 2, 5)[0]
            
            # Calculate the bottom-left corner of the text
            text_x = width - text_size[0] - 10  # 10 pixels from the right edge
            text_y = text_size[1] + 10          # 10 pixels from the top edge
            
            # Put the text on the frame
            cv2.putText(frame, words[i], (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
            cut_frame1 = np.copy(frame[y_min_r-60:y_min_r+80, x_min_r+5:x_min_r+110, :])
            cut_frame2 = np.copy(frame)#[y_min-60:y_min+80, x_min-50:x_min+40, :])
            # Draw keyboard
            keyboard_page = make_black_page(size=(size_screen[0]//2,size_screen[1]//2))
            display_keyboard_(img=keyboard_page, keys=key_points)
            text_page = make_white_page(size=(200, 800))
            text_page = cv2.resize(text_page, (800, 200))
            # Pass the right frame to the model for projection on the keyboard
            results = model(cut_frame1)  # Detect right pupil
            boxes = results[0].boxes.xyxy  # Bounding boxes
            labels = results[0].boxes.cls  # Class labels
            confidences = results[0].boxes.conf  # Confidence scores
            names = results[0].names  # Class names dictionary
            pupil_coordinates = []
            # Iterate through each detected object in the right frame
            for box, label in zip(boxes, labels):
                x1, y1, x2, y2 = map(int, box)  # Convert box coordinates to integers
                label = int(label)  # Convert tensor to integer
                if label == 0:  # Pupil
                    pupil_coordinates.append((x1, y1, x2, y2))
            if pupil_coordinates:
                px1, py1, px2, py2 = pupil_coordinates[0]
                margin = 1  # Increase this value to make the bounding box larger
                # Expand the bounding box by the margin
                px1 = max(0, px1 - margin)
                py1 = max(0, py1 - margin)
                px2 = min(cut_frame1.shape[1], px2 + margin)
                py2 = min(cut_frame1.shape[0], py2 + margin)
                pupil_center = ((px1 + px2) // 2, (py1 + py2) // 2)  # Get pupil center
                pupil_on_cut = np.array([pupil_center[0], pupil_center[1]])
                # Draw pupil on cut frame
                cv2.rectangle(cut_frame1, (px1, py1), (px2, py2), (255, 0, 0), 2)
                # Calculate the scaling factors for projection
                scaling_factor_x = keyboard_page.shape[1] / cut_frame1.shape[1]
                scaling_factor_y = keyboard_page.shape[0] / cut_frame1.shape[0]
                # Project pupil onto keyboard page
                pupil_on_keyboard = (int(pupil_on_cut[0] * scaling_factor_x), int(pupil_on_cut[1] * scaling_factor_y))
                # Validate pupil position on keyboard
                x_min, x_max = 0, keyboard_page.shape[1] - 1
                y_min, y_max = 0, keyboard_page.shape[0] - 1
                valid = is_pupil_center_valid_(pupil_on_keyboard, (x_min, x_max, y_min, y_max))
                if valid:
                    # Draw circle at pupil_on_keyboard on the keyboard
                    cv2.circle(keyboard_page, (pupil_on_keyboard[0], pupil_on_keyboard[1]), 40, (0, 255, 0), 3)
                    blink = check_blink(cut_frame2, detector, predictor)
                    if blink:
                        blink = False
                        pressed_key = identify_key(key_points=key_points, coordinate_X=pupil_on_keyboard[1], coordinate_Y=pupil_on_keyboard[0])
                        if pressed_key and isinstance(pressed_key, str):  # Ensure pressed_key is not None and is a string
                            if pressed_key == 'del':
                                string_to_write = string_to_write[:-1]
                            else:
                                string_to_write += pressed_key
                            if string_to_write in words:
                                break
                            time.sleep(0.3)  # To avoid is_blinking=True in the next frame
            # Display text and keyboards
            if len(string_to_write.split()) > 10:
                txt_written = string_to_write + "\n"
            else:
                txt_written = string_to_write
            cv2.putText(text_page, txt_written, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 5)
            show_window('projection', keyboard_page)
            show_window('cut_frame1', cut_frame1)
            show_window('cut_frame2', cut_frame2)
            show_window('text_page', text_page)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        input_end_time = datetime.datetime.now()
        time_taken = (input_end_time - input_start_time).total_seconds()
        
        written_words.append(string_to_write)
        times_taken.append(time_taken)
        string_to_write = ""
    data = []
    user = "user3"
    for W_A, W_R, t in zip(words, written_words, times_taken):
        data = [algorithm_name, user, W_A, W_R]
        data += evaluate_results(W_R, W_A)
        data += [t]
        write_to_csv(file_name,metrics_headers,data)
# Shutdown camera and windows
shut_off(camera)
cv2.destroyAllWindows()