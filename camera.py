import cv2
import numpy as np
import imutils
import json
import time
import math
from imutils.video import VideoStream, FPS
from enum import Enum

# =================================================== Constants ==========================================================
CONFIG_FILE = "config.json"
IMAGE_PATH = "icon.png"

def nothing(x):
    pass

class Color(Enum):
    ORANGE = "ORANGE"
    CYAN = "CYAN"
    MAGENTA = "MAGENTA"
    MATCH = "MATCH"
    STAND_BY = "STAND_BY"

# =================================================== Load Configurations ======================================================
def load_config(file_path):
    try:
        with open(file_path, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        default_config = {
            "ORANGE": {"lower": [10, 100, 100], "upper": [25, 255, 255]},
            "CYAN": {"lower": [80, 100, 100], "upper": [100, 255, 255]},
            "MAGENTA": {"lower": [140, 100, 100], "upper": [160, 255, 255]},
        }
        with open(file_path, 'w') as file:
            json.dump(default_config, file, indent=4)
        return default_config

def save_config(file_path, new_lower, new_upper, color):
    with open(file_path, "r+") as file:
        data = json.load(file)
        if color.value in data:
            data[color.value]["lower"] = new_lower
            data[color.value]["upper"] = new_upper
            file.seek(0)
            json.dump(data, file, indent=4)
            file.truncate()


# ====================================================== Trackbar Functions ==================================================
def create_trackbars(window_name, color: Color):
    config = load_config(CONFIG_FILE)
    color_lower = config[color.value]["lower"]
    color_upper = config[color.value]["upper"]
    cv2.createTrackbar("Lower H", window_name, 0, 179, nothing)
    cv2.createTrackbar("Lower S", window_name, 0, 255, nothing)
    cv2.createTrackbar("Lower V", window_name, 0, 255, nothing)
    cv2.createTrackbar("Upper H", window_name, 0, 179, nothing)
    cv2.createTrackbar("Upper S", window_name, 0, 255, nothing)
    cv2.createTrackbar("Upper V", window_name, 0, 255, nothing)
    cv2.setTrackbarPos("Lower H", window_name, color_lower[0])
    cv2.setTrackbarPos("Lower S", window_name, color_lower[1])
    cv2.setTrackbarPos("Lower V", window_name, color_lower[2])
    cv2.setTrackbarPos("Upper H", window_name, color_upper[0])
    cv2.setTrackbarPos("Upper S", window_name, color_upper[1])
    cv2.setTrackbarPos("Upper V", window_name, color_upper[2])

def get_trackbar_values(window_name):
    lower_h = cv2.getTrackbarPos("Lower H", window_name)
    lower_s = cv2.getTrackbarPos("Lower S", window_name)
    lower_v = cv2.getTrackbarPos("Lower V", window_name)
    upper_h = cv2.getTrackbarPos("Upper H", window_name)
    upper_s = cv2.getTrackbarPos("Upper S", window_name)
    upper_v = cv2.getTrackbarPos("Upper V", window_name)
    return [lower_h, lower_s, lower_v], [upper_h, upper_s, upper_v]


# ====================================================== Helper Functions ======================================================
def calculate_angle(origin, point):
    dx, dy = point[0] - origin[0], origin[1] - point[1]
    angle = math.degrees(math.atan2(dy, dx))
    return (90 - angle) % 360

def draw_direction_arrow(frame, center_frame):
    arrow_length = 50
    arrow_end = (center_frame[0], center_frame[1] - arrow_length)
    cv2.arrowedLine(frame, center_frame, arrow_end, (0, 0, 255), 2, tipLength=0.3)

def config_frame(frame, lower, upper):
    height, width, _ = frame.shape
    center_frame = (width // 2, height // 2)
    draw_direction_arrow(frame, center_frame)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
    mask = cv2.erode(mask, np.ones((5, 5), np.uint8), iterations=1)
    mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) > 1000:
            ((x, y), radius) = cv2.minEnclosingCircle(contour)
            center = (int(x), int(y))
            if radius > 20:
                cv2.circle(frame, center, int(radius), (0, 255, 0), 2)
                cv2.line(frame, center_frame, center, (255, 255, 0), 2)
                angle = calculate_angle(center_frame, center)
                cv2.putText(frame, f"Angle: {angle:.2f} deg", (center[0] + 10, center[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    return frame, mask



# =================================== Main Function ===================================================
def main():
    config = load_config(CONFIG_FILE)
    vs = VideoStream(src=0).start()
    time.sleep(2.0)
    trackbar_active = Color.STAND_BY
    fps = FPS().start()

    while True:
        frame = vs.read()
        if frame is None:
            break
        frame = imutils.resize(frame, width=500)

        if trackbar_active != Color.STAND_BY:
            lower, upper = get_trackbar_values("Settings")
            frame, mask = config_frame(frame, lower, upper)
            cv2.imshow(f"Frame {trackbar_active.value}", frame)
        else:
            image = cv2.imread(IMAGE_PATH)
            if image is not None:
                cv2.imshow("Stand By", cv2.resize(image, (600, 400)))

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("1"):
            trackbar_active = Color.ORANGE
            cv2.destroyAllWindows()
            cv2.namedWindow("Settings")
            create_trackbars("Settings", trackbar_active)
        elif key == ord("s") and trackbar_active != Color.STAND_BY:
            new_lower, new_upper = get_trackbar_values("Settings")
            save_config(CONFIG_FILE, new_lower, new_upper, trackbar_active)

        fps.update()

    fps.stop()
    vs.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
