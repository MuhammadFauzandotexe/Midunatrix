import cv2
import numpy as np
import imutils
import json
import time
import math
from imutils.video import VideoStream, FPS
from enum import Enum
import os

#=================================== initial config or Final Variable

CONFIG_FILE = "config.json"
image_path = "icon.png"
def nothing(x):
    pass

class Color(Enum):
    ORANGE = "ORANGE"
    CYAN = "CYAN"
    MAGENTA = "MAGENTA"
    MATCH = "MATCH"
    STAND_BY = "STAND_BY"

#============================================================ track bar ==========================================#
# Fungsi untuk membuat trackbar

def create_trackbars(window_name, color: Color):
    # Muat konfigurasi dari file
    config = load_config(CONFIG_FILE)
    
    # Pastikan warna ada dalam konfigurasi
    if color.value not in config:
        raise ValueError(f"Color '{color.value}' not found in configuration file.")
    # Ambil nilai lower dan upper dari konfigurasi
    color_lower = config[color.value]["lower"]
    color_upper = config[color.value]["upper"]
    
    # Buat trackbars
    cv2.createTrackbar("Lower H", window_name, 0, 179, nothing)
    cv2.createTrackbar("Lower S", window_name, 0, 255, nothing)
    cv2.createTrackbar("Lower V", window_name, 0, 255, nothing)
    cv2.createTrackbar("Upper H", window_name, 0, 179, nothing)
    cv2.createTrackbar("Upper S", window_name, 0, 255, nothing)
    cv2.createTrackbar("Upper V", window_name, 0, 255, nothing)
    
    # Set nilai default berdasarkan konfigurasi
    cv2.setTrackbarPos("Lower H", window_name, color_lower[0])
    cv2.setTrackbarPos("Lower S", window_name, color_lower[1])
    cv2.setTrackbarPos("Lower V", window_name, color_lower[2])
    cv2.setTrackbarPos("Upper H", window_name, color_upper[0])
    cv2.setTrackbarPos("Upper S", window_name, color_upper[1])
    cv2.setTrackbarPos("Upper V", window_name, color_upper[2])


# Fungsi untuk mendapatkan nilai dari trackbar
def get_trackbar_values(window_name):
    lower_h = cv2.getTrackbarPos("Lower H", window_name)
    lower_s = cv2.getTrackbarPos("Lower S", window_name)
    lower_v = cv2.getTrackbarPos("Lower V", window_name)
    upper_h = cv2.getTrackbarPos("Upper H", window_name)
    upper_s = cv2.getTrackbarPos("Upper S", window_name)
    upper_v = cv2.getTrackbarPos("Upper V", window_name)
    lower = [lower_h, lower_s, lower_v]
    upper = [upper_h, upper_s, upper_v]
    return lower, upper


#============================================================== File Processing
# Tambahkan warna cyan dan magenta ke dalam konfigurasi default
def load_config(file_path):
    try:
        with open(file_path, 'r') as file:
            config = json.load(file)
        return config
    except FileNotFoundError:
        print(f"[WARNING] Config file not found. Creating default config at {file_path}.")
        default_config = {
            "ORANGE": {"lower": [10, 100, 100], "upper": [25, 255, 255]},
            "CYAN": {"lower": [80, 100, 100], "upper": [100, 255, 255]},
            "MAGENTA": {"lower": [140, 100, 100], "upper": [160, 255, 255]}
        }
        with open(file_path, 'w') as file:
            json.dump(default_config, file, indent=4)
        print(f"[INFO] Configuration saved to {file_path}")
        return default_config

# Fungsi untuk menyimpan konfigurasi warna ke file
def save_config(file_path, new_lower, new_upper, color):
    with open(file_path, "r+") as file:
        data = json.load(file)
        if color.value in data:
            data[color.value]["lower"] = new_lower
            data[color.value]["upper"] = new_upper
            file.seek(0)  # Kembali ke awal file untuk menulis ulang
            json.dump(data, file, indent=4)
            file.truncate()  # Menghapus konten file setelah posisi akhir data baru
    print(f"Config for {color.value} has been updated.")

# Fungsi untuk menghitung sudut
def calculate_angle(origin, point):
    dx = point[0] - origin[0]
    dy = origin[1] - point[1]
    angle = math.degrees(math.atan2(dy, dx))
    angle = (90 - angle) % 360
    return angle

# Fungsi untuk menggambar panah
def draw_direction_arrow(frame, center_frame):
    arrow_length = 50
    arrow_thickness = 2
    arrow_color = (0, 0, 255)
    arrow_end = (center_frame[0], center_frame[1] - arrow_length)
    cv2.arrowedLine(frame, center_frame, arrow_end, arrow_color, arrow_thickness, tipLength=0.3)

# Fungsi untuk mendeteksi lingkaran oranye
def config_frame(frame, lower_orange, upper_orange):
    height, width, _ = frame.shape
    center_frame = (width // 2, height // 2)
    draw_direction_arrow(frame, center_frame)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(lower_orange), np.array(upper_orange))
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 1000:
            ((x, y), radius) = cv2.minEnclosingCircle(contour)
            center = (int(x), int(y))
            if radius > 20:
                cv2.circle(frame, center, int(radius), (0, 255, 0), 2)
                cv2.circle(frame, center, 5, (255, 0, 0), -1)
                cv2.line(frame, center_frame, center, (255, 255, 0), 2)
                angle = calculate_angle(center_frame, center)
                cv2.putText(frame, f"Angle: {angle:.2f} deg", (center[0] + 10, center[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.line(frame, (center_frame[0], 0), (center_frame[0], height), (255, 255, 255), 1)
    cv2.line(frame, (0, center_frame[1]), (width, center_frame[1]), (255, 255, 255), 1)
    return frame, mask



# Fungsi utama
def main():
    # Variabel untuk trackbar
    trackbar_active:Color = Color.STAND_BY
    trackbar_window = "Settings"

    config = load_config(CONFIG_FILE)
    orange_lower = config[Color.ORANGE.value]["lower"]
    orange_upper = config[Color.ORANGE.value]["upper"]
    vs = VideoStream(src=0).start()
    time.sleep(2.0)
    fps = FPS().start()

    while True:
        frame = vs.read()
        if frame is None:
            print("Cam error")
            break
        frame = imutils.resize(frame, width=500)

        # Tentukan warna yang sedang digunakan
        if trackbar_active == Color.ORANGE:
            lower, upper = get_trackbar_values(trackbar_window)
            frame, mask = config_frame(frame, lower, upper)
        elif trackbar_active == Color.CYAN:
            lower, upper = get_trackbar_values(trackbar_window)
            frame, mask = config_frame(frame, lower, upper)
        elif trackbar_active == Color.MAGENTA:
            lower, upper = get_trackbar_values(trackbar_window)
            frame, mask = config_frame(frame, lower, upper)
        else:
            frame, mask = config_frame(frame, orange_lower, orange_upper)
        
        # tampilan default nya 
        if trackbar_active == Color.STAND_BY:
            image = cv2.imread(image_path)  # Membaca gambar
            image = cv2.resize(image,[600,400])
            if image is not None:
                cv2.imshow("Stand By", image)  # Menampilkan gambar
            else:
                print("Image Not Found!")

        
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):  # Keluar
            break
        elif key == ord("x"):  # Buka gambar ulang
            trackbar_active = Color.STAND_BY
            cv2.destroyAllWindows()
            image = cv2.imread(image_path)  # Membaca gambar
            image = cv2.resize(image,[600,400])
            if image is not None:
                cv2.imshow("Stand By", image)  # Menampilkan gambar
            else:
                print("Image Not Found!")
        elif key in [ord("1"), ord("2"), ord("3")]:  # Beralih trackbar
            color_map = {ord("1"): Color.ORANGE, ord("2"): Color.CYAN, ord("3"): Color.MAGENTA}
            selected_color = color_map[key]
            if trackbar_active != selected_color:
                cv2.destroyAllWindows()
                cv2.namedWindow(trackbar_window)
                create_trackbars(trackbar_window, selected_color)
                trackbar_active = selected_color
                cv2.imshow(f"Frame {selected_color.value}", frame)
        elif key == ord("s") and trackbar_active not in [Color.MATCH, Color.STAND_BY]:
            new_lower_hsv, new_upper_hsv = get_trackbar_values(trackbar_window)
            save_config(CONFIG_FILE, new_lower_hsv, new_upper_hsv, trackbar_active)

        # Tampilkan frame dan mask untuk trackbar aktif
        if trackbar_active not in [Color.MATCH, Color.STAND_BY]:
            cv2.imshow(f"Frame {trackbar_active.value}", frame)        

        fps.update()
    fps.stop()
    print(f"[INFO] elapsed time: {fps.elapsed():.2f}")
    print(f"[INFO] approx. FPS: {fps.fps():.2f}")
    cv2.destroyAllWindows()
    vs.stop()

if __name__ == "__main__":
    main()