import cv2
import numpy as np
import imutils
from imutils.video import VideoStream
from imutils.video import FPS
import time
import math

# Fungsi untuk menghitung sudut dalam ruang 360 derajat (searah jarum jam dengan panah kuning sebagai 0)
def calculate_angle(origin, point):
    dx = point[0] - origin[0]
    dy = origin[1] - point[1]  # Membalik Y karena Y positif mengarah ke bawah
    angle = math.degrees(math.atan2(dy, dx))  # Sudut relatif terhadap sumbu X
    angle = (90 - angle) % 360  # Menyesuaikan sehingga 0 derajat adalah arah panah kuning
    return angle

# Fungsi untuk menggambar panah arah depan (titik nol derajat)
def draw_direction_arrow(frame, center_frame):
    arrow_length = 50  # Panjang panah
    arrow_thickness = 2  # Ketebalan panah
    arrow_color = (0, 0, 255)  # Warna panah (merah)
    
    # Titik akhir panah (arah depan/titik nol)
    arrow_end = (center_frame[0], center_frame[1] - arrow_length)
    
    # Gambar panah
    cv2.arrowedLine(
        frame, 
        center_frame, 
        arrow_end, 
        arrow_color, 
        arrow_thickness, 
        tipLength=0.3  # Panjang ujung panah
    )

# Fungsi untuk mendeteksi objek bundar berwarna oranye
def detect_orange_circle(frame):
    # Ukuran frame
    height, width, _ = frame.shape
    center_frame = (width // 2, height // 2)

    # Gambar panah arah depan
    draw_direction_arrow(frame, center_frame)

    # Konversi ke ruang warna HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Definisi rentang warna oranye
    lower_orange = np.array([10, 100, 100])  # Rentang bawah HSV warna oranye
    upper_orange = np.array([25, 255, 255])  # Rentang atas HSV warna oranye
    mask = cv2.inRange(hsv, lower_orange, upper_orange)

    # Pembersihan noise dengan erosi dan dilasi
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)

    # Temukan kontur
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        # Menghitung area kontur
        area = cv2.contourArea(contour)
        if area > 1000:  # Threshold area
            # Memperkirakan lingkaran
            ((x, y), radius) = cv2.minEnclosingCircle(contour)
            center = (int(x), int(y))

            # Validasi radius
            if radius > 20:
                # Gambarkan lingkaran dan titik tengah
                cv2.circle(frame, center, int(radius), (0, 255, 0), 2)
                cv2.circle(frame, center, 5, (255, 0, 0), -1)

                # Gambar garis dari pusat frame ke objek yang terdeteksi
                cv2.line(frame, center_frame, center, (255, 255, 0), 2)  # Warna kuning untuk garis

                # Hitung sudut terhadap tengah frame
                angle = calculate_angle(center_frame, center)
                cv2.putText(
                    frame, 
                    f"Angle: {angle:.2f} deg", 
                    (center[0] + 10, center[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2,
                )

    # Tambahkan garis pembagi ke frame
    cv2.line(frame, (center_frame[0], 0), (center_frame[0], height), (255, 255, 255), 1)
    cv2.line(frame, (0, center_frame[1]), (width, center_frame[1]), (255, 255, 255), 1)

    return frame, mask

def main():
    # Inisialisasi video streaming
    vs = VideoStream(src=0).start()
    time.sleep(2.0)
    fps = FPS().start()

    while True:
        # Baca frame dari kamera
        frame = vs.read()
        if frame is None:
            print("Gagal menangkap gambar")
            break

        # Ubah ukuran frame
        frame = imutils.resize(frame, width=500)

        # Deteksi lingkaran oranye
        processed_frame, mask = detect_orange_circle(frame)

        # Tampilkan hasil
        cv2.imshow("Frame", processed_frame)
        cv2.imshow("Mask", mask)

        # Keluar dengan menekan 'q'
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

        # Update FPS
        fps.update()

    # Stop FPS dan release resources
    fps.stop()
    print(f"[INFO] elapsed time: {fps.elapsed():.2f}")
    print(f"[INFO] approx. FPS: {fps.fps():.2f}")
    cv2.destroyAllWindows()
    vs.stop()

# Panggil fungsi utama
if __name__ == "__main__":
    main()
