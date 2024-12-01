import cv2
import numpy as np

# Fungsi untuk mendeteksi objek bundar berwarna oranye
def detect_orange_circle():
    # Aktifkan kamera
    cap = cv2.VideoCapture(0)

    while True:
        # 1. Tangkap frame dari kamera
        ret, frame = cap.read()
        if not ret:
            print("Gagal menangkap gambar")
            break

        # Ukuran frame
        height, width, _ = frame.shape
        center_frame = (width // 2, height // 2)

        # Konversi ke ruang warna HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 2. Definisi rentang warna oranye
        lower_orange = np.array([10, 100, 100])  # Rentang bawah HSV warna oranye
        upper_orange = np.array([25, 255, 255])  # Rentang atas HSV warna oranye
        mask = cv2.inRange(hsv, lower_orange, upper_orange)

        # 3. Pembersihan noise dengan erosi dan dilasi
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=2)

        # 4. Temukan kontur
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            # Menghitung area kontur
            area = cv2.contourArea(contour)
            if area > 1000:  # Threshold area (disesuaikan dengan ukuran bola)
                # Memperkirakan lingkaran
                ((x, y), radius) = cv2.minEnclosingCircle(contour)
                center = (int(x), int(y))

                # Validasi radius (hanya proses lingkaran yang besar)
                if radius > 20:  # Threshold radius
                    # Gambarkan lingkaran dan titik tengah
                    cv2.circle(frame, center, int(radius), (0, 255, 0), 2)
                    cv2.circle(frame, center, 5, (255, 0, 0), -1)

                    # Gambar garis dari tengah frame ke objek
                    cv2.line(frame, center_frame, center, (255, 255, 0), 2)

                    # Cetak koordinat dan jarak
                    # print(f"Titik tengah objek: {center}")
                    # print(f"Garis dari tengah frame ke objek: {center_frame} -> {center}")

        # 5. Tambahkan garis pembagi ke frame
        cv2.line(frame, (center_frame[0], 0), (center_frame[0], height), (255, 255, 255), 1)  # Garis vertikal
        cv2.line(frame, (0, center_frame[1]), (width, center_frame[1]), (255, 255, 255), 1)  # Garis horizontal

        # Tampilkan hasil
        cv2.imshow("Frame", frame)
        cv2.imshow("Mask", mask)

        # Keluar dengan menekan 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release kamera dan tutup semua jendela
    cap.release()
    cv2.destroyAllWindows()

# Panggil fungsi utama
if __name__ == "__main__":
    detect_orange_circle()
