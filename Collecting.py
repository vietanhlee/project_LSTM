import cv2
import csv
import os
import tkinter as tk
from tkinter import simpledialog
from tool import TOOL

# Khởi tạo TOOL
tool = TOOL()

# data_type = 1 là tục và ngược lại
def capture_data(data_type):
    # Hiển thị hộp thoại để nhập tên dữ liệu
    data_label = simpledialog.askstring(
        "Enter data", "Name of data:")
    if not data_label:
        return
    data_label = data_label.strip().replace(" ", "_")
    # Tạo đường dẫn cho tệp CSV
    if data_type == 1:
        file_name = os.path.join(
            r"DATA_TUC", f"{data_label}.csv")
    else:
        file_name = os.path.join(
            r"DATA_KHONG_TUC", f"{data_label}.csv")

    cap = cv2.VideoCapture(0)

    # Mở tệp CSV để ghi dữ liệu
    with open(file_name, "a", newline="") as file:
        writer = csv.writer(file)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)

            tool.set_input_image(frame)  # Truyền ảnh vào công cụ TOOL
            mouth_points = tool.point_output()  # Lấy ra các điểm ảnh của miệng
            writer.writerow(mouth_points)  # Ghi dữ liệu vào tệp CSV

            image_with_points = tool.pic_draw_point()  # Vẽ các điểm trên hình ảnh
            # Hiển thị hình ảnh với các điểm
            cv2.imshow("Mouth Points", image_with_points)

            # Nếu người dùng nhấn phím 'q', thoát khỏi vòng lặp
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Giải phóng tài nguyên camera và đóng cửa sổ hiển thị
    cap.release()
    cv2.destroyAllWindows()


def start_gui(data_type):
    root = tk.Tk()
    root.title("Collecting mouth data")
    root.geometry("1000x300")  # Kích thước cửa sổ

    # Nút bấm để bắt đầu thu thập dữ liệu
    btn_capture = tk.Button(
        root, text="Starting collecting data", command= lambda : capture_data(data_type))
    btn_capture.pack(pady=20)

    # Nút bấm để thoát ứng dụng
    btn_quit = tk.Button(root, text="Exit", command=root.quit)
    btn_quit.pack()

    # Chạy vòng lặp chính của giao diện
    root.mainloop()


if __name__ == "__main__":
    # Nhớ thay đổi data_type để thu thập dữ liệu cho tục (1) hoặc không tục (2)
    start_gui(1)
