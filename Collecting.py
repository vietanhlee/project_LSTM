import cv2
import csv
import os
from tool import TOOL
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import numpy as np

tool = TOOL()

# Đọc danh sách từ xấu
bad_word = pd.read_csv(r"bad_words.csv")
bad_word = bad_word.values.flatten()
current_index = 0

# Thư mục lưu dữ liệu
DATA_DIR = r"DATA_TUC"
Is_collecting = True
saving = False

# Khởi tạo camera
cap = cv2.VideoCapture(0)
cv2.namedWindow('Mouth Points')

# Đường dẫn đến font hỗ trợ tiếng Việt
font_path = "arial.ttf"  # Thay bằng đường dẫn font của bạn nếu cần
font_size = 30
font = ImageFont.truetype(font_path, font_size)

while Is_collecting and current_index < len(bad_word):
    word = bad_word[current_index]
    index_word = [f"0{x+1}" for x in range(len(bad_word))]
    file_name = os.path.join(DATA_DIR, f'{index_word[current_index]}.csv')
    video_file = os.path.join(DATA_DIR, f'{index_word[current_index]}.avi')

    # Lấy thông số video
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = 20  # Số khung hình trên giây

    # Định dạng codec và tạo đối tượng VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(video_file, fourcc, fps, (frame_width, frame_height))

    with open(file_name, "w", newline="") as file:
        writer = csv.writer(file)
        word = bad_word[current_index]
        print(f"Đang chuẩn bị thu thập dữ liệu cho từ: {word}")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            tool.set_input_image(frame)  # Truyền ảnh vào công cụ TOOL
            mouth_points = tool.point_output()  # Lấy ra các điểm ảnh của miệng

            # Lấy hình ảnh với các điểm miệng đã được vẽ
            image_with_points = tool.pic_draw_point()

            # Chuyển image_with_points từ OpenCV (BGR) sang PIL (RGB)
            image_rgb = cv2.cvtColor(image_with_points, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)
            draw = ImageDraw.Draw(pil_image)

           # Vẽ văn bản lên ảnh chính (frame)
            draw.text((50, 10), "Nhấn S để bắt đầu thu thập dữ liệu",
                      font=font, fill=(255, 0, 0))  # Đỏ
            draw.text((50, 40), f"Nói: {bad_word[current_index]}",
                      font=font, fill=(0, 255, 0))  # Xanh lá
            draw.text((50, 70), f"Nhấn B để nếu muốn nói lại",
                      font=font, fill=(0, 255, 0))  # Xanh lá
            draw.text((50, 100), "Nhấn D để chuyển sang từ tiếp theo",
                      font=font, fill=(0, 0, 255))  # Xanh dương

            # Chuyển lại từ PIL sang OpenCV
            image_with_points = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            bounding_face = tool.draw_bounding_box()
            bounding_face = cv2.cvtColor(bounding_face, cv2.COLOR_BGR2RGB)

            # Hiển thị image_with_points với text trên cửa sổ Mouth Points
            cv2.imshow("Mouth Points", image_with_points)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('s'):
                saving = True
                print(f"Bắt đầu lưu dữ liệu cho từ: {word}")

            elif key == ord('d'):
                saving = False
                current_index += 1
                print(f"Hoàn tất lưu dữ liệu cho từ: {word}. Chuyển sang từ tiếp theo...")
                break
            elif key == ord("b"):
                if current_index >0:
                    current_index -= 1
                print("Quay lại từ trước đó")
                break
            elif key == ord('q'):  # Nhấn 'q' để thoát toàn bộ chương trình
                print("Dừng thu thập dữ liệu.")
                Is_collecting = False
                cap.release()
                out.release()
                cv2.destroyAllWindows()
                exit()

            if saving:
                if mouth_points:
                    writer.writerow(mouth_points)  # Lưu dữ liệu vào CSV
                    out.write(bounding_face)  # Ghi frame vào video (frame gốc từ webcam)

    out.release()  # Đóng file video
    
cap.release()
cv2.destroyAllWindows()