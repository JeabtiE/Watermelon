import cv2
import numpy as np

# ---------------------------
# สมการประมาณน้ำหนักแบบง่าย
# weight = k * area
# คุณสามารถเก็บข้อมูลจริงแล้วหาค่า k ใหม่ได้
# ---------------------------
K = 0.0035   # ค่าตัวคูณสมมติ (ปรับตามข้อมูลจริงของคุณ)

def estimate_weight(area):
    return K * area


# ---------------------------
# ฟังก์ชันประมวลผลภาพ
# ---------------------------
def process_image(path):
    # อ่านภาพ
    img = cv2.imread(path)
    if img is None:
        print("ไม่พบไฟล์ภาพ")
        return

    # Resize เล็กลงเพื่อให้คำนวณเร็วขึ้น
    img = cv2.resize(img, (600, 400))

    # แปลงเป็น Gray
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Blur ลด Noise
    blur = cv2.GaussianBlur(gray, (7, 7), 0)

    # Threshold แยกแตงโมออกจากพื้นหลัง
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # หา Contours
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        print("ไม่พบวัตถุในภาพ")
        return

    # เลือกคอนทัวร์ที่ใหญ่ที่สุด (คิดว่าเป็นแตงโม)
    c = max(contours, key=cv2.contourArea)

    area = cv2.contourArea(c)
    weight = estimate_weight(area)

    # วาดเส้น contour
    output = img.copy()
    cv2.drawContours(output, [c], -1, (0, 255, 0), 2)

    # แสดงผล
    print(f"Area ที่ตรวจพบ: {area:.2f}")
    print(f"ประมาณน้ำหนัก: {weight:.2f} กรัม")

    cv2.imshow("Original", img)
    cv2.imshow("Threshold", thresh)
    cv2.imshow("Detected Watermelon", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ---------------------------
# เรียกใช้งาน
# ---------------------------
process_image("watermelon.jpg")   # เปลี่ยนชื่อไฟล์ตามของคุณ
