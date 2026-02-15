import cv2
import numpy as np
from scipy.spatial import distance as dist
from imutils import perspective
import imutils

# ================================
# ฟังก์ชัน midpoint (เหมือนของเดิม)
# ================================
def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

# ================================
# เปิดกล้อง
# ================================
cap = cv2.VideoCapture(0)

# ==============
# NOTE: ผมแก้ตรงนี้
# ของเดิมคุณใช้ while(cap.read()) ซึ่งผิด → ทำให้กล้องค้าง
# ==============
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # resize ลงเพื่อให้เบาเครื่อง
    frame = cv2.resize(frame, None, fx=1, fy=1)

    # =============================
    # แปลงภาพเป็น Gray → Blur → Threshold
    # =============================
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (15, 15), 0)

    # -------------------------
    # NOTE: โค้ดเดิมมี BUG ผิดสะกด THRESH_BINARY_INV
    # ผมแก้จาก THRESH_BINARY_ INV → THRESH_BINARY_INV
    # -------------------------
    thresh = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11, 2
    )

    # -------------------------
    # NOTE: แก้ชื่อ kernel (ของเดิม kernal)
    # -------------------------
    kernel = np.ones((3, 3), np.uint8)
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)

    result_img = closing.copy()

    # ============================
    # หา Contours
    # ============================
    cnts = cv2.findContours(result_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    hitung_objek = 0

    for cnt in cnts:
        area = cv2.contourArea(cnt)

        # กรองวัตถุที่เล็กเกินไป
        if area < 3000:
            continue

        hitung_objek += 1

        # ======================
        # NOTE: คุณใช้ BoxPoints แบบเก่า ผมแก้ให้เป็นเวอร์ชันใหม่
        # ======================
        box = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(box)
        box = np.array(box, dtype="int")
        box = perspective.order_points(box)

        # วาดกรอบ
        orig = frame.copy()
        cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)

        # ----------------------
        # หา midpoint
        # ----------------------
        (tl, tr, br, bl) = box
        (tltrX, tltrY) = midpoint(tl, tr)
        (blbrX, blbrY) = midpoint(bl, br)

        (tlblX, tlblY) = midpoint(tl, bl)
        (trbrX, trbrY) = midpoint(tr, br)

        # วาดจุด
        cv2.circle(orig, (int(tltrX), int(tltrY)), 5, (0, 255, 0), -1)
        cv2.circle(orig, (int(blbrX), int(blbrY)), 5, (0, 255, 0), -1)
        cv2.circle(orig, (int(tlblX), int(tlblY)), 5, (0, 255, 0), -1)
        cv2.circle(orig, (int(trbrX), int(trbrY)), 5, (0, 255, 0), -1)

        # เส้นกากบาท
        cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)), (255, 0, 255), 2)
        cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)), (255, 0, 255), 2)

        # ==========================
        # คำนวณระยะพิกเซล
        # ==========================
        lebar_pixel = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
        panjang_pixel = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

        # -------------------------
        # NOTE: คุณคำนวณผิดหลายจุด ผมแก้ดังนี้:
        # - 1 pixel = 1/25.5 cm (ต้องปรับตามกล้องจริง)
        # -------------------------
        cm_per_pixel = 1 / 25.5

        real_width = lebar_pixel * cm_per_pixel
        real_height = panjang_pixel * cm_per_pixel

        # ==========================
        # โมเดลคำนวณน้ำหนักแบบทรงรี (ตามสูตรของคุณ)
        # ==========================
        weight = (0.0019 * ((2 * real_width * (real_height ** 2) * 3.14) / 3)) + 0.2228

        # ==========================
        # แสดงข้อมูลบนจอ
        # ==========================
        cv2.putText(orig, f"L: {real_width:.2f} CM", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.putText(orig, f"H: {real_height:.2f} CM", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.putText(orig, f"Weight: {weight:.2f} kg", (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 20, 127), 2)

        cv2.putText(orig, f"Objects: {hitung_objek}", (10, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (30, 144, 255), 2)

        cv2.putText(orig, "*** Keep 60 cm distance ***",
                    (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 50, 50), 2)

        cv2.imshow("Camera", orig)

    # ปิดโปรแกรมเมื่อกด ESC
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()