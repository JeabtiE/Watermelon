#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <cmath>

using namespace cv;
using namespace std;

// ==============================
// ฟังก์ชัน midpoint 
// ==============================
Point2f midpoint(Point2f a, Point2f b) {
    return Point2f((a.x + b.x) * 0.5f, (a.y + b.y) * 0.5f);
}

int main() {
    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cout << "Cannot open camera" << endl;
        return -1;
    }

    while (true) {
        Mat frame;
        cap >> frame;
        if (frame.empty()) break;

        // Resize
        resize(frame, frame, Size(), 1, 1);

        // =============================
        // Gray → Blur → Threshold
        // =============================
        Mat gray, blurImg, thresh;
        cvtColor(frame, gray, COLOR_BGR2GRAY);
        GaussianBlur(gray, blurImg, Size(15, 15), 0);

        adaptiveThreshold(
            blurImg, thresh, 255,
            ADAPTIVE_THRESH_GAUSSIAN_C,
            THRESH_BINARY_INV,
            11, 2
        );

        // Morph Close
        Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
        Mat closing;
        morphologyEx(thresh, closing, MORPH_CLOSE, kernel, Point(-1, -1), 3);

        // ==========================
        // หา Contours
        // ==========================
        vector<vector<Point>> contours;
        findContours(closing, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        int hitung_objek = 0;

        for (auto &cnt : contours) {
            double area = contourArea(cnt);

            if (area < 3000) continue;

            hitung_objek++;

            // ==========================
            // คำนวณกล่องล้อม (minAreaRect)
            // ==========================
            RotatedRect rect = minAreaRect(cnt);
            Point2f box[4];
            rect.points(box);

            // วาดกรอบ
            Mat orig = frame.clone();
            for (int i = 0; i < 4; i++)
                line(orig, box[i], box[(i + 1) % 4], Scalar(0, 255, 0), 2);

            // จัดเรียงมุม (manual)
            // tl, tr, br, bl
            vector<Point2f> pts(box, box + 4);
            sort(pts.begin(), pts.end(),
                [](Point2f a, Point2f b) { return a.x < b.x; });

            Point2f left1 = pts[0], left2 = pts[1];
            Point2f right1 = pts[2], right2 = pts[3];

            Point2f tl = (left1.y < left2.y) ? left1 : left2;
            Point2f bl = (left1.y > left2.y) ? left1 : left2;
            Point2f tr = (right1.y < right2.y) ? right1 : right2;
            Point2f br = (right1.y > right2.y) ? right1 : right2;

            // อ้างอิง midpoint
            Point2f midTop = midpoint(tl, tr);
            Point2f midBottom = midpoint(bl, br);
            Point2f midLeft = midpoint(tl, bl);
            Point2f midRight = midpoint(tr, br);

            // วาดจุด midpoint
            circle(orig, midTop, 5, Scalar(0, 255, 0), -1);
            circle(orig, midBottom,5, Scalar(0, 255, 0), -1);
            circle(orig, midLeft, 5, Scalar(0, 255, 0), -1);
            circle(orig, midRight,5, Scalar(0, 255, 0), -1);

            // เส้นกากบาท
            line(orig, midTop, midBottom, Scalar(255, 0, 255), 2);
            line(orig, midLeft, midRight, Scalar(255, 0, 255), 2);

            // ============================
            // คำนวณระยะพิกเซล
            // ============================
            float width_px = norm(midTop - midBottom);
            float height_px = norm(midLeft - midRight);

            float cm_per_pixel = 1.0f / 25.5f;

            float width_cm = width_px * cm_per_pixel;
            float height_cm = height_px * cm_per_pixel;

            float weight = (0.0019f * ((2 * width_cm * height_cm * height_cm * 3.14f) / 3)) + 0.2228f;

            // =============================
            // แสดงผลบนจอ
            // =============================
            putText(orig, "W: " + to_string(width_cm) + " cm",
                    Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.7,
                    Scalar(0, 0, 255), 2);

            putText(orig, "H: " + to_string(height_cm) + " cm",
                    Point(10, 60), FONT_HERSHEY_SIMPLEX, 0.7,
                    Scalar(0, 0, 255), 2);

            putText(orig, "Weight: " + to_string(weight) + " kg",
                    Point(10, 100), FONT_HERSHEY_SIMPLEX, 0.9,
                    Scalar(255, 20, 127), 2);

            putText(orig, "Objects: " + to_string(hitung_objek),
                    Point(10, 140), FONT_HERSHEY_SIMPLEX, 0.9,
                    Scalar(20, 200, 255), 2);

            putText(orig, "*** Keep 60 cm distance ***",
                    Point(10, 450), FONT_HERSHEY_SIMPLEX, 0.7,
                    Scalar(50, 50, 50), 2);

            imshow("Camera", orig);
        }

        if (waitKey(1) == 27) break;  // ESC
    }

    cap.release();
    destroyAllWindows();
    return 0;
}