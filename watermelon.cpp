#include <opencv2/opencv.hpp>
#include <iostream>
using namespace cv;
using namespace std;

// ค่าตัวคูณสมมติ เช่น weight = K * area
const double K = 0.0035;

// ฟังก์ชันประมาณน้ำหนักจากพื้นที่
double estimateWeight(double area) {
    return K * area;
}

int main() {
    // -----------------------------
    // อ่านภาพ
    // -----------------------------
    Mat img = imread("watermelon.jpg");
    if (img.empty()) {
        cout << "ไม่พบไฟล์ภาพ!" << endl;
        return -1;
    }

    // Resize ให้เล็กลง เพื่อคำนวณง่าย
    resize(img, img, Size(600, 400));

    // -----------------------------
    // Preprocessing
    // -----------------------------
    Mat gray, blurImg, thresh;
    cvtColor(img, gray, COLOR_BGR2GRAY);
    GaussianBlur(gray, blurImg, Size(7, 7), 0);

    // Threshold แบบ Otsu
    threshold(blurImg, thresh, 0, 255, THRESH_BINARY | THRESH_OTSU);

    // -----------------------------
    // หา Contours
    // -----------------------------
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;

    findContours(thresh, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    if (contours.empty()) {
        cout << "ไม่พบวัตถุในภาพ!" << endl;
        return -1;
    }

    // เลือก Contour ที่ใหญ่ที่สุด (แตงโม)
    double maxArea = 0;
    int maxIndex = 0;

    for (int i = 0; i < contours.size(); i++) {
        double area = contourArea(contours[i]);
        if (area > maxArea) {
            maxArea = area;
            maxIndex = i;
        }
    }

    // คำนวณน้ำหนัก
    double weight = estimateWeight(maxArea);

    cout << "Area ที่ตรวจพบ: " << maxArea << endl;
    cout << "ประมาณน้ำหนัก: " << weight << " กรัม" << endl;

    // -----------------------------
    // วาด contour
    // -----------------------------
    Mat output = img.clone();
    drawContours(output, contours, maxIndex, Scalar(0, 255, 0), 2);

    // -----------------------------
    // แสดงผล
    // -----------------------------
    imshow("Original", img);
    imshow("Threshold", thresh);
    imshow("Detected Watermelon", output);

    waitKey(0);
    return 0;
}