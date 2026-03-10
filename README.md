# Bumper Detection

The goal of this project is to detect robot bumpers and estimate the position and team number of each robot within the camera's field of view.

The pipeline performs the following steps:

1. Detect bumpers in the image
2. Measure bumper height in pixels
3. Estimate distance from the camera
4. Compute 3D position coordinates
5. Extract team numbers from the bumpers
6. Track each robot and output its location and team number

---

# How It Works

This system uses a **YOLOv9 object detection model** to identify bumpers in each frame.

Once bumpers are detected:

1. **Contour Extraction**  
   Contours are extracted from the detected region.

2. **Bounding Rectangle Measurement**  
   The minimum-area rectangle enclosing the contour is computed, and its pixel height is measured.

3. **Distance Estimation**  
   The pixel height is compared with the known real-world bumper height using camera calibration parameters to estimate the distance from the camera.

4. **Coordinate Calculation**  
   Using the camera's field of view (FOV) and the bumper's relative position within the frame, the system computes the robot's spatial coordinates.

5. **Top-Down Projection**  
   The calculated coordinates are projected onto a top-down field map for visualization.

6. **Team Number Recognition**  
   **Tesseract OCR** is applied to the detected bumper region to read the team number.  
   Results are refined using **Levenshtein distance** to select the most likely valid team number.

7. **Tracking and Smoothing**  
   The final robot position is passed through a **Kalman filter** to reduce noise and produce smoother position estimates.

# Output

For every robot detected within the camera's field of view, the system outputs:

- Estimated position
- Recognized team number

---

# Branch Information
## **Main**
- This branch is for information purposes

## **OrangePi5**
- This branch is for development on the Orange PI 5, this branch will **NOT** work with other devices

## **GPU**
- This branch **requires NVIDIA GPU**
  
