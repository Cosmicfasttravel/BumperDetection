#ifndef TOPDOWNVIEW_H
#define TOPDOWNVIEW_H

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <vector>
#include <string>

struct RobotPosition {
    double x;  // depth (forward/backward from camera)
    double y;  // lateral (left/right from camera)
    double z;  // vertical (up/down from camera) - not used in top-down view
    std::string label;
    cv::Scalar color;

    RobotPosition(double x_val, double y_val, double z_val,
                  const std::string& lbl = "",
                  const cv::Scalar& col = cv::Scalar(0, 255, 0))
        : x(x_val), y(y_val), z(z_val), label(lbl), color(col) {}
};

class TopDownVisualizer {
private:
    int canvas_width;
    int canvas_height;
    double scale;  // meters per pixel
    cv::Mat canvas;
    cv::Point camera_pos;  // Camera position on canvas (center)

    // Grid settings
    bool show_grid;
    double grid_spacing;  // in meters

public:
    TopDownVisualizer(int width = 800, int height = 800, double scale_meters = 0.05, bool grid = true);

    // Convert world coordinates to canvas coordinates
    cv::Point worldToCanvas(double x, double y);

    // Clear the canvas
    void clear();

    // Draw the camera at center
    void drawCamera();

    // Draw grid lines
    void drawGrid();

    // Draw a single robot
    void drawRobot(const RobotPosition& robot, int radius = 15);

    // Draw multiple robots
    void drawRobots(const std::vector<RobotPosition>& robots);

    // Draw field of view cone
    void drawFieldOfView(double fov_angle = 70.0, double max_distance = 10.0);

    // Add distance circles
    void drawDistanceCircles(const std::vector<double>& distances);

    // Get the canvas
    cv::Mat getCanvas() const { return canvas.clone(); }

    // Show the visualization
    void show(const std::string& window_name = "Top-Down View");

    // Update scale
    void setScale(double new_scale) { scale = new_scale; }

    // Toggle grid
    void toggleGrid() { show_grid = !show_grid; }
};

class RobotTracker {
private:
    TopDownVisualizer visualizer;
    std::vector<RobotPosition> tracked_robots;

public:
    RobotTracker() : visualizer(800, 800, 0.02, true) {
        // 800x800 canvas, 0.02 meters per pixel (2cm per pixel)
        // This gives us a ~16m x 16m view area
    }

    void updateRobotPosition(double x, double y, double z,
                            const std::string& label,
                            const cv::Scalar& color) {
        // Find existing robot or add new one
        bool found = false;
        for (auto& robot : tracked_robots) {
            if (robot.label == label) {
                robot.x = x;
                robot.y = y;
                robot.z = z;
                found = true;
                break;
            }
        }

        if (!found) {
            tracked_robots.emplace_back(x, y, z, label, color);
        }
    }

    void clearRobots() {
        tracked_robots.clear();
    }

    void render() {
        // Clear canvas
        visualizer.clear();

        // Draw grid
        visualizer.drawGrid();

        // Draw distance circles (1m, 2m, 3m, 5m, 8m)
        visualizer.drawDistanceCircles({1.0, 2.0, 3.0, 5.0, 8.0});

        // Draw field of view
        visualizer.drawFieldOfView(70.0, 10.0);  // 70 degree FOV, 10m max

        // Draw all robots
        visualizer.drawRobots(tracked_robots);

        // Draw camera last (so it's on top)
        visualizer.drawCamera();

        // Show the visualization
        visualizer.show("Robot Positions - Top Down View");
    }

    cv::Mat getCanvas() {
        return visualizer.getCanvas();
    }
};

#endif // TOPDOWNVIEW_H