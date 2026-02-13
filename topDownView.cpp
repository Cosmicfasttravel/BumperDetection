#include "topDownView.h"
#include <cmath>
#include <iomanip>
#include <sstream>

TopDownVisualizer::TopDownVisualizer(int width, int height, double scale_meters, bool grid)
    : canvas_width(width), canvas_height(height), scale(scale_meters), show_grid(grid) {
    canvas = cv::Mat(canvas_height, canvas_width, CV_8UC3);
    camera_pos = cv::Point(canvas_width / 2, canvas_height / 2);
    grid_spacing = 1.0;  // 1 meter grid by default
}

cv::Point TopDownVisualizer::worldToCanvas(double x, double y) {
    // x (depth) maps to canvas y-axis (inverted - positive x goes up on screen)
    // y (lateral) maps to canvas x-axis (positive y goes right on screen)
    int canvas_x = camera_pos.x + static_cast<int>(y / scale);
    int canvas_y = camera_pos.y - static_cast<int>(x / scale);  // Inverted: positive depth goes up
    return cv::Point(canvas_x, canvas_y);
}

void TopDownVisualizer::clear() {
    canvas = cv::Scalar(30, 30, 30);  // Dark gray background
}

void TopDownVisualizer::drawGrid() {
    if (!show_grid) return;

    cv::Scalar grid_color(60, 60, 60);  // Lighter gray for grid
    cv::Scalar axis_color(100, 100, 100);  // Even lighter for axes

    // Calculate grid line spacing in pixels
    int grid_pixel_spacing = static_cast<int>(grid_spacing / scale);

    // Draw vertical lines (constant x in world, constant canvas_x)
    for (int x = camera_pos.x % grid_pixel_spacing; x < canvas_width; x += grid_pixel_spacing) {
        cv::Scalar color = (x == camera_pos.x) ? axis_color : grid_color;
        cv::line(canvas, cv::Point(x, 0), cv::Point(x, canvas_height), color, 1);
    }

    // Draw horizontal lines (constant y in world, constant canvas_y)
    for (int y = camera_pos.y % grid_pixel_spacing; y < canvas_height; y += grid_pixel_spacing) {
        cv::Scalar color = (y == camera_pos.y) ? axis_color : grid_color;
        cv::line(canvas, cv::Point(0, y), cv::Point(canvas_width, y), color, 1);
    }
}

void TopDownVisualizer::drawCamera() {
    // Draw camera as a triangle pointing upward (forward in world coordinates)
    int size = 20;
    std::vector<cv::Point> triangle;
    triangle.push_back(cv::Point(camera_pos.x, camera_pos.y - size));  // Top point (forward)
    triangle.push_back(cv::Point(camera_pos.x - size/2, camera_pos.y + size/2));  // Bottom left
    triangle.push_back(cv::Point(camera_pos.x + size/2, camera_pos.y + size/2));  // Bottom right

    // Fill the triangle
    cv::fillConvexPoly(canvas, triangle, cv::Scalar(255, 255, 0));  // Cyan camera

    // Draw outline
    cv::polylines(canvas, triangle, true, cv::Scalar(255, 255, 255), 2);

    // Add label
    cv::putText(canvas, "Camera",
                cv::Point(camera_pos.x - 30, camera_pos.y + size + 20),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
}

void TopDownVisualizer::drawFieldOfView(double fov_angle, double max_distance) {
    // Draw a cone representing the camera's field of view
    double half_fov = fov_angle / 2.0 * CV_PI / 180.0;

    // Calculate the cone endpoints
    cv::Point left_point = worldToCanvas(max_distance * cos(half_fov),
                                         -max_distance * sin(half_fov));
    cv::Point right_point = worldToCanvas(max_distance * cos(half_fov),
                                          max_distance * sin(half_fov));

    // Draw cone lines
    cv::line(canvas, camera_pos, left_point, cv::Scalar(100, 100, 200), 1, cv::LINE_AA);
    cv::line(canvas, camera_pos, right_point, cv::Scalar(100, 100, 200), 1, cv::LINE_AA);

    // Draw arc at the end
    cv::ellipse(canvas, camera_pos,
                cv::Size(static_cast<int>(max_distance / scale),
                         static_cast<int>(max_distance / scale)),
                -90,  // Rotation
                -fov_angle / 2.0,  // Start angle
                fov_angle / 2.0,   // End angle
                cv::Scalar(100, 100, 200), 1, cv::LINE_AA);
}

void TopDownVisualizer::drawDistanceCircles(const std::vector<double>& distances) {
    for (double dist : distances) {
        int radius = static_cast<int>(dist / scale);
        cv::circle(canvas, camera_pos, radius, cv::Scalar(80, 80, 80), 1, cv::LINE_AA);

        // Add distance label
        std::stringstream ss;
        ss << std::fixed << std::setprecision(1) << dist << "m";
        cv::putText(canvas, ss.str(),
                    cv::Point(camera_pos.x + 5, camera_pos.y - radius + 15),
                    cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(120, 120, 120), 1);
    }
}

void TopDownVisualizer::drawRobot(const RobotPosition& robot, int radius) {
    cv::Point pos = worldToCanvas(robot.x, robot.y);

    // Check if position is within canvas bounds
    if (pos.x < 0 || pos.x >= canvas_width || pos.y < 0 || pos.y >= canvas_height) {
        return;  // Skip robots outside view
    }

    // Draw robot as a filled circle
    cv::circle(canvas, pos, radius, robot.color, -1, cv::LINE_AA);

    // Draw outline
    cv::circle(canvas, pos, radius, cv::Scalar(255, 255, 255), 2, cv::LINE_AA);

    // Draw direction indicator (small line pointing forward)
    cv::Point forward_point(pos.x, pos.y - radius - 5);
    cv::line(canvas, pos, forward_point, cv::Scalar(255, 255, 255), 2, cv::LINE_AA);

    // Add label if provided
    if (!robot.label.empty()) {
        cv::putText(canvas, robot.label,
                    cv::Point(pos.x - radius, pos.y - radius - 10),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
    }

    // Add coordinate text
    std::stringstream ss;
    ss << std::fixed << std::setprecision(2)
       << "(" << robot.x << ", " << robot.y << ")";
    cv::putText(canvas, ss.str(),
                cv::Point(pos.x - radius, pos.y + radius + 15),
                cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(200, 200, 200), 1);
}

void TopDownVisualizer::drawRobots(const std::vector<RobotPosition>& robots) {
    for (const auto& robot : robots) {
        drawRobot(robot);
    }
}

void TopDownVisualizer::show(const std::string& window_name) {
    cv::imshow(window_name, canvas);
}