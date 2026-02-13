# Bumper Detection
The goal of this project is to
Detect bumpers -> Find height -> get distance -> get other cordinates -> get numbers -> and finally have the location and different team numbers of each robot inside our FOV

# How it works
This workflow functions with a YoloV9 model that detects the bumpers it then finds the contours and gets the height of the smallest rectangle that can fit the contour, after that it compares it to the pixel height and the real height which allows us to get the distance away from the camera, after that we can get the other 2 cordinates by comparing the FOV of the camera with the percent of FOV taken up, with the cordinates, I plot them onto a top down view and use Tesseract OCR to find the numbers on the bumpers which are then filtered down with a Levenshtein distance to find the most likely, after that it is assigned into a Kalman filter to smooth out the distance
