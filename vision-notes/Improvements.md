# Detection Improvements
### Problem:
- ##### Yolo can output bounding boxes that are significantly wider than the reality if a false detection or a real detection overlaps
### Approaches:
 - ##### Directly after Yolo outputs bounding boxes check if they are greater than the typical aspect ratio
	 - **Implementation details**:
		 - Calculate bounding box aspect ratio
		 - Use that aspect ratio and compare it against the ideal one, and ensure that the aspect ratio has been converted from normal to the distorted because of the distortion when the images are given to the model
 - ##### Right after the Yolo output, trim down box size by using masks to fit the tightest box around the bumper
	 - **Implementation details**:
		 - Use `cv::Rect` and `cv::findNonZero()` to calculate the mask
		 - Measure the width and the height of the color and use that height to trim down the box and then use the aspect ratio to determine the width and determine where the width starts by finding the first color pixel along the edge
 - ##### Check if the amount of pixels that matches the mask is greater than a certain percent filled and if not use a tighter fitting algorithm
	 - **Implementation details**:
		 - Use `countNonZero()` on a mask to determine the amount of pixels filled and compare it to a threshold to ensure the tight box is above a certain threshold
		 - If the amount of pixels filled is not enough then tighten the bounding box further using a more strict algorithm (ex. slicing top off and counting pixels several times until the amount of white pixels is greater than the black pixels)
 - ##### Implement a number detection system to determine false positives and detect whether there are multiple robots touching
	 - **Implementation details**:
		 - Use an `mlpack` model and train it on digits to detect where they are
		 - Use that location and feed it to Yolo functioning as a 2 stage detector
 - ##### A system to fill the numbers on the bumpers to ensure measurements can be taken with maximum accuracy
	 - **Implementation details**:
		 - Once bounding boxes have been tightened use a white mask that will then merge to the blue mask, which then will have its area filled
		 - Calculate area filled to ensure that the bumper did not lose a chunk or gain a chunk
## Todo:
- [ ] **Aspect ratio filtering**
	- #### Pros:
		- Fast
		- Cheap
		- Improves reliability
		- Cleans up random noise
	- #### Cons:
		- Slower than tuning Yolo
		- Requires tightening bounding box system
	- #### Priority: VERY HIGH
- [ ] **Bounding box trimming**
	- #### Pros:
		- Increases performance
		- Reduces need for Yolo training
	- #### Cons:
		- Could cause slowdown when there is a large amount of detections
	- #### Priority: VERY HIGH
	- [ ] **Adjust old height system to the tight bounding box measurements**
	- [ ] **New Data structure for bumper measurements**
- [ ] **Filled pixels and tighten algorithm**
	 - #### Pros:
		 - Makes measurements more reliable
		 - Allows tighten boxes algorithm to iterate on itself
	- #### Cons:
		- Iteration with this system would cause slowdowns
	- #### Priority: HIGH
- [ ] **Number Detection system**
	- #### Pros:
		- Improves reliability
		- Simplifies pipeline
	- #### Cons:
		- Not needed to function
		- Could be slower than just filtering out
		- The **lazy** way of filtering
	- #### Priority: VERY LOW
- [ ] **Number filling**
	- #### Pros:
		- Fast
		- Helps other pieces of the program
	- #### Cons:
		- Could cause crash due to larger mask required
	- #### Priority: MEDIUM

### Problem:
- ##### The program feeds images that have been modified to the camera settings but the training images are using different camera settings
### Approaches: 
- ##### Implement a system to flow the images through the camera settings and bulk modify them so the model can be fed the same images the camera is feeding to it
	- **Implementation details**:
		- Separate program that takes the camera settings from the config file and applies it to the image then overwrites or saves the image into another folder and continues

### Problem:
- ##### The program currently uses the centermost pixel to determine the color of the bumper which is fragile
### Approaches:
- ##### Use masks to find the color of the bumper by finding most pixel count (See LED project (`main.cpp`) on how to do it)
	- **Implementation details**:
		- Use a mask and `countNonZero()` to determine which color is most likely and make sure they are above a certain threshold, also performing this after tightening bounding boxes is important otherwise noise can skew the result

## Todo:
- [ ] **Masks instead of centermost pixel**
	- #### Pros:
		- Fast
		- Improves reliability in edge cases
	- #### Cons:
		- Slower than center pixel
	- #### Priority: HIGH
- [ ] **Backflow image system**
	- #### Pros:
		- Improves training
		- Improves overall reliability of the program
	 - #### Cons:
		 - Doesn't help during runtime
	- #### Priority: MEDIUM

# Readability Improvements

### Problem:
- ##### The current config file is messy and long
### Approaches:
- ##### Separate runtime options from modes which are meant to be set during startup
- ##### Separate config editing viewer to change the options

## Todo:
- [ ] **Separate runtime options from modes**
	- #### Pros:
		- Cleaner
		- Reduces user error
	- #### Cons:
		- Adds another thing for the user to have to navigate
	- #### Priority: MEDIUM
- [ ] **Separate config editing tool**
	- #### Pros:
		- Easier to read than JSON
		- Incorporates the framework for a final installer
	- #### Cons:
		- Does nothing towards performance
	- #### Priority: VERY LOW

# Tuning Improvements

### Problem:
- ##### To debug with a video right now requires writing to an annotated video which is expensive during runtime
### Approaches:
- ##### Writing annotations to a text file and using the already made video capture module then take those annotations and deploy them onto the video using a separate program
	- **Implementation details**:
		- Write the exact timestamp and the annotations needed
		- Scrub the text file and on the timestamps given make the annotations (needs a custom parser or use JSON)

### Problem:
- ##### To find how long each module is taking requires long string of text and reduces readability
### Approaches:
- ##### Add a module that can time each module to find which ones use the most resources
	- **Implementation details**:
		- Module that allows you to call a `start` function and a `end` function that are thread safe and after the end is called it writes to an avg or a file
		- Or find a library that already exists to do this

## Todo:
- [ ] **Live video replay system**
	- #### Pros:
		- Faster debugging
		- Separate
	- #### Cons:
		- Lots of extra work
		- Unknown if debugging time save is worth the work
	- #### Priority: VERY LOW
- [ ] **Add module performance timing system**
	- #### Pros:
		- Improves optimization speed
		- Shows where performance is being spent
		- Helps compare changes
	- #### Cons:
		- Slight overhead
		- Extra code
	- #### Priority: LOW