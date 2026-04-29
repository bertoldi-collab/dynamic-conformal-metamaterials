import cv2
import numpy as np
from scripts.tracking.utils import find_markers


# Load the video file
cap = cv2.VideoCapture('/path/to/video.mp4')

# Read the first frame
ret, frame = cap.read()

# Convert the frame to grayscale
gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Threshold the frame to get binary image
thresh = cv2.adaptiveThreshold(gray_frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 1001, 45)

# # Show the binary image and wait for key press
# cv2.imshow('Binary Image', thresh)
# cv2.waitKey(0)

# Find contours in the binary image
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

# Show the contours and wait for key press
cv2.drawContours(frame, contours, -1, (0, 0, 255), 2)
cv2.imshow('Contours', frame)
cv2.waitKey(0)

# Collect marker positions from the user by clicking on the image
marker_positions = []


def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        marker_positions.append((x, y))


cv2.namedWindow('Select Markers')
cv2.setMouseCallback('Select Markers', mouse_callback)
while True:
    cv2.imshow('Select Markers', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()

# # Compute the marker positions based on the contours
# marker_positions = []
# for contour in contours:
#     # Compute the area of the contour
#     area = cv2.contourArea(contour)
#     if area < 100 or area > 3000:
#         continue
#     # # Pick four points on the contour
#     # for i in range(0, len(contour), len(contour)//4):
#     #     marker_positions.append(tuple(contour[i][0]))
#     # Use the four corners of minAreaRect to get the position of the marker
#     rect = cv2.minAreaRect(contour)
#     (x, y), (w, h), theta = rect
#     box = cv2.boxPoints(((x, y), (0.8*w, 0.8*h), theta))
#     box = np.int0(box)
#     for point in box:
#         marker_positions.append(tuple(point))

# Draw the markers on the frame
for position in marker_positions:
    cv2.circle(frame, position, 2, (0, 255, 0), -1)

# Show the frame and wait for key press
cv2.imshow('Frame', frame)
cv2.waitKey(0)

# Define the size of the marker template
marker_template_size = 18
search_window_size = 1.5*marker_template_size
scaling_factor = 5
template_update_rate = 0
search_window_update_rate = 1

# Initialize the positions of the markers in the first frame
marker_positions = np.array(marker_positions, dtype=np.float32)
template_markers = marker_positions.copy()
search_markers = marker_positions.copy()
marker_positions_current = marker_positions.copy()

template_frame = gray_frame.copy()

# Loop over the rest of the frames
while True:
    # Read the next frame and convert to grayscale
    ret, frame = cap.read()
    if not ret:
        break
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Print frame number
    print("Frame #", cap.get(cv2.CAP_PROP_POS_FRAMES))

    marker_positions_current = find_markers(
        template_frame, gray_frame, template_markers, search_markers, search_window_size=search_window_size, marker_template_size=marker_template_size, upscaling_factor=scaling_factor)

    # Update the template frame
    if template_update_rate != 0 and cap.get(cv2.CAP_PROP_POS_FRAMES) % template_update_rate == 0:
        template_frame = gray_frame.copy()
        template_markers = marker_positions_current.copy()

    # Update the search window
    if search_window_update_rate != 0 and cap.get(cv2.CAP_PROP_POS_FRAMES) % search_window_update_rate == 0:
        search_markers = marker_positions_current.copy()

    # Draw the markers on the current frame
    for position in marker_positions_current:
        position_int = int(position[0]), int(position[1])
        cv2.circle(frame, position_int, 2, (0, 255, 0), -1)

    # Display the current frame
    cv2.imshow("frame", frame)
    # cv2.waitKey(0)

    # Exit if the user presses 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close the window
cap.release()
cv2.destroyAllWindows()
