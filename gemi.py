import cv2
import numpy as np


def detect_all_lanes(image):
    # 1. Preprocessing: HLS Color Masking
    # This isolates white/yellow lines much better than simple Grayscale
    hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)

    # Focus on pixels with high lightness (White lines)
    lower_white = np.array([0, 190, 0])
    upper_white = np.array([180, 255, 255])
    mask = cv2.inRange(hls, lower_white, upper_white)

    # 2. Edge Detection on the mask
    edges = cv2.Canny(mask, 50, 150)

    # 3. Broad Region of Interest
    # We expand the ROI to cover the full width of the highway
    height, width = edges.shape
    roi_poly = np.array(
        [
            [
                (0, height),
                (width, height),
                (int(width * 0.7), int(height * 0.2)),
                (int(width * 0.3), int(height * 0.2)),
            ]
        ],
        np.int32,
    )

    roi_mask = np.zeros_like(edges)
    cv2.fillPoly(roi_mask, roi_poly, 255)
    masked_edges = cv2.bitwise_and(edges, roi_mask)

    # 4. Hough Transform (Tuned for Multiple/Dashed Lines)
    # We increase maxLineGap to bridge the segments of dashed lines
    lines = cv2.HoughLinesP(
        masked_edges,
        rho=1,
        theta=np.pi / 180,
        threshold=40,
        minLineLength=25,
        maxLineGap=200,
    )

    line_image = np.copy(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Calculate slope to filter out horizontal noise (cars/barriers)
            if x2 - x1 == 0:
                continue
            slope = (y2 - y1) / (x2 - x1)

            # Keep only steep lines typical of road markings
            if abs(slope) > 0.4:
                # Cyan color (B, G, R)
                cv2.line(line_image, (x1, y1), (x2, y2), (255, 255, 0), 3)

    return line_image


# Load and Run
raw_img = cv2.imread("road_image.jpg")
result = detect_all_lanes(raw_img)

cv2.imshow("8-Lane Detection", result)
cv2.waitKey(0)
