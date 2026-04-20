import cv2
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.linear_model import LinearRegression


def improve_lane_detection(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return "File not found"
    h, w = img.shape[:2]

    # 1. High-Fidelity Masking
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    # Target white and yellow precisely
    white_mask = cv2.inRange(hls, np.array([0, 190, 0]), np.array([180, 255, 255]))
    yellow_mask = cv2.inRange(hls, np.array([15, 40, 100]), np.array([35, 255, 255]))
    mask = cv2.bitwise_or(white_mask, yellow_mask)

    # 2. Edge Extraction & ROI
    edges = cv2.Canny(mask, 50, 150)
    roi_poly = np.array(
        [[(0, h), (w, h), (int(w * 0.7), int(h * 0.3)), (int(w * 0.3), int(h * 0.3))]],
        np.int32,
    )
    roi_mask = np.zeros_like(edges)
    cv2.fillPoly(roi_mask, roi_poly, 255)
    masked = cv2.bitwise_and(edges, roi_mask)

    # 3. Hough Lines with Aggressive Gap Filling
    lines = cv2.HoughLinesP(
        masked, 1, np.pi / 180, 30, minLineLength=30, maxLineGap=200
    )
    if lines is None:
        return img

    # 4. Feature Preparation for Clustering
    points_per_cluster = {}
    features = []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x1 == x2:
            continue
        m = (y2 - y1) / (x2 - x1)
        if abs(m) < 0.35:
            continue  # Reject horizontal noise (vehicles/shadows)

        b = y1 - m * x1
        x_bottom = (h - b) / m  # Project to the bottom of the screen
        theta = np.arctan(m)

        # Normalize features for DBSCAN
        features.append([x_bottom / w, theta])
        points_per_cluster[len(features) - 1] = [(x1, y1), (x2, y2)]

    if not features:
        return img

    # 5. Clustering segments into unique lane markers
    # eps is the 'proximity' threshold. Adjust to group lines closer or further apart.
    db = DBSCAN(eps=0.05, min_samples=1).fit(features)
    labels = db.labels_

    # 6. Master Line Generation (Linear Regression)
    result_img = np.copy(img)
    unique_markers = set(labels)

    for cluster_id in unique_markers:
        if cluster_id == -1:
            continue

        # Collect all points belonging to this marker
        indices = np.where(labels == cluster_id)[0]
        all_pts = []
        for idx in indices:
            all_pts.extend(points_per_cluster[idx])

        # Fit a single line to all points in the cluster
        all_pts = np.array(all_pts)
        X = all_pts[:, 1].reshape(
            -1, 1
        )  # Y values (dependent variable for vertical lines)
        y = all_pts[:, 0]  # X values

        model = LinearRegression().fit(X, y)

        # Draw the extended line from the ROI bottom to the horizon
        y_range = np.array([h, int(h * 0.35)]).reshape(-1, 1)
        x_preds = model.predict(y_range)

        pt1 = (int(x_preds[0]), int(y_range[0]))
        pt2 = (int(x_preds[1]), int(y_range[1]))

        # Draw the consolidated master line
        cv2.line(result_img, pt1, pt2, (255, 255, 0), 4)

    return result_img


result = improve_lane_detection('raw.jpg')
if isinstance(result, str):
    print(result)
else:
    cv2.imshow("Lane Detection Result", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
