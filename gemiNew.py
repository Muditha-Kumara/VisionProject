import cv2
import numpy as np
from sklearn.cluster import DBSCAN


class LaneDetector:
    def __init__(self):
        # Parameters for Hough Transform and Clustering
        self.rho = 1
        self.theta = np.pi / 180
        self.threshold = 20
        self.min_line_len = 20
        self.max_line_gap = 150

    def get_roi(self, img):
        """Applies a wide trapezoidal mask to cover all highway lanes."""
        mask = np.zeros_like(img)
        h, w = img.shape[:2]
        # Coordinates tuned for typical high-angle highway perspective
        points = np.array(
            [
                [
                    (int(w * 0.05), h),
                    (int(w * 0.95), h),
                    (int(w * 0.60), int(h * 0.35)),
                    (int(w * 0.40), int(h * 0.35)),
                ]
            ],
            np.int32,
        )
        cv2.fillPoly(mask, points, 255)
        return cv2.bitwise_and(img, mask)

    def color_mask(self, hls):
        """Isolates white and yellow lane markings."""
        # White mask (High lightness)
        lower_white = np.array([0, 190, 0])
        upper_white = np.array([180, 255, 255])
        white_mask = cv2.inRange(hls, lower_white, upper_white)

        # Yellow mask (Specific hue/saturation range)
        lower_yellow = np.array([10, 50, 80])
        upper_yellow = np.array([45, 255, 255])
        yellow_mask = cv2.inRange(hls, lower_yellow, upper_yellow)

        return cv2.bitwise_or(white_mask, yellow_mask)

    def process_image(self, frame):
        h, w = frame.shape[:2]
        hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)

        # 1. Edge Detection
        mask = self.color_mask(hls)
        edges = cv2.Canny(mask, 50, 150)
        roi_edges = self.get_roi(edges)

        # 2. Extract Line Segments
        lines = cv2.HoughLinesP(
            roi_edges,
            self.rho,
            self.theta,
            self.threshold,
            np.array([]),
            self.min_line_len,
            self.max_line_gap,
        )

        if lines is None:
            return frame, 0

        # 3. Feature Extraction for Clustering
        features = []
        metadata = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x1 == x2:
                continue

            m = (y2 - y1) / (x2 - x1)
            if abs(m) < 0.3:
                continue  # Filter horizontal noise

            b = y1 - m * x1
            x_bottom = (h - b) / m  # X-coordinate at the bottom of the image

            # Feature vector: [Normalized X-Bottom, Slope]
            features.append([x_bottom / w, m])
            metadata.append((m, b))

        if not features:
            return frame, 0

        # 4. DBSCAN Clustering
        # eps 0.05 means lines within 5% of image width are likely the same marker
        db = DBSCAN(eps=0.06, min_samples=1).fit(features)
        labels = db.labels_

        # 5. Averaging and Visualization
        output = np.copy(frame)
        unique_labels = set(labels)
        line_count = 0

        for label in unique_labels:
            if label == -1:
                continue  # Noise

            indices = np.where(labels == label)[0]
            avg_m = np.mean([metadata[i][0] for i in indices])
            avg_b = np.mean([metadata[i][1] for i in indices])

            # Extrapolate marker line
            y_start, y_end = h, int(h * 0.4)
            x_start = int((y_start - avg_b) / avg_m)
            x_end = int((y_end - avg_b) / avg_m)

            # Draw line with thickness relative to image size
            cv2.line(output, (x_start, y_start), (x_end, y_end), (255, 255, 0), 4)
            line_count += 1

        # Calculate Lanes: N markings roughly define N-1 lanes
        detected_lanes = max(0, line_count - 1)

        # Overlay Info
        overlay = output.copy()
        cv2.rectangle(overlay, (10, 10), (350, 110), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, output, 0.5, 0, output)
        cv2.putText(
            output,
            f"Markings: {line_count}",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            2,
        )
        cv2.putText(
            output,
            f"Est. Lanes: {detected_lanes}",
            (30, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        return output, detected_lanes


# Boilerplate for Execution
if __name__ == "__main__":
    detector = LaneDetector()
    image = cv2.imread("img00001.jpg")

    if image is not None:
        result, count = detector.process_image(image)
        cv2.imshow("Multi-Lane Detection Result", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Error: Could not load image.")
