import cv2
import numpy as np


def _build_lane_candidate_mask(img):
    h, w = img.shape[:2]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_eq = clahe.apply(gray)
    sobel_x = np.absolute(cv2.Sobel(gray_eq, cv2.CV_64F, 1, 0, ksize=3))
    sobel_x = np.uint8(255 * sobel_x / (sobel_x.max() + 1e-6))
    _, grad_mask = cv2.threshold(sobel_x, 38, 255, cv2.THRESH_BINARY)

    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    white_hls = cv2.inRange(hls, np.array([0, 130, 0]), np.array([180, 255, 120]))

    mask = cv2.bitwise_or(grad_mask, white_hls)
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)),
        iterations=1,
    )
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (2, 7)),
        iterations=1,
    )

    roi = np.zeros_like(mask)
    polygon = np.array(
        [
            [
                (int(0.05 * w), h),
                (int(0.05 * w), int(0.08 * h)),
                (int(0.95 * w), int(0.08 * h)),
                (int(0.95 * w), h),
            ]
        ],
        dtype=np.int32,
    )
    cv2.fillPoly(roi, polygon, 255)
    return cv2.bitwise_and(mask, roi)


def _extract_vertical_segments(mask):
    lines = cv2.HoughLinesP(
        mask,
        rho=1,
        theta=np.pi / 180,
        threshold=12,
        minLineLength=8,
        maxLineGap=35,
    )

    segments = []
    if lines is None:
        return segments

    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        if 35 <= angle <= 145:
            length = float(np.hypot(x2 - x1, y2 - y1))
            segments.append((x1, y1, x2, y2, length))

    return segments


def _cluster_segments_by_x(segments, y_ref, cluster_gap=38):
    candidates = []
    for x1, y1, x2, y2, length in segments:
        if y2 == y1:
            continue
        m = (x2 - x1) / (y2 - y1)
        b = x1 - m * y1
        x_ref = m * y_ref + b
        candidates.append((x_ref, x1, y1, x2, y2, length))

    if not candidates:
        return []

    candidates.sort(key=lambda item: item[0])
    clusters = [[candidates[0]]]

    for item in candidates[1:]:
        if abs(item[0] - clusters[-1][-1][0]) <= cluster_gap:
            clusters[-1].append(item)
        else:
            clusters.append([item])

    return clusters


def _fit_cluster_line(cluster, y_bottom, y_top):
    seg_slopes = []
    for _, x1, y1, x2, y2, _ in cluster:
        dy = y2 - y1
        if dy == 0:
            continue
        seg_slopes.append((x2 - x1) / dy)

    if not seg_slopes:
        return None

    median_slope = float(np.median(seg_slopes))
    slope_window = 0.35

    x_vals = []
    y_vals = []
    weights = []

    for _, x1, y1, x2, y2, length in cluster:
        dy = y2 - y1
        if dy == 0:
            continue
        seg_slope = (x2 - x1) / dy
        if abs(seg_slope - median_slope) > slope_window:
            continue

        x_vals.extend([x1, x2])
        y_vals.extend([y1, y2])
        weights.extend([length, length])

    if len(x_vals) < 4:
        return None

    fit = np.polyfit(y_vals, x_vals, deg=1, w=weights)
    m, b = fit

    if abs(m) > 0.65:
        return None

    # Avoid large extrapolation: draw only where cluster actually has support.
    supported_top = max(y_top, int(np.percentile(y_vals, 15)))
    x_bottom = int(m * y_bottom + b)
    x_top = int(m * supported_top + b)

    return (x_bottom, int(y_bottom), x_top, int(supported_top))


def _classify_cluster(cluster, y_bottom, y_top):
    intervals = []
    for _, _, y1, _, y2, _ in cluster:
        intervals.append((min(y1, y2), max(y1, y2)))

    intervals.sort(key=lambda item: item[0])
    merged = []
    for start, end in intervals:
        if not merged or start > merged[-1][1] + 12:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)

    covered = sum(end - start for start, end in merged)
    span = max(1, y_bottom - y_top)
    coverage_ratio = covered / span
    max_gap = 0
    for idx in range(1, len(merged)):
        max_gap = max(max_gap, merged[idx][0] - merged[idx - 1][1])

    if coverage_ratio >= 0.58 and max_gap < 28:
        return "continuous"
    return "dashed"


def _draw_dashed_line(canvas, pt1, pt2, color, thickness=4, dash_len=18, gap_len=12):
    x1, y1 = pt1
    x2, y2 = pt2
    total_len = float(np.hypot(x2 - x1, y2 - y1))
    if total_len < 1:
        return

    dx = (x2 - x1) / total_len
    dy = (y2 - y1) / total_len

    dist = 0.0
    while dist < total_len:
        start = dist
        end = min(dist + dash_len, total_len)
        sx = int(x1 + dx * start)
        sy = int(y1 + dy * start)
        ex = int(x1 + dx * end)
        ey = int(y1 + dy * end)
        cv2.line(canvas, (sx, sy), (ex, ey), color, thickness)
        dist += dash_len + gap_len


def _clusters_to_detections(clusters, y_bottom, y_top, width):
    detections = []
    for cluster in clusters:
        total_length = sum(item[-1] for item in cluster)
        if len(cluster) < 2 or total_length < 60:
            continue

        line = _fit_cluster_line(cluster, y_bottom, y_top)
        if line is None:
            continue

        x_bottom = line[0]
        if x_bottom < int(0.06 * width) or x_bottom > int(0.94 * width):
            continue

        line_type = _classify_cluster(cluster, y_bottom, y_top)
        detections.append((x_bottom, line, line_type, total_length))

    detections.sort(key=lambda item: item[0])
    return detections


def _suppress_close_detections(detections, distance=18, max_count=12):
    if not detections:
        return []

    filtered = [detections[0]]
    for item in detections[1:]:
        prev = filtered[-1]
        if abs(item[0] - prev[0]) < distance:
            if item[3] > prev[3]:
                filtered[-1] = item
        else:
            filtered.append(item)

    if len(filtered) > max_count:
        filtered = sorted(filtered, key=lambda item: item[3], reverse=True)[:max_count]
        filtered.sort(key=lambda item: item[0])

    return filtered


def _remove_crossing_detections(detections):
    filtered = detections[:]
    changed = True

    while changed and len(filtered) > 1:
        changed = False
        filtered.sort(key=lambda item: item[0])

        for idx in range(len(filtered) - 1):
            left = filtered[idx]
            right = filtered[idx + 1]

            left_top_x = left[1][2]
            right_top_x = right[1][2]
            # Small inversions can appear from perspective/noise; only suppress clear crossings.
            if left_top_x <= right_top_x + 25:
                continue

            if (left_top_x - right_top_x) < 80:
                continue

            # Keep the stronger supported line.
            if left[3] >= right[3]:
                del filtered[idx + 1]
            else:
                del filtered[idx]
            changed = True
            break

    filtered.sort(key=lambda item: item[0])
    return filtered


def _detect_lane_markings(img):
    h, w = img.shape[:2]
    mask = _build_lane_candidate_mask(img)
    segments = _extract_vertical_segments(mask)

    y_bottom = int(h * 0.95)
    y_top = int(h * 0.32)
    y_ref = int(h * 0.78)
    clusters = _cluster_segments_by_x(segments, y_ref=y_ref, cluster_gap=14)

    raw_detections = _clusters_to_detections(clusters, y_bottom, y_top, w)
    filtered = _suppress_close_detections(raw_detections, distance=18, max_count=12)
    filtered = _remove_crossing_detections(filtered)

    detections = [(x_bottom, line, line_type) for x_bottom, line, line_type, _ in filtered]
    detections.sort(key=lambda item: item[0])
    return detections


def generate_drive_visualization(image_path, output_path="lane_output_improved.jpg"):
    img = cv2.imread(image_path)
    if img is None:
        return None

    detections = _detect_lane_markings(img)
    overlay = np.zeros_like(img)

    for _, line, line_type in detections:
        x1, y1, x2, y2 = line
        if line_type == "continuous":
            cv2.line(overlay, (x1, y1), (x2, y2), (0, 255, 0), 5)
        else:
            _draw_dashed_line(overlay, (x1, y1), (x2, y2), (0, 255, 255), thickness=5)

    glow = cv2.GaussianBlur(overlay, (11, 11), 0)
    final = cv2.addWeighted(img, 1.0, overlay, 0.70, 0)
    final = cv2.addWeighted(final, 1.0, glow, 0.45, 0)

    cv2.rectangle(final, (10, 10), (420, 82), (0, 0, 0), -1)
    cv2.putText(
        final,
        f"Detected parallel lanes: {len(detections)}",
        (20, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        final,
        "Green=continuous  Yellow=dashed",
        (20, 72),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
    )

    cv2.imwrite(output_path, final)
    cv2.imshow("Improved Lane Detection", final)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return final


if __name__ == "__main__":
    generate_drive_visualization("img00001.jpg")
