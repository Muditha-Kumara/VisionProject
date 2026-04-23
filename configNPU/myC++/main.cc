#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <vector>
#include <stdint.h>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <memory>
#include <set>
#include <dirent.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <errno.h>
#include <sys/time.h>
#include <rknn_api.h>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

// ==================== CONFIGURATION ====================
const char* MODEL_PATH   = "yolov8n.rknn";
const char* POLYGON_CSV  = "lane_polygons.csv";
const char* IMAGE_DIR    = "dataset";
const char* OUTPUT_CSV   = "luckfox_results.csv";
const char* OUTPUT_IMAGE_DIR = "output_images";
const float OBJ_THRESH   = 0.25f;
const float NMS_THRESH   = 0.45f;
const int   MODEL_W      = 640;
const int   MODEL_H      = 640;
const int   YOLO_CLASS_NUM = 80;
const int   YOLO_PROP_BOX_SIZE = 5 + YOLO_CLASS_NUM;

const int YOLO_ANCHORS[3][6] = {
    {10, 13, 16, 30, 33, 23},
    {30, 61, 62, 45, 59, 119},
    {116, 90, 156, 198, 373, 326}
};

// COCO vehicle classes we care about: car=2, motorcycle=3, bus=5, truck=7
static const int VEHICLE_CLASSES[] = {2, 3, 5, 7};

// ==================== DATA STRUCTURES ====================
struct Lane {
    string name;
    vector<Point> points;
};

struct Detection {
    float cx, cy, w, h; // center x/y, width, height (in model coords)
    float conf;
    int   class_id;
};

bool ensure_dir(const string& dir_path) {
    struct stat st;
    if (stat(dir_path.c_str(), &st) == 0 && S_ISDIR(st.st_mode)) {
        return true;
    }
    if (mkdir(dir_path.c_str(), 0755) == 0 || errno == EEXIST) {
        return true;
    }
    return false;
}

string class_name(int class_id) {
    switch (class_id) {
        case 2: return "car";
        case 3: return "motorcycle";
        case 5: return "bus";
        case 7: return "truck";
        default: return "vehicle";
    }
}

string with_jpg_ext(const string& filename) {
    size_t dot_pos = filename.find_last_of('.');
    if (dot_pos == string::npos) return filename + ".jpg";
    return filename.substr(0, dot_pos) + ".jpg";
}

bool save_ppm(const string& out_path, const Mat& src_bgr) {
    if (src_bgr.empty()) return false;
    Mat bgr;
    if (src_bgr.type() == CV_8UC3) {
        bgr = src_bgr;
    } else if (src_bgr.channels() == 1) {
        cvtColor(src_bgr, bgr, COLOR_GRAY2BGR);
    } else {
        src_bgr.convertTo(bgr, CV_8UC3);
    }

    FILE* fp = fopen(out_path.c_str(), "wb");
    if (!fp) return false;
    fprintf(fp, "P6\n%d %d\n255\n", bgr.cols, bgr.rows);
    for (int y = 0; y < bgr.rows; ++y) {
        const Vec3b* row = bgr.ptr<Vec3b>(y);
        for (int x = 0; x < bgr.cols; ++x) {
            fputc(row[x][2], fp);
            fputc(row[x][1], fp);
            fputc(row[x][0], fp);
        }
    }
    fclose(fp);
    return true;
}

bool save_annotated_image(const string& out_path_jpg, const Mat& src_bgr) {
    if (imwrite(out_path_jpg, src_bgr)) {
        return true;
    }
    size_t dot_pos = out_path_jpg.find_last_of('.');
    string ppm_path = (dot_pos == string::npos) ? (out_path_jpg + ".ppm") : (out_path_jpg.substr(0, dot_pos) + ".ppm");
    return save_ppm(ppm_path, src_bgr);
}

// ==================== CSV LOADER ====================
vector<Lane> load_polygons(const char* path) {
    vector<Lane> lanes;
    ifstream file(path);
    string line;
    if (!file.is_open()) {
        printf("ERROR: Cannot open %s\n", path);
        return lanes;
    }
    getline(file, line); // skip header
    while (getline(file, line)) {
        stringstream ss(line);
        string lname, idx, xs, ys;
        getline(ss, lname, ',');
        getline(ss, idx, ',');
        getline(ss, xs, ',');
        getline(ss, ys, ',');
        int x = stoi(xs);
        int y = stoi(ys);

        auto it = find_if(lanes.begin(), lanes.end(),
                          [&lname](const Lane& l){ return l.name == lname; });
        if (it == lanes.end()) {
            Lane ln; ln.name = lname; ln.points.push_back(Point(x,y));
            lanes.push_back(ln);
        } else {
            it->points.push_back(Point(x,y));
        }
    }
    return lanes;
}

// ==================== POINT IN POLYGON ====================
bool point_in_polygon(int x, int y, const vector<Point>& poly) {
    bool c = false;
    int n = (int)poly.size();
    for (int i = 0, j = n-1; i < n; j = i++) {
        int xi = poly[i].x, yi = poly[i].y;
        int xj = poly[j].x, yj = poly[j].y;
        if (((yi > y) != (yj > y)) &&
            (x < (xj - xi) * (y - yi) / (yj - yi + 1e-6) + xi))
            c = !c;
    }
    return c;
}

// ==================== NMS ====================
float iou(const Detection& a, const Detection& b) {
    float x1 = max(a.cx - a.w/2, b.cx - b.w/2);
    float y1 = max(a.cy - a.h/2, b.cy - b.h/2);
    float x2 = min(a.cx + a.w/2, b.cx + b.w/2);
    float y2 = min(a.cy + a.h/2, b.cy + b.h/2);
    float inter = max(0.0f, x2 - x1) * max(0.0f, y2 - y1);
    float area = a.w * a.h + b.w * b.h - inter;
    return area > 0 ? inter / area : 0;
}

vector<Detection> nms(vector<Detection>& dets) {
    sort(dets.begin(), dets.end(),
         [](const Detection& a, const Detection& b){ return a.conf > b.conf; });

    vector<Detection> keep;
    vector<bool> suppressed(dets.size(), false);
    for (size_t i = 0; i < dets.size(); i++) {
        if (suppressed[i]) continue;
        keep.push_back(dets[i]);
        for (size_t j = i+1; j < dets.size(); j++) {
            if (suppressed[j]) continue;
            if (dets[i].class_id == dets[j].class_id && iou(dets[i], dets[j]) > NMS_THRESH)
                suppressed[j] = true;
        }
    }
    return keep;
}

static inline int8_t qnt_f32_to_affine(float f32, int32_t zp, float scale)
{
    float dst_val = (f32 / scale) + zp;
    if (dst_val < -128.f) dst_val = -128.f;
    if (dst_val > 127.f) dst_val = 127.f;
    return (int8_t)dst_val;
}

static inline float deqnt_affine_to_f32(int8_t qnt, int32_t zp, float scale)
{
    return ((float)qnt - (float)zp) * scale;
}

static int process_i8_rv1106(
    int8_t *input,
    const int *anchor,
    int grid_h,
    int grid_w,
    int stride,
    float threshold,
    int32_t zp,
    float scale,
    vector<Detection>& dets)
{
    int valid_count = 0;
    int8_t thres_i8 = qnt_f32_to_affine(threshold, zp, scale);
    int anchor_per_branch = 3;
    int align_c = YOLO_PROP_BOX_SIZE * anchor_per_branch;

    for (int h = 0; h < grid_h; h++) {
        for (int w = 0; w < grid_w; w++) {
            for (int a = 0; a < anchor_per_branch; a++) {
                int hw_offset = h * grid_w * align_c + w * align_c + a * YOLO_PROP_BOX_SIZE;
                int8_t *hw_ptr = input + hw_offset;
                int8_t box_confidence = hw_ptr[4];

                if (box_confidence >= thres_i8) {
                    int8_t max_class_probs = hw_ptr[5];
                    int max_class_id = 0;
                    for (int k = 1; k < YOLO_CLASS_NUM; ++k) {
                        int8_t prob = hw_ptr[5 + k];
                        if (prob > max_class_probs) {
                            max_class_id = k;
                            max_class_probs = prob;
                        }
                    }

                    float box_conf_f32 = deqnt_affine_to_f32(box_confidence, zp, scale);
                    float class_prob_f32 = deqnt_affine_to_f32(max_class_probs, zp, scale);
                    float score = box_conf_f32 * class_prob_f32;

                    if (score > threshold) {
                        float box_x = deqnt_affine_to_f32(hw_ptr[0], zp, scale) * 2.0f - 0.5f;
                        float box_y = deqnt_affine_to_f32(hw_ptr[1], zp, scale) * 2.0f - 0.5f;
                        float box_w = deqnt_affine_to_f32(hw_ptr[2], zp, scale) * 2.0f;
                        float box_h = deqnt_affine_to_f32(hw_ptr[3], zp, scale) * 2.0f;
                        box_w = box_w * box_w;
                        box_h = box_h * box_h;

                        box_x = (box_x + w) * (float)stride;
                        box_y = (box_y + h) * (float)stride;
                        box_w *= (float)anchor[a * 2];
                        box_h *= (float)anchor[a * 2 + 1];

                        Detection d;
                        d.cx = box_x;
                        d.cy = box_y;
                        d.w = box_w;
                        d.h = box_h;
                        d.conf = score;
                        d.class_id = max_class_id;
                        dets.push_back(d);
                        valid_count++;
                    }
                }
            }
        }
    }
    return valid_count;
}

vector<Detection> post_process_yolov5_i8(
    const vector<rknn_tensor_mem*>& output_mems,
    const vector<rknn_tensor_attr>& output_attrs,
    int model_h)
{
    vector<Detection> raw_dets;
    int branches = min((int)output_mems.size(), 3);
    for (int i = 0; i < branches; ++i) {
        if (!output_mems[i] || !output_mems[i]->virt_addr) continue;
        int grid_h = output_attrs[i].dims[2];
        int grid_w = output_attrs[i].dims[1];
        if (grid_h <= 0 || grid_w <= 0) continue;
        int stride = model_h / grid_h;
        process_i8_rv1106(
            (int8_t*)output_mems[i]->virt_addr,
            YOLO_ANCHORS[i],
            grid_h,
            grid_w,
            stride,
            OBJ_THRESH,
            output_attrs[i].zp,
            output_attrs[i].scale,
            raw_dets);
    }

    vector<Detection> filtered;
    for (auto& d : raw_dets) {
        bool is_vehicle = false;
        for (int vc : VEHICLE_CLASSES) {
            if (d.class_id == vc) {
                is_vehicle = true;
                break;
            }
        }
        if (is_vehicle) filtered.push_back(d);
    }
    return nms(filtered);
}

// ==================== POST PROCESS ====================
vector<Detection> post_process(float* buf, const rknn_tensor_attr& attr, int orig_w, int orig_h) {
    vector<Detection> dets;
    // YOLOv8 RKNN output is typically [1, 84, 8400] or [1, 8400, 84]
    int d1 = attr.dims[1];
    int d2 = attr.dims[2];
    int feat_len, num_anchors;
    bool transposed = false;

    if (d1 == 84 && d2 == 8400) {
        feat_len = 84; num_anchors = 8400; transposed = false;
    } else if (d1 == 8400 && d2 == 84) {
        feat_len = 84; num_anchors = 8400; transposed = true;
    } else {
        printf("WARN: Unexpected output shape [%d, %d, %d], skipping post-process\n",
               attr.dims[0], d1, d2);
        return dets;
    }
    int num_classes = feat_len - 4;

    for (int i = 0; i < num_anchors; i++) {
        float cx, cy, w, h;
        if (!transposed) {
            // [1, 84, 8400]
            cx = buf[0 * num_anchors + i];
            cy = buf[1 * num_anchors + i];
            w  = buf[2 * num_anchors + i];
            h  = buf[3 * num_anchors + i];
        } else {
            // [1, 8400, 84]
            cx = buf[i * feat_len + 0];
            cy = buf[i * feat_len + 1];
            w  = buf[i * feat_len + 2];
            h  = buf[i * feat_len + 3];
        }

        float best_score = 0;
        int   best_cls   = 0;
        for (int c = 4; c < feat_len; c++) {
            float s = transposed ? buf[i * feat_len + c] : buf[c * num_anchors + i];
            if (s > best_score) { best_score = s; best_cls = c - 4; }
        }

        if (best_score > OBJ_THRESH) {
            bool is_vehicle = false;
            for (int vc : VEHICLE_CLASSES) if (best_cls == vc) { is_vehicle = true; break; }
            if (is_vehicle) {
                Detection d = {cx, cy, w, h, best_score, best_cls};
                dets.push_back(d);
            }
        }
    }
    return nms(dets);
}

// ==================== IMAGE LIST ====================
vector<string> list_images(const char* dir) {
    vector<string> out;
    DIR* d = opendir(dir);
    if (!d) return out;
    struct dirent* ent;
    while ((ent = readdir(d)) != NULL) {
        string n = ent->d_name;
        size_t len = n.size();
        if ((len > 4 && n.substr(len-4) == ".jpg") ||
            (len > 4 && n.substr(len-4) == ".png") ||
            (len > 5 && n.substr(len-5) == ".jpeg"))
            out.push_back(n);
    }
    closedir(d);
    sort(out.begin(), out.end());
    return out;
}

// ==================== MAIN ====================
int main(int argc, char** argv) {
    // --- 1. Load RKNN model ---
    rknn_context ctx;
    int ret = rknn_init(&ctx, (char*)MODEL_PATH, 0, 0, NULL);
    if (ret < 0) { printf("ERROR: rknn_init failed %d\n", ret); return -1; }
    printf("--> RKNN model loaded\n");

    // --- 2. Query I/O info ---
    rknn_input_output_num io_num;
    rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));

    rknn_tensor_attr input_attr;
    memset(&input_attr, 0, sizeof(input_attr));
    input_attr.index = 0;
    rknn_query(ctx, RKNN_QUERY_NATIVE_INPUT_ATTR, &input_attr, sizeof(input_attr));
    printf("Input dim: [%d,%d,%d,%d], fmt=%d, type=%d, size=%d\n",
           input_attr.dims[0], input_attr.dims[1], input_attr.dims[2], input_attr.dims[3],
           input_attr.fmt, input_attr.type, input_attr.size);

    int model_h = MODEL_H;
    int model_w = MODEL_W;
    if (input_attr.fmt == RKNN_TENSOR_NHWC) {
        model_h = input_attr.dims[1];
        model_w = input_attr.dims[2];
    } else if (input_attr.fmt == RKNN_TENSOR_NCHW) {
        model_h = input_attr.dims[2];
        model_w = input_attr.dims[3];
    }
    printf("Using model input size: %dx%d\n", model_w, model_h);

    // Use io_mem input path (official RV1106 pattern)
    input_attr.type = RKNN_TENSOR_UINT8;
    input_attr.fmt = RKNN_TENSOR_NHWC;
    rknn_tensor_mem* input_mem = rknn_create_mem(ctx, input_attr.size_with_stride);
    if (!input_mem) { printf("ERROR: rknn_create_mem input failed\n"); return -1; }
    ret = rknn_set_io_mem(ctx, input_mem, &input_attr);
    if (ret < 0) { printf("ERROR: rknn_set_io_mem input failed %d\n", ret); return -1; }

    vector<rknn_tensor_attr> output_attrs(io_num.n_output);
    vector<rknn_tensor_mem*> output_mems(io_num.n_output, nullptr);
    for (uint32_t i = 0; i < io_num.n_output; ++i) {
        memset(&output_attrs[i], 0, sizeof(rknn_tensor_attr));
        output_attrs[i].index = i;
        rknn_query(ctx, RKNN_QUERY_NATIVE_NHWC_OUTPUT_ATTR, &output_attrs[i], sizeof(rknn_tensor_attr));
        output_mems[i] = rknn_create_mem(ctx, output_attrs[i].size_with_stride);
        if (!output_mems[i]) {
            printf("ERROR: rknn_create_mem output[%u] failed\n", i);
            return -1;
        }
        ret = rknn_set_io_mem(ctx, output_mems[i], &output_attrs[i]);
        if (ret < 0) {
            printf("ERROR: rknn_set_io_mem output[%u] failed %d\n", i, ret);
            return -1;
        }
        if (i == 0) {
            printf("Output dim: [%d,%d,%d,%d]\n",
                   output_attrs[i].dims[0], output_attrs[i].dims[1], output_attrs[i].dims[2], output_attrs[i].dims[3]);
        }
    }

    // --- 3. Load metadata ---
    vector<Lane> lanes = load_polygons(POLYGON_CSV);
    vector<string> images = list_images(IMAGE_DIR);
    printf("--> Found %zu images, %zu lanes\n", images.size(), lanes.size());
    if (!ensure_dir(OUTPUT_IMAGE_DIR)) {
        printf("WARN: Cannot create output image dir %s\n", OUTPUT_IMAGE_DIR);
    }

    // --- 4. Prepare CSV ---
    ofstream csv(OUTPUT_CSV);
    if (!csv.is_open()) { printf("ERROR: Cannot write %s\n", OUTPUT_CSV); return -1; }
    csv << "image,latency_ms";
    for (auto& l : lanes) csv << "," << l.name;
    csv << "\n";

    // --- 5. Inference loop ---
    for (const string& img_name : images) {
        string path = string(IMAGE_DIR) + "/" + img_name;
        Mat frame = imread(path);
        if (frame.empty()) { printf("WARN: Cannot read %s\n", path.c_str()); continue; }

        int orig_w = frame.cols;
        int orig_h = frame.rows;
        Mat annotated = frame.clone();

        // Preprocess: resize + BGR2RGB
        Mat resized;
        resize(frame, resized, Size(model_w, model_h));
        cvtColor(resized, resized, COLOR_BGR2RGB);

        int stride_w = input_attr.w_stride > 0 ? input_attr.w_stride : model_w;
        if (stride_w == model_w) {
            memcpy(input_mem->virt_addr, resized.data, model_w * model_h * 3);
        } else {
            uint8_t* dst = (uint8_t*)input_mem->virt_addr;
            const uint8_t* src = resized.data;
            int src_row = model_w * 3;
            int dst_row = stride_w * 3;
            for (int h = 0; h < model_h; ++h) {
                memcpy(dst, src, src_row);
                src += src_row;
                dst += dst_row;
            }
        }

        // Run
        struct timeval t0, t1;
        gettimeofday(&t0, NULL);
        ret = rknn_run(ctx, NULL);
        if (ret != RKNN_SUCC) { printf("WARN: rknn_run failed\n"); continue; }

        // output data already populated in output_mems by zero-copy io_mem path
        ret = RKNN_SUCC;
        gettimeofday(&t1, NULL);

        float infer_ms = (t1.tv_sec - t0.tv_sec) * 1000.0f +
                         (t1.tv_usec - t0.tv_usec) / 1000.0f;

        // Post-process
        vector<Detection> dets;
        if (ret == RKNN_SUCC) {
            dets = post_process_yolov5_i8(output_mems, output_attrs, model_h);
        } else {
            printf("WARN: rknn_outputs_get failed %d, writing latency with zero detections\n", ret);
        }

        // Lane counting
        vector<int> counts(lanes.size(), 0);
        vector<int> obj_counts(80, 0);
        float scale_x = (float)orig_w / model_w;
        float scale_y = (float)orig_h / model_h;

        for (auto& d : dets) {
            if (d.class_id >= 0 && d.class_id < (int)obj_counts.size()) {
                obj_counts[d.class_id]++;
            }
            int cx = (int)(d.cx * scale_x);
            int cy = (int)(d.cy * scale_y);
            int x1 = max(0, (int)((d.cx - d.w * 0.5f) * scale_x));
            int y1 = max(0, (int)((d.cy - d.h * 0.5f) * scale_y));
            int x2 = min(orig_w - 1, (int)((d.cx + d.w * 0.5f) * scale_x));
            int y2 = min(orig_h - 1, (int)((d.cy + d.h * 0.5f) * scale_y));
            rectangle(annotated, Point(x1, y1), Point(x2, y2), Scalar(0, 255, 255), 2);

            int lane_idx = -1;
            for (size_t li = 0; li < lanes.size(); li++) {
                if (point_in_polygon(cx, cy, lanes[li].points)) {
                    counts[li]++;
                    lane_idx = (int)li;
                }
            }
            string lbl = class_name(d.class_id) + " " + to_string((int)(d.conf * 100.0f)) + "%";
            if (lane_idx >= 0) lbl += " " + lanes[lane_idx].name;
            putText(annotated, lbl, Point(x1, max(15, y1 - 6)), FONT_HERSHEY_SIMPLEX, 0.45, Scalar(0, 255, 255), 1);
        }

        for (size_t li = 0; li < lanes.size(); ++li) {
            if (lanes[li].points.size() >= 2) {
                vector<vector<Point>> polys = {lanes[li].points};
                polylines(annotated, polys, true, Scalar(0, 255, 0), 2);
                Point p = lanes[li].points[0];
                putText(annotated, lanes[li].name + ": " + to_string(counts[li]),
                        Point(max(0, p.x), max(18, p.y)), FONT_HERSHEY_SIMPLEX, 0.55, Scalar(0, 255, 0), 2);
            }
        }

        string obj_summary = "Objects: ";
        bool has_obj = false;
        for (int cid : VEHICLE_CLASSES) {
            if (cid >= 0 && cid < (int)obj_counts.size() && obj_counts[cid] > 0) {
                if (has_obj) obj_summary += ", ";
                obj_summary += class_name(cid) + "=" + to_string(obj_counts[cid]);
                has_obj = true;
            }
        }
        if (!has_obj) obj_summary += "none";
        putText(annotated, obj_summary, Point(8, 24), FONT_HERSHEY_SIMPLEX, 0.65, Scalar(255, 255, 0), 2);

        // Log
        csv << img_name << "," << infer_ms;
        for (size_t li = 0; li < lanes.size(); li++) csv << "," << counts[li];
        csv << "\n";
        printf("Processed %s - %.1f ms, %zu detections\n", img_name.c_str(), infer_ms, dets.size());

        string out_img = string(OUTPUT_IMAGE_DIR) + "/" + with_jpg_ext(img_name);
        if (!save_annotated_image(out_img, annotated)) {
            printf("WARN: failed to save %s\n", out_img.c_str());
        }

    }

    csv.close();
    if (input_mem) {
        rknn_destroy_mem(ctx, input_mem);
    }
    for (auto* mem : output_mems) {
        if (mem) {
            rknn_destroy_mem(ctx, mem);
        }
    }
    rknn_destroy(ctx);
    printf("--> Done. Results saved to %s\n", OUTPUT_CSV);
    return 0;
}
