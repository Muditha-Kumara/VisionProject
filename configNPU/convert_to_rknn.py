from rknn.api import RKNN

def convert():
    rknn = RKNN(verbose=True)

    # 1. Configuration for RV1106 (Luckfox Pico Zero)
    print('--> Configuring RKNN')
    rknn.config(
        mean_values=[[0, 0, 0]], 
        std_values=[[255, 255, 255]], 
        target_platform='rv1106'
    )

    # 2. Load the ONNX model you generated earlier
    print('--> Loading ONNX model')
    ret = rknn.load_onnx(model='check3_fuse_ops.onnx')
    if ret != 0:
        print('Load failed!'); return

    # 3. Build the model with INT8 Quantization
    print('--> Building RKNN model (Quantization on)')
    ret = rknn.build(do_quantization=True, dataset='./dataset.txt')
    if ret != 0:
        print('Build failed!'); return

    # 4. Export the final model
    print('--> Exporting yolov8n.rknn')
    ret = rknn.export_rknn('yolov8n.rknn')
    if ret != 0:
        print('Export failed!'); return

    print('Conversion Complete!')
    rknn.release()

if __name__ == '__main__':
    convert()
