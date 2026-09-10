import os
from rknn.api import RKNN

ONNX_MODEL_PATH = "./LATEST.onnx" 
RKNN_MODEL_PATH = "./LATEST.rknn"

TARGET_PLATFORM = "rk3588" 

if __name__ == "__main__":
    rknn = RKNN(verbose=False)
    

    print(f"--> Configuring for {TARGET_PLATFORM} in FP16 precision...")
    rknn.config(
        target_platform=TARGET_PLATFORM,
        optimization_level=3,
        float_dtype="float16",   
        mean_values=[0, 0,0 ],        
        std_values=[255, 255, 255]          
    )
    
    print(f"--> Loading ONNX model from: {ONNX_MODEL_PATH}")
    if not os.path.exists(ONNX_MODEL_PATH):
        print(f"Error: Could not locate {ONNX_MODEL_PATH}")
        exit(-1)
        
    ret = rknn.load_onnx(model=ONNX_MODEL_PATH)
    if ret != 0:
        print("Error: Loading ONNX model graph failed!")
        exit(ret)

    # Enable quant later        
    print("--> Building RKNN model graph...")
    ret = rknn.build(do_quantization=False, dataset=None)
    if ret != 0:
        print("Error: Building RKNN model failed!")
        exit(ret)
        
    print(f"--> Exporting final compiled file to: {RKNN_MODEL_PATH}")
    ret = rknn.export_rknn(RKNN_MODEL_PATH)
    if ret != 0:
        print("Error: Exporting RKNN file failed!")
        exit(ret)
        
    print("Successfully converted 320x320 ONNX model to FP16 RKNN format!")
    rknn.release()
