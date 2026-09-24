import os
from rknn.api import RKNN

ONNX_MODEL_PATH = "./best.onnx" 
RKNN_MODEL_PATH = "../best.rknn"
CALIBRATION_FILE_PATH = "../calibration/calibration_data.txt"
TARGET_PLATFORM = "rk3588" 

if __name__ == "__main__":
    rknn = RKNN(verbose=False)
    
    print(f"Applying chip configurations for target: {TARGET_PLATFORM}")
    rknn.config(
        target_platform=TARGET_PLATFORM,
        optimization_level=3,
        quantized_dtype="w8a8",
        mean_values=[0, 0, 0],        
        std_values=[255, 255, 255]          
    )
    
    print(f"Reading ONNX model structure: {ONNX_MODEL_PATH}")
    if not os.path.exists(ONNX_MODEL_PATH):
        print(f"Error: Missing ONNX input file at {ONNX_MODEL_PATH}")
        exit(-1)
        
    ret = rknn.load_onnx(model=ONNX_MODEL_PATH)
    if ret != 0:
        print("Error: Failed to parse input ONNX model structure")
        exit(ret)

    
    original_dir = os.getcwd()
    output_dir = "./internal"
    os.makedirs(output_dir, exist_ok=True)
    os.chdir(output_dir)

    print("Analyzing model layers to generate quantization map file...")
    if not os.path.exists(CALIBRATION_FILE_PATH):
        print(f"Error: Missing calibration image reference list at {CALIBRATION_FILE_PATH}")
        exit(-1)
    
        
    ret = rknn.hybrid_quantization_step1(dataset=CALIBRATION_FILE_PATH)
    if ret != 0:
        print("Error: Failed to analyze model layers for integer conversion")
        exit(ret)
    
    cfg_file = "./best.quantization.cfg"
    print(f"Excluding final prediction heads from integer compression inside: {cfg_file}")
    
    if os.path.exists(cfg_file):
        with open(cfg_file, "r") as f:
            lines = f.readlines()
        
        with open(cfg_file, "w") as f:
            for line in lines:
                if "custom_quantize_layers:" in line:
                    f.write("custom_quantize_layers:\n  output0-rs: float16\n  output0: float16\n")
                else:
                    f.write(line)
    else:
        print(f"Error: Quantization map configuration file was not created at {cfg_file}")
        exit(-1)

    print("Compiling final hybrid model using modified layer guidelines...")
    ret = rknn.hybrid_quantization_step2(
        model_input="./best.model",
        data_input="./best.data",
        model_quantization_cfg=cfg_file
    )
    if ret != 0:
        print("Error: Model compilation failed during hybrid building stage")
        exit(ret)
        
    print(f"Saving compiled hardware-accelerated binary file to: {RKNN_MODEL_PATH}")
    ret = rknn.export_rknn(RKNN_MODEL_PATH)
    if ret != 0:
        print("Error: Failed to write final compiled file to storage disk")
        exit(ret)
        
    print("Success")
    rknn.release()

    os.chdir("../" + output_dir)
