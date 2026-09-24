import os
import shutil
from dotenv import load_dotenv
from roboflow import Roboflow
from ultralytics import YOLO

load_dotenv()
api_key = os.getenv("ROBOFLOW_API_KEY")

INPUT_SIZE = 320
EPOCHS = 250
NAME = 'Yolo_v26s_' + str(EPOCHS) + 'ep'

def download_dataset():
    print("DOWNLOADING DATASET FROM ROBOFLOW")
    rf = Roboflow(api_key=api_key)
    project = rf.workspace("train-1jrhy").project("frc-bumper-detection-mwwkd-uddpe")
    version = project.version(2)
    dataset = version.download(
        "yolo26",
        location="./dataset",
        overwrite=True,
    )
    return dataset

def train_and_export(dataset):
    print("STARTING TRAINING")
    model = YOLO("yolo26s.pt")
    
    model.train(
        data=f'{dataset.location}/data.yaml',
        epochs=EPOCHS,
        imgsz=INPUT_SIZE,

        batch=32,
        workers=16,

        optimizer='MuSGD', lr0=0.01,

        patience=25,
        name=NAME,

        exist_ok=True,
        verbose=True,
        plots=True,
        augment=True,
        amp=True,
        cos_lr=True,

        close_mosaic=15,
        iou=0.65,

        mosaic=1.0, degrees=10.0, scale=0.5, fliplr=0.5,
        mixup=0.15, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4
    )
    
    print("EXPORTING TO ONNX")
    best_model = YOLO('runs/detect/' + NAME + '/weights/best.pt')
    onnx_path = best_model.export(
        format='onnx',
        imgsz=INPUT_SIZE,
        simplify=True,
        dynamic=False,
        opset=12,
        nms=None,
    )

    output_dir = "./"
    os.makedirs(output_dir, exist_ok=True)
    output_path_compat = os.path.join(output_dir, os.path.basename(onnx_path))
    shutil.copy(onnx_path, output_path_compat)
    
    print(f"Model successfully saved to: {output_path_compat}")

if __name__ == '__main__':
    dataset = download_dataset()
    train_and_export(dataset)
