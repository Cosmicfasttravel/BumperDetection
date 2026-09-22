import os
import shutil
from dotenv import load_dotenv
from roboflow import Roboflow
from ultralytics import YOLO

load_dotenv()
api_key = os.getenv("ROBOFLOW_API_KEY")

INPUT_SIZE = 320

def download_dataset():
    print("DOWNLOADING DATASET FROM ROBOFLOW")
    rf = Roboflow(api_key=api_key)
    project = rf.workspace("train-1jrhy").project("yolodataset-dwciq")
    version = project.version(2)
    dataset = version.download("yolo26")
    return dataset

def train_and_export(dataset):
    print("STARTING TRAINING")
    model = YOLO("yolo26s.pt")
    
    model.train(
        data=f'{dataset.location}/data.yaml',
        epochs=1000,
        imgsz=INPUT_SIZE,
        batch=8,
        optimizer='MuSGD',
        lr0=0.01,
        patience=50,
        name='bumper_detector_pi',
        exist_ok=True,
        workers=8,
        verbose=True,
        plots=True,
        augment=True,
        close_mosaic=15,
        iou=0.65,
    )
    
    print("EXPORTING TO ONNX")
    best_model = YOLO('runs/detect/bumper_detector_pi/weights/best.pt')
    onnx_path = best_model.export(
        format='onnx',
        imgsz=INPUT_SIZE,
        simplify=True,
        dynamic=False,
        opset=19,
        end2end=False,
    )

    output_dir = "../onnx2rknn"
    os.makedirs(output_dir, exist_ok=True)
    output_path_compat = os.path.join(output_dir, os.path.basename(onnx_path))
    shutil.copy(onnx_path, output_path_compat)
    
    print(f"Model successfully saved to: {output_path_compat}")

if __name__ == '__main__':
    dataset = download_dataset()
    train_and_export(dataset)
