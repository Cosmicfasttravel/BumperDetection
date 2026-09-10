from marshal import version

from ultralytics import YOLO
from roboflow import Roboflow
import os
import torch
import shutil
import glob
import time
import cv2

INPUT_SIZE = 320

def check_cuda():
    print("="*60)
    print("CHECKING CUDA AVAILABILITY")
    print("="*60)
    
    if torch.cuda.is_available():
        print(f"✓ CUDA is available!")
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA Version: {torch.version.cuda}")
        device = '0'
    else:
        print("✗ CUDA not available - will use CPU")
        device = 'cpu'
    
    return device

def download_dataset():
    print("\n" + "="*60)
    print("DOWNLOADING DATASET FROM ROBOFLOW")
    print("="*60)

    # Edit these lines
    rf = Roboflow(api_key="")
    project = rf.workspace("train-1jrhy").project("yolodataset-dwciq")
    version = project.version(2)
    dataset = version.download("yolo26")
    
    print(f"\n✓ Dataset downloaded to: {dataset.location}")
    return dataset

def load_model():
    print("\n" + "="*60)
    print("LOADING PRE-TRAINED YOLO MODEL")
    print("="*60)
    
    model = YOLO("yolo26n.pt")
    return model

def train_model(model, dataset, device):
    print("\n" + "="*60)
    print("STARTING TRAINING")
    print("="*60)
    
    results = model.train(
        data=f'{dataset.location}/data.yaml',
        epochs=1,
        imgsz=INPUT_SIZE,
        batch=8 if device == '0' else 8,
        optimizer='AdamW',
        lr0=0.001,
        patience=10,
        name='bumper_detector_pi',
        exist_ok=True,
        device=device,
        workers=8 if device == '0' else 2,
        verbose=True,
        plots=True,
        augment=True,
    )
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    
    return results

def evaluate_model(model):
    print("\n" + "="*60)
    print("EVALUATING MODEL")
    print("="*60)
    
    metrics = model.val()
    return metrics

def export_model(model):
    best_model = YOLO('runs/detect/bumper_detector_pi/weights/best.pt')
    
    onnx_path = best_model.export(
        format='onnx',
        imgsz=INPUT_SIZE,
        simplify=True,
        dynamic=False,
        opset=12,
        end2end = False,
    )

    output_dir = "../onnx2rknn"
    os.makedirs(output_dir, exist_ok=True)
    
    output_path_compat = os.path.join(output_dir, os.path.basename(onnx_path))
    shutil.copy(onnx_path, output_path_compat)
    
    return onnx_path, output_path_compat, best_model

def test_model(best_model, dataset):
    test_images = glob.glob(f'{dataset.location}/valid/images/*.jpg')[:5]
    
    for img_path in test_images:
        results = best_model(img_path, conf=0.25, imgsz=320)
        for result in results:
            boxes = result.boxes
            for box in boxes:
                conf = box.conf[0].item()

def benchmark_model(best_model, dataset):
    test_images = glob.glob(f'{dataset.location}/valid/images/*.jpg')
    if not test_images:
        return
    
    test_img = cv2.imread(test_images[0])
    for _ in range(5):
        _ = best_model(test_img, verbose=False, imgsz=INPUT_SIZE)
    
    num_runs = 20
    start = time.time()
    for _ in range(num_runs):
        _ = best_model(test_img, verbose=False, imgsz=INPUT_SIZE)
    elapsed = time.time() - start
    
    avg_time = (elapsed / num_runs) * 1000
    fps_gpu = 1000 / avg_time
    pi_inference_time = avg_time * 15
    pi_fps = 1000 / pi_inference_time

def print_summary(metrics, output_path, device):
    print("\n" + "="*60)
    print("TRAINING COMPLETE - SUMMARY")
    print("="*60)
    print(f"\nModel files: {output_path}")

if __name__ == '__main__':
    device = check_cuda()
    dataset = download_dataset()
    model = load_model()
    results = train_model(model, dataset, device)
    metrics = evaluate_model(model)
    onnx_path, output_path, best_model = export_model(model)
    test_model(best_model, dataset)
    benchmark_model(best_model, dataset)
    print_summary(metrics, output_path, device)