from ultralytics import YOLO
import torch

# print(torch.cuda.is_available())  # True dönerse CUDA aktif
# print(torch.cuda.current_device())  # GPU'nun cihaz numarasını gösterir
# print(torch.cuda.get_device_name(0))  # Kullanılabilir GPU'nun adını gösterir



def main():
    # Load a COCO-pretrained YOLOv8n model

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = YOLO("yolov8n.pt")

    # Display model information (optional)
    model.info()

    # Train the model on the COCO8 example dataset for 100 epochs
    results = model.train(data="D:\Deep Learning Dersleri\Yolo\dataset\data.yaml", epochs=5, imgsz=640, device=device)
    # results = model.train(data="coco128.yaml", epochs=5, imgsz=640, amp=False)


if __name__ == "__main__":
    main()