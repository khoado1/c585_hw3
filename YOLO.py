from ultralytics import YOLO
import time


def evaluate():
    model = YOLO("yolov8n.pt")

    results = model.val(data="voc.yaml", imgsz=640)

    print("\nYOLO Evaluation:")
    print(results)


def inference_speed():
    model = YOLO("yolov8n.pt")

    start = time.time()
    results = model.predict(source="https://ultralytics.com/images/bus.jpg", imgsz=640)
    end = time.time()

    print(f"Inference time: {(end-start)*1000:.2f} ms")


def fine_tune():
    model = YOLO("yolov8n.pt")

    model.train(
        data="voc.yaml",
        epochs=3,
        imgsz=640,
        batch=16
    )


def main():
    print("Running YOLO evaluation...")
    evaluate()

    print("\nRunning inference speed test...")
    inference_speed()

    print("\nFine-tuning YOLO...")
    fine_tune()


if __name__ == "__main__":
    main()