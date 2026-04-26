import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from PIL import Image, ImageDraw
import torchvision.transforms as T


COCO_CLASSES = [
    "__background__", "person", "bicycle", "car", "motorcycle",
    "airplane", "bus", "train", "truck", "boat"
]


def load_model(device):
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    model.to(device)
    model.eval()
    return model


def predict(model, image_path, device, threshold=0.5):
    transform = T.Compose([T.ToTensor()])
    image = Image.open(image_path).convert("RGB")
    img_tensor = transform(image).to(device)

    with torch.no_grad():
        outputs = model([img_tensor])[0]

    boxes = outputs["boxes"]
    scores = outputs["scores"]
    labels = outputs["labels"]

    draw = ImageDraw.Draw(image)

    for box, score, label in zip(boxes, scores, labels):
        if score < threshold:
            continue

        x1, y1, x2, y2 = box.tolist()
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)

        cls_name = COCO_CLASSES[label] if label < len(COCO_CLASSES) else str(label)
        draw.text((x1, y1), f"{cls_name}: {score:.2f}", fill="red")

    return image


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(device)

    img1 = "image1.jpg"
    img2 = "image2.jpg"

    out1 = predict(model, img1, device)
    out2 = predict(model, img2, device)

    out1.save("output1.jpg")
    out2.save("output2.jpg")

    print("Saved output1.jpg and output2.jpg")


if __name__ == "__main__":
    main()