import torch
import torchvision
from torchvision import transforms
from torchvision.datasets import VOCDetection
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import time


def get_loader(root="./data", batch_size=4):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    dataset = VOCDetection(
        root=root,
        year="2007",
        image_set="test",
        download=True,
        transform=transform
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda x: tuple(zip(*x))
    )

    return loader


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    model = fasterrcnn_resnet50_fpn(pretrained=True)
    model.to(device)
    model.eval()

    loader = get_loader()

    total_time = 0
    total_images = 0

    with torch.no_grad():
        for images, targets in loader:
            images = [img.to(device) for img in images]

            start = time.time()
            outputs = model(images)
            end = time.time()

            total_time += (end - start)
            total_images += len(images)

    ms_per_image = (total_time / total_images) * 1000

    print("\nFaster R-CNN Results:")
    print(f"Inference time: {ms_per_image:.2f} ms/image")
    print("mAP: (OPTIONAL – requires COCO/VOC evaluator, can approximate or omit)")


if __name__ == "__main__":
    main()