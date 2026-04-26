# ann_snn.py
import argparse
import time
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import snntorch as snn
from snntorch import spikegen


# ----------------------------
# 1. CNN ARCHITECTURE
# Must match your cnn.py model
# ----------------------------
class CIFAR10CNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.features = nn.Sequential(
            # input: 3 x 32 x 32
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 64 x 16 x 16

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 128 x 8 x 8

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 256 x 4 x 4
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# ----------------------------
# 2. Converted SNN
# ----------------------------
class ConvertedSNN(nn.Module):
    def __init__(self, cnn_model, beta=0.95):
        super().__init__()

        # features indices from your cnn.py
        self.conv1 = cnn_model.features[0]
        self.bn1 = cnn_model.features[1]
        self.lif1 = snn.Leaky(beta=beta)

        self.conv2 = cnn_model.features[3]
        self.bn2 = cnn_model.features[4]
        self.lif2 = snn.Leaky(beta=beta)
        self.pool1 = cnn_model.features[6]

        self.conv3 = cnn_model.features[7]
        self.bn3 = cnn_model.features[8]
        self.lif3 = snn.Leaky(beta=beta)

        self.conv4 = cnn_model.features[10]
        self.bn4 = cnn_model.features[11]
        self.lif4 = snn.Leaky(beta=beta)
        self.pool2 = cnn_model.features[13]

        self.conv5 = cnn_model.features[14]
        self.bn5 = cnn_model.features[15]
        self.lif5 = snn.Leaky(beta=beta)
        self.pool3 = cnn_model.features[17]

        # classifier indices from your cnn.py
        self.flatten = cnn_model.classifier[0]
        self.fc1 = cnn_model.classifier[1]
        self.lif6 = snn.Leaky(beta=beta)

        # skip classifier[3] Dropout for SNN inference
        self.fc2 = cnn_model.classifier[4]

    def forward(self, x, num_steps):
        batch_size = x.size(0)

        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        mem4 = self.lif4.init_leaky()
        mem5 = self.lif5.init_leaky()
        mem6 = self.lif6.init_leaky()

        output_sum = torch.zeros(batch_size, 10, device=x.device)
        total_spikes = 0

        spike_data = spikegen.rate(x, num_steps=num_steps)

        for step in range(num_steps):
            cur = spike_data[step]

            cur = self.conv1(cur)
            cur = self.bn1(cur)
            spk1, mem1 = self.lif1(cur, mem1)
            total_spikes += spk1.sum().item()

            cur = self.conv2(spk1)
            cur = self.bn2(cur)
            spk2, mem2 = self.lif2(cur, mem2)
            total_spikes += spk2.sum().item()
            cur = self.pool1(spk2)

            cur = self.conv3(cur)
            cur = self.bn3(cur)
            spk3, mem3 = self.lif3(cur, mem3)
            total_spikes += spk3.sum().item()

            cur = self.conv4(spk3)
            cur = self.bn4(cur)
            spk4, mem4 = self.lif4(cur, mem4)
            total_spikes += spk4.sum().item()
            cur = self.pool2(spk4)

            cur = self.conv5(cur)
            cur = self.bn5(cur)
            spk5, mem5 = self.lif5(cur, mem5)
            total_spikes += spk5.sum().item()
            cur = self.pool3(spk5)

            cur = self.flatten(cur)

            cur = self.fc1(cur)
            spk6, mem6 = self.lif6(cur, mem6)
            total_spikes += spk6.sum().item()

            out = self.fc2(spk6)

            output_sum += out

        return output_sum, total_spikes


# ----------------------------
# 3. Data Loaders
# ----------------------------
def get_test_loader(batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2470, 0.2435, 0.2616)
        )
    ])

    test_set = torchvision.datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=transform
    )

    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )

    return test_loader


# ----------------------------
# 4. Evaluation
# ----------------------------
def evaluate_snn(model, test_loader, device, num_steps):
    model.eval()

    correct = 0
    total = 0
    total_spikes = 0
    total_images = 0

    start_time = time.time()

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs, spike_count = model(images, num_steps)

            _, predicted = outputs.max(1)

            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

            total_spikes += spike_count
            total_images += labels.size(0)

    end_time = time.time()

    accuracy = 100.0 * correct / total
    avg_spikes_per_image = total_spikes / total_images
    avg_inference_time_ms = ((end_time - start_time) / total_images) * 1000
    avg_firing_rate = total_spikes / (total_images * num_steps)

    return accuracy, avg_spikes_per_image, avg_inference_time_ms, avg_firing_rate


# ----------------------------
# 5. Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--weights", type=str, default="cnn_cifar10.pth")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_steps", type=int, default=25)
    parser.add_argument("--beta", type=float, default=0.95)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    test_loader = get_test_loader(args.batch_size)

    cnn = CIFAR10CNN().to(device)
    cnn.load_state_dict(torch.load(args.weights, map_location=device))
    cnn.eval()

    snn_model = ConvertedSNN(cnn, beta=args.beta).to(device)

    accuracy, avg_spikes, avg_time_ms, avg_firing_rate = evaluate_snn(
        snn_model,
        test_loader,
        device,
        args.num_steps
    )

    print("\nConverted ANN-to-SNN Results")
    print("----------------------------")
    print(f"Time steps T: {args.num_steps}")
    print(f"Test accuracy: {accuracy:.2f}%")
    print(f"Average spikes per image: {avg_spikes:.2f}")
    print(f"Average inference time: {avg_time_ms:.4f} ms/image")
    print(f"Average firing rate proxy: {avg_firing_rate:.4f} spikes/image/timestep")


if __name__ == "__main__":
    main()