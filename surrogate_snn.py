import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

import snntorch as snn
from snntorch import surrogate

#update to latest snntorch version
#my updates
# ----------------------------
# Data
# ----------------------------

def get_loaders(batch_size=128):
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_set = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=train_transform
    )

    test_set = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=test_transform
    )

    train_size = int(0.9 * len(train_set))
    val_size = len(train_set) - train_size

    train_set, val_set = torch.utils.data.random_split(
        train_set,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=2
    )

    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=batch_size, shuffle=False, num_workers=2
    )

    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=2
    )

    return train_loader, val_loader, test_loader


# ----------------------------
# Surrogate SNN Model
# ----------------------------

class SurrogateSNN(nn.Module):
    def __init__(self, beta=0.95):
        super().__init__()

        spike_grad = surrogate.fast_sigmoid(slope=25)

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.pool1 = nn.MaxPool2d(2)

        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.lif3 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.pool2 = nn.MaxPool2d(2)

        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.lif4 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.pool3 = nn.MaxPool2d(2)

        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.lif5 = snn.Leaky(beta=beta, spike_grad=spike_grad)

        self.fc2 = nn.Linear(512, 10)
        self.lif_out = snn.Leaky(beta=beta, spike_grad=spike_grad)

    def forward(self, x, num_steps=25):
        batch_size = x.size(0)

        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        mem4 = self.lif4.init_leaky()
        mem5 = self.lif5.init_leaky()
        mem_out = self.lif_out.init_leaky()

        spk_out_rec = []
        mem_out_rec = []

        spike_counts = {
            "lif1": 0,
            "lif2": 0,
            "lif3": 0,
            "lif4": 0,
            "lif5": 0,
            "lif_out": 0,
        }

        neuron_counts = {
            "lif1": batch_size * 64 * 32 * 32 * num_steps,
            "lif2": batch_size * 128 * 32 * 32 * num_steps,
            "lif3": batch_size * 128 * 16 * 16 * num_steps,
            "lif4": batch_size * 256 * 8 * 8 * num_steps,
            "lif5": batch_size * 512 * num_steps,
            "lif_out": batch_size * 10 * num_steps,
        }

        for _ in range(num_steps):
            cur1 = self.bn1(self.conv1(x))
            spk1, mem1 = self.lif1(cur1, mem1)
            spike_counts["lif1"] += spk1.sum().item()

            cur2 = self.bn2(self.conv2(spk1))
            spk2, mem2 = self.lif2(cur2, mem2)
            spike_counts["lif2"] += spk2.sum().item()
            spk2 = self.pool1(spk2)

            cur3 = self.bn3(self.conv3(spk2))
            spk3, mem3 = self.lif3(cur3, mem3)
            spike_counts["lif3"] += spk3.sum().item()
            spk3 = self.pool2(spk3)

            cur4 = self.bn4(self.conv4(spk3))
            spk4, mem4 = self.lif4(cur4, mem4)
            spike_counts["lif4"] += spk4.sum().item()
            spk4 = self.pool3(spk4)

            flat = spk4.view(batch_size, -1)

            cur5 = self.fc1(flat)
            spk5, mem5 = self.lif5(cur5, mem5)
            spike_counts["lif5"] += spk5.sum().item()

            cur_out = self.fc2(spk5)
            spk_out, mem_out = self.lif_out(cur_out, mem_out)
            spike_counts["lif_out"] += spk_out.sum().item()

            spk_out_rec.append(spk_out)
            mem_out_rec.append(mem_out)

        spk_out_rec = torch.stack(spk_out_rec)
        mem_out_rec = torch.stack(mem_out_rec)

        return spk_out_rec, mem_out_rec, spike_counts, neuron_counts


# ----------------------------
# Train / Evaluate
# ----------------------------

def train_one_epoch(model, loader, optimizer, loss_fn, device, num_steps):
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        spk_rec, mem_rec, _, _ = model(images, num_steps=num_steps)

        # Sum output membrane potentials over time
        logits = mem_rec.sum(dim=0)

        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, 100.0 * correct / total


@torch.no_grad()
def evaluate(model, loader, loss_fn, device, num_steps):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    total_spikes = {
        "lif1": 0,
        "lif2": 0,
        "lif3": 0,
        "lif4": 0,
        "lif5": 0,
        "lif_out": 0,
    }

    total_neurons = {
        "lif1": 0,
        "lif2": 0,
        "lif3": 0,
        "lif4": 0,
        "lif5": 0,
        "lif_out": 0,
    }

    start = time.time()

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        spk_rec, mem_rec, spike_counts, neuron_counts = model(images, num_steps=num_steps)

        logits = mem_rec.sum(dim=0)
        loss = loss_fn(logits, labels)

        total_loss += loss.item() * images.size(0)

        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        for k in total_spikes:
            total_spikes[k] += spike_counts[k]
            total_neurons[k] += neuron_counts[k]

    end = time.time()

    avg_loss = total_loss / total
    accuracy = 100.0 * correct / total
    ms_per_image = ((end - start) / total) * 1000

    firing_rates = {
        k: total_spikes[k] / total_neurons[k]
        for k in total_spikes
    }

    total_spike_count = sum(total_spikes.values())
    total_neuron_count = sum(total_neurons.values())

    avg_spikes_per_image = total_spike_count / total
    avg_firing_rate = total_spike_count / total_neuron_count
    sparsity = 100.0 * (1.0 - avg_firing_rate)

    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "ms_per_image": ms_per_image,
        "avg_spikes_per_image": avg_spikes_per_image,
        "avg_firing_rate": avg_firing_rate,
        "sparsity": sparsity,
        "layer_firing_rates": firing_rates,
    }


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ----------------------------
# Main
# ----------------------------

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_steps", type=int, default=25)
    parser.add_argument("--beta", type=float, default=0.95)
    parser.add_argument("--save_path", type=str, default="surrogate_snn.pth")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader = get_loaders(args.batch_size)

    model = SurrogateSNN(beta=args.beta).to(device)

    print(f"Trainable parameters: {count_parameters(model):,}")

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            optimizer,
            loss_fn,
            device,
            args.num_steps,
        )

        val_metrics = evaluate(
            model,
            val_loader,
            loss_fn,
            device,
            args.num_steps,
        )

        val_loss = val_metrics["loss"]
        val_acc = val_metrics["accuracy"]

        print(
            f"Epoch [{epoch}/{args.epochs}] "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.2f}%"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), args.save_path)
            print(f"Saved best model to {args.save_path}")

    print("\nLoading best model for final test...")
    model.load_state_dict(torch.load(args.save_path, map_location=device))

    test_metrics = evaluate(
        model,
        test_loader,
        loss_fn,
        device,
        args.num_steps,
    )

    print("\n========== FINAL TEST RESULTS ==========")
    print(f"Test Accuracy: {test_metrics['accuracy']:.2f}%")
    print(f"Test Loss: {test_metrics['loss']:.4f}")
    print(f"Inference Time: {test_metrics['ms_per_image']:.4f} ms/image")
    print(f"Time Steps T: {args.num_steps}")
    print(f"Average Spikes Per Image: {test_metrics['avg_spikes_per_image']:.2f}")
    print(f"Average Firing Rate: {test_metrics['avg_firing_rate']:.6f}")
    print(f"Sparsity: {test_metrics['sparsity']:.2f}%")
    print(f"Model Size: {count_parameters(model):,} parameters")

    print("\nLayer-wise firing rates:")
    for layer, rate in test_metrics["layer_firing_rates"].items():
        print(f"{layer}: {rate:.6f}")

    print("\nEfficiency notes for report:")
    print("- This SNN uses LIF neurons trained directly with BPTT.")
    print("- Surrogate gradient: fast sigmoid surrogate.")
    print("- Latency is proportional to T time steps.")
    print("- Sparsity is estimated as percentage of inactive neurons.")
    print("- Activation memory increases approximately linearly with T during BPTT.")


if __name__ == "__main__":
    main()