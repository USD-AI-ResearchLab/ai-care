import torch
from torch import optim
from torch.utils.data import DataLoader
from utils.energy_meter import EnergyMeter


def train_model(model, dataset, cfg):
    device = torch.device(
        cfg["device"]
        if cfg["device"] != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model.to(device)

    loader = DataLoader(
        dataset,
        batch_size=cfg["batch_size"],
        shuffle=True
    )

    opt = optim.Adam(model.parameters(), lr=cfg["learning_rate"])
    criterion = torch.nn.CrossEntropyLoss()

    # Energy-only measurement
    meter = EnergyMeter(device=device.type)

    meter.start()
    model.train()

    for _ in range(cfg["epochs"]):
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            opt.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            opt.step()

    meter.stop()

    train_kwh = meter.report()
    return model, train_kwh
