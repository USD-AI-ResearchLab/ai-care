import torch
from utils.energy_meter import EnergyMeter
from utils.metrics import accuracy


def evaluate_model(model, dataset, cfg):
    device = next(model.parameters()).device
    loader = torch.utils.data.DataLoader(dataset, batch_size=128)

    meter = EnergyMeter(device=device.type)

    meter.start()
    acc = accuracy(model, loader, device)
    meter.stop()

    infer_kwh = meter.report()
    return acc, infer_kwh
