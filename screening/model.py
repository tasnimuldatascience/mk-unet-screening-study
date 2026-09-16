import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'vendor/MK-UNet'))
from mkunet_network import MK_UNet

def build_model():
    return MK_UNet(num_classes=1, in_channels=3, channels=[16, 32, 64, 96, 160], kernel_sizes=[1, 3, 5])
