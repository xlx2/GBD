from utils import dB2pow
from fluid_antenna_system import FluidAntennaSystem
import numpy as np

np.random.seed(1)  # fix the random parameters

# Initialization
NUM_OF_USERS: int = 5
NUM_OF_ANTENNAS: int = 16
NUM_OF_SELECTED_ANTENNAS: int = 10
CHANNEL_NOISE: float = dB2pow(0)
SENSING_NOISE: float = dB2pow(10)
REFLECTION_COEFFICIENT: float = dB2pow(0)
QOS_THRESHOLD: float = dB2pow(12)
EVE_SENSING_THRESHOLD: float = dB2pow(12)
CRB_THRESHOLD: float = dB2pow(0)
DOA: np.ndarray = np.array([[np.pi / 6]])
indices: np.ndarray = np.arange(0, (NUM_OF_ANTENNAS - 1) + 1).reshape(-1, 1)
STEERING_VECTOR: np.ndarray = np.exp(-1j * np.pi * indices * np.sin(DOA))
DIFF_STEERING_VECTOR: np.ndarray = -1j * np.pi * indices * np.cos(DOA) * STEERING_VECTOR

fas = FluidAntennaSystem(num_of_yaxis_antennas=NUM_OF_ANTENNAS, num_of_users=NUM_OF_USERS,noise_variance=CHANNEL_NOISE)
CHANNEL = fas.get_channel()
print(CHANNEL)

Parameters = {
    "numOfAntennas": NUM_OF_ANTENNAS,
    "num_of_users": NUM_OF_USERS,
    "numOfSelectedAntennas": NUM_OF_SELECTED_ANTENNAS,
    "qosThreshold": QOS_THRESHOLD,
    "eveSensingThreshold": EVE_SENSING_THRESHOLD,
    "crbThreshold": CRB_THRESHOLD,
    "reflectionCoefficient": REFLECTION_COEFFICIENT,
    "channelNoise": CHANNEL_NOISE,
    "sensingNoise": SENSING_NOISE,
    "doa": DOA,
    "steeringVector": STEERING_VECTOR,
    "diffSteeringVector": DIFF_STEERING_VECTOR,
    "channel": CHANNEL
}
