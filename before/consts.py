from utils import dB2pow
from fluid_antenna_system import FluidAntennaSystem
import numpy as np

np.random.seed(1)  # fix the random parameters

# Initialization
NUM_OF_USERS: int = 5
NUM_OF_ANTENNAS: int = 16
NUM_OF_SELECTED_ANTENNAS: int = 10
SNAPSHOT: int = 10

CHANNEL_NOISE: float = dB2pow(0)
SENSING_NOISE: float = dB2pow(10)
REFLECTION_COEFFICIENT: complex = 1.5 + 1j*1.5
QOS_THRESHOLD: float = dB2pow(12)
EVE_SENSING_THRESHOLD: float = dB2pow(12)
CRB_THRESHOLD: float = dB2pow(0)

DOA: np.ndarray = np.array([[-np.pi / 6, 0]])
indices: np.ndarray = np.arange(0, (NUM_OF_ANTENNAS - 1) + 1).reshape(-1, 1)
STEERING_VECTOR: np.ndarray = np.exp(-1j * np.pi * indices * np.sin(DOA))
DIFF_STEERING_VECTOR: np.ndarray = -1j * np.pi * indices * np.cos(DOA) * STEERING_VECTOR


fas = FluidAntennaSystem(num_of_yaxis_antennas=NUM_OF_ANTENNAS,
                         num_of_users=NUM_OF_USERS,
                         noise_variance=CHANNEL_NOISE)
CHANNEL = fas.get_channel()


Parameters = {
    "num_of_antennas": NUM_OF_ANTENNAS,
    "num_of_users": NUM_OF_USERS,
    "num_of_selected_antennas": NUM_OF_SELECTED_ANTENNAS,
    "snapshot": SNAPSHOT,
    "qos_threshold": QOS_THRESHOLD,
    "eve_sensing_threshold": EVE_SENSING_THRESHOLD,
    "crb_threshold": CRB_THRESHOLD,
    "reflection_coefficient": REFLECTION_COEFFICIENT,
    "channel_noise": CHANNEL_NOISE,
    "sensing_noise": SENSING_NOISE,
    "doa": DOA,
    "steering_vector": STEERING_VECTOR,
    "diff_steering_vector": DIFF_STEERING_VECTOR,
    "channel": CHANNEL
}
