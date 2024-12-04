from utils import dBm2pow, dB2pow
from fluidAntennaSystem import FluidAntennaSystem
import numpy as np

np.random.seed(1)  # fix the random parameters

# Initialization
NUM_OF_USERS = 5
NUM_OF_ANTENNAS = 16
NUM_OF_SELECTED_ANTENNAS = 16
CHANNEL_NOISE = dB2pow(-10)
QOS_THRESHOLD = dB2pow(12)
DOA = np.pi / 6
indices = np.arange(0, (NUM_OF_ANTENNAS - 1) + 1).reshape(-1, 1)
STEERING_VECTOR = np.exp(-1j * np.pi * indices * np.sin(DOA))

fas = FluidAntennaSystem(numOfYaxisAntennas=NUM_OF_ANTENNAS, numOfUsers=NUM_OF_USERS)
CHANNEL, _, _, _ = fas.get_channel()

Parameters = {"numOfAntennas": NUM_OF_ANTENNAS,
              "numOfUsers": NUM_OF_USERS,
              "numOfSelectedAntennas": NUM_OF_SELECTED_ANTENNAS,
              "qosThreshold": QOS_THRESHOLD,
              "channelNoise": CHANNEL_NOISE,
              "doa": DOA,
              "steeringVector": STEERING_VECTOR,
              "channel": CHANNEL}
