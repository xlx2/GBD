from utils import dB2pow
from fluid_antenna_system import FluidAntennaSystem
import numpy as np

np.random.seed(100)  # fix the random parameters

# Initialization
NUM_OF_USERS = 5
NUM_OF_ANTENNAS = 16
NUM_OF_SELECTED_ANTENNAS = 4
CHANNEL_NOISE = dB2pow(-10)
SENSING_NOISE = dB2pow(-10)
REFLECTION_COEFFICIENT = dB2pow(-10)
QOS_THRESHOLD = dB2pow(12)
SENSING_UPPER_THRESHOLD = dB2pow(12)
SENSING_LOWER_THRESHOLD = dB2pow(8)
DOA = np.array([[np.pi / 6, 0, -np.pi/6]])
indices = np.arange(0, (NUM_OF_ANTENNAS - 1) + 1).reshape(-1, 1)
STEERING_VECTOR = np.exp(-1j * np.pi * indices * np.sin(DOA))

fas = FluidAntennaSystem(numOfYaxisAntennas=NUM_OF_ANTENNAS, numOfUsers=NUM_OF_USERS)
CHANNEL, _, _, _ = fas.get_channel()

Parameters = {"numOfAntennas": NUM_OF_ANTENNAS,
              "numOfUsers": NUM_OF_USERS,
              "numOfSelectedAntennas": NUM_OF_SELECTED_ANTENNAS,
              "qosThreshold": QOS_THRESHOLD,
              "sensingUpperThreshold": SENSING_UPPER_THRESHOLD,
              "sensingLowerThreshold": SENSING_LOWER_THRESHOLD,
              "reflectionCoefficient": REFLECTION_COEFFICIENT,
              "channelNoise": CHANNEL_NOISE,
              "sensingNoise": SENSING_NOISE,
              "doa": DOA,
              "steeringVector": STEERING_VECTOR,
              "channel": CHANNEL}
