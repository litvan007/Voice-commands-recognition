from adafruit_pca9685 import PCA9685
from board import SCL, SDA
import busio
import time

# I2C setup
i2c = busio.I2C(SCL, SDA)
pca = PCA9685(i2c)
pca.frequency = 1000  # Частота в Гц

channel = 15

# Включить звук (50% скважность)
pca.channels[channel].duty_cycle = 0x7FFF  # Половина максимума

time.sleep(0.1)

# Выключить звук
pca.channels[channel].duty_cycle = 0
