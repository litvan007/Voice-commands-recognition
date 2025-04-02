import smbus
import time
import math
import logging
from dataclasses import dataclass
from settings import Settings

import yaml
import os

logger = logging.getLogger(__name__)
SERVO_STATE_PATH = "./configs/servo_state.yaml"

# ============================================================================
# PCA9685 Constants
# ============================================================================
PCA9685_ADDRESS = 0x40

class PCA9685:
    MODE1 = 0x00
    MODE2 = 0x01
    PRESCALE = 0xFE
    LED0_ON_L = 0x06
    LED0_OFF_L = 0x08
    ALL_LED_ON_L = 0xFA
    ALL_LED_OFF_L = 0xFC
    RESTART = 0x80
    SLEEP = 0x10
    ALLCALL = 0x01
    OUTDRV = 0x04

    def __init__(self, i2cBus, address=PCA9685_ADDRESS):
        self.i2cBus = i2cBus
        self.address = address
        self.begin()

    def begin(self):
        self.set_all_pwm(0, 0)
        self.i2cBus.write_byte_data(self.address, self.MODE2, self.OUTDRV)
        self.i2cBus.write_byte_data(self.address, self.MODE1, self.ALLCALL)
        time.sleep(0.005)
        mode1 = self.i2cBus.read_byte_data(self.address, self.MODE1)
        mode1 &= ~self.SLEEP
        self.i2cBus.write_byte_data(self.address, self.MODE1, mode1)
        time.sleep(0.005)

    def reset(self):
        self.i2cBus.write_byte_data(self.address, self.MODE1, self.RESTART)
        time.sleep(0.01)

    def set_pwm(self, channel, on, off):
        self.i2cBus.write_byte_data(self.address, self.LED0_ON_L + 4 * channel, on & 0xFF)
        self.i2cBus.write_byte_data(self.address, self.LED0_ON_L + 4 * channel + 1, on >> 8)
        self.i2cBus.write_byte_data(self.address, self.LED0_OFF_L + 4 * channel, off & 0xFF)
        self.i2cBus.write_byte_data(self.address, self.LED0_OFF_L + 4 * channel + 1, off >> 8)

    def set_all_pwm(self, on, off):
        self.i2cBus.write_byte_data(self.address, self.ALL_LED_ON_L, on & 0xFF)
        self.i2cBus.write_byte_data(self.address, self.ALL_LED_ON_L + 1, on >> 8)
        self.i2cBus.write_byte_data(self.address, self.ALL_LED_OFF_L, off & 0xFF)
        self.i2cBus.write_byte_data(self.address, self.ALL_LED_OFF_L + 1, off >> 8)

    def set_pwm_freq(self, freq_hz):
        prescaleval = 25000000.0 / (4096.0 * freq_hz) - 1.0
        prescale = int(math.floor(prescaleval + 0.5))
        oldmode = self.i2cBus.read_byte_data(self.address, self.MODE1)
        newmode = (oldmode & 0x7F) | self.SLEEP
        self.i2cBus.write_byte_data(self.address, self.MODE1, newmode)
        self.i2cBus.write_byte_data(self.address, self.PRESCALE, prescale)
        self.i2cBus.write_byte_data(self.address, self.MODE1, oldmode)
        time.sleep(0.005)
        self.i2cBus.write_byte_data(self.address, self.MODE1, oldmode | self.RESTART)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.reset()

# ============================================================================
# Servo PCA9685 Implementation
# ============================================================================

@dataclass
class ServoState:
    servo: 'ServoPCA9685'
    prev_angle: int = 0

class ServoPCA9685:
    def __init__(self, pca9685, channel, servo_min=130, servo_max=510):
        self.pca9685 = pca9685
        self.channel = channel
        self.servo_min = servo_min
        self.servo_max = servo_max
        self.set_pwm_freq(50)
        self.set_pulse((servo_min + servo_max) // 2)

    def set_pwm_freq(self, freq=50):
        self.pca9685.set_pwm_freq(freq)

    def set_angle(self, angle):
        pulse = self._map(angle, 0, 180, self.servo_min, self.servo_max)
        self.set_pulse(pulse)

    def move_smoothly(self, from_angle, to_angle, step=1, delay=0.01):
        step = step if to_angle > from_angle else -step
        for angle in range(from_angle, to_angle + step, step):
            self.set_angle(angle)
            time.sleep(delay)

    def set_pulse(self, pulse):
        pulse = max(min(pulse, self.servo_max), self.servo_min)
        self.pca9685.set_pwm(self.channel, 0, pulse)

    def disable(self):
        self.pca9685.set_pwm(self.channel, 0, 0)

    @staticmethod
    def _map(x, in_min, in_max, out_min, out_max):
        return int((x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min)

# ============================================================================
# Arm Controller
# ============================================================================

class ArmController:
    def __init__(self, driver: PCA9685, channels=(0, 1, 2, 3)):
        self.servo_map = {
            ch: ServoState(ServoPCA9685(driver, ch))
            for ch in channels
        }
        self.disable_all_servos()
        logger.info("Контроллер руки инициализирован (PWM 50 Гц)")

    def save_servo_positions(self, filepath: str = SERVO_STATE_PATH):
        """
        Сохраняет текущие углы всех сервоприводов в YAML-файл.
        """
        positions = {
            channel: servo_state.prev_angle
            for channel, servo_state in self.servo_map.items()
        }
        with open(filepath, "w") as f:
            yaml.dump({"servo_positions": positions}, f)
        logger.info("Позиции сервоприводов сохранены.")

    def load_servo_positions(self, filepath: str = SERVO_STATE_PATH):
        """
        Загружает углы из YAML-файла и устанавливает их на сервоприводы.
        """
        if not os.path.exists(filepath):
            logger.warning("Файл сохранения позиций не найден.")
            return

        with open(filepath, "r") as f:
            data = yaml.safe_load(f)
            positions = data.get("servo_positions", {})

        for channel, angle in positions.items():
            channel = int(channel)
            if channel in self.servo_map:
                self.servo_map[channel].servo.set_angle(angle)
                self.servo_map[channel].prev_angle = angle
                time.sleep(0.01)

        self.disable_all_servos()

        logger.info("Позиции сервоприводов восстановлены.")


    def execute_command(self, command: str, settings: Settings):
        motion = settings.arm_commands.commands.get(command)
        if not motion:
            logger.warning(f"Неизвестная команда '{command}'")
            return

        logger.info(f"Выполнение команды: {command}")

        for channel, final_angle in motion.items():
            servo_state = self.servo_map.get(channel)
            if servo_state is None:
                logger.warning(f"Неизвестный канал: {channel}")
                continue
            servo_state.servo.move_smoothly(servo_state.prev_angle, final_angle)
            servo_state.prev_angle = final_angle

        self.disable_all_servos()

    def disable_all_servos(self):
        for servo_state in self.servo_map.values():
            servo_state.servo.disable()

