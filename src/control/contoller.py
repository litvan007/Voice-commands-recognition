import smbus
import time
import math
import logging
from settings import Settings

logger = logging.getLogger(__name__)

# ============================================================================
# Raspi PCA9685 16-Channel PWM Servo Driver
# ============================================================================
class PCA9685:
    # Registers
    __SUBADR1 = 0x02
    __SUBADR2 = 0x03
    __SUBADR3 = 0x04
    __MODE1 = 0x00
    __PRESCALE = 0xFE
    __LED0_ON_L = 0x06
    __LED0_ON_H = 0x07
    __LED0_OFF_L = 0x08
    __LED0_OFF_H = 0x09
    __ALLLED_ON_L = 0xFA
    __ALLLED_ON_H = 0xFB
    __ALLLED_OFF_L = 0xFC
    __ALLLED_OFF_H = 0xFD

    def __init__(self, address=0x40):
        self.bus = smbus.SMBus(1)
        self.address = address
        logger.info(f"Инициализация PCA9685 по адресу 0x{self.address:X}")
        self.write(self.__MODE1, 0x00)

    def write(self, reg, value):
        self.bus.write_byte_data(self.address, reg, value)
        logger.debug(f"I2C: Write 0x{value:02X} → reg 0x{reg:02X}")

    def read(self, reg):
        result = self.bus.read_byte_data(self.address, reg)
        logger.debug(f"I2C: Read 0x{result:02X} ← reg 0x{reg:02X}")
        return result

    def setPWMFreq(self, freq):
        prescaleval = 25000000.0 / 4096.0 / float(freq) - 1.0
        prescale = math.floor(prescaleval + 0.5)

        logger.info(f"Установка частоты PWM: {freq} Гц (prescale={prescale})")

        oldmode = self.read(self.__MODE1)
        newmode = (oldmode & 0x7F) | 0x10  # sleep
        self.write(self.__MODE1, newmode)
        self.write(self.__PRESCALE, int(prescale))
        self.write(self.__MODE1, oldmode)
        time.sleep(0.005)
        self.write(self.__MODE1, oldmode | 0x80)

    def setPWM(self, channel, on, off):
        self.write(self.__LED0_ON_L + 4 * channel, on & 0xFF)
        self.write(self.__LED0_ON_H + 4 * channel, on >> 8)
        self.write(self.__LED0_OFF_L + 4 * channel, off & 0xFF)
        self.write(self.__LED0_OFF_H + 4 * channel, off >> 8)
        logger.debug(f"PWM канал {channel}: ON={on}, OFF={off}")

    def setServoPulse(self, channel, pulse):
        pwm_val = int(pulse * 4096 / 20000)  # 50 Гц → период 20000 мкс
        self.setPWM(channel, 0, pwm_val)
        logger.debug(f"Серво канал {channel}: импульс {pulse} мкс → PWM {pwm_val}")


class ArmController:
    """
    Контроллер роборуки через PCA9685.
    """

    def __init__(self, driver: PCA9685):
        self.driver = driver
        self.driver.setPWMFreq(50)
        logger.info("Контроллер руки инициализирован (PWM 50 Гц)")

    def set_joint(self, channel: int, pulse: int):
        """
        Управляет одним каналом сервопривода.
        """
        logger.debug(f"Движение: канал {channel} → импульс {pulse}")
        self.driver.setServoPulse(channel, pulse)

    def demo_motion(self):
        """
        Демонстрационное движение на одном канале.
        """
        logger.info("Запуск demo-motion")
        for i in range(0, 4000, 200):
            self.set_joint(2, i)
            time.sleep(0.02)

    def execute_command(self, command: str, settings: Settings):
        motion = settings.arm_commands.commands.get(command)
        if motion is None:
            logger.warning(f"Команда '{command}' не найдена в карте движений")
            return

        logger.info(f"Выполнение команды: {command}")
        for channel, pulse in motion.items():
            logger.debug(f" → Канал {channel}: {pulse}")
            self.set_joint(channel, pulse)
            time.sleep(0.02)
