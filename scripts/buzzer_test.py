import RPi.GPIO as GPIO
import time

BUZZER_PIN = 18
NOTES = {
    'C4': 261,
    'D4': 294,
    'E4': 329,
    'F4': 349,
    'G4': 392,
    'A4': 440,
    'B4': 493,
    'C5': 523
}

GPIO.setmode(GPIO.BCM)
GPIO.setup(BUZZER_PIN, GPIO.OUT)
pwm = GPIO.PWM(BUZZER_PIN, 440)  # начальная частота
pwm.start(50)

for note, freq in NOTES.items():
    pwm.ChangeFrequency(freq)
    time.sleep(0.3)

pwm.stop()
GPIO.cleanup()
