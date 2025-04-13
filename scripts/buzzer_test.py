import RPi.GPIO as GPIO
import time

BUZZER_PIN = 17  # номер GPIO, не физический пин

GPIO.setmode(GPIO.BCM)
GPIO.setup(BUZZER_PIN, GPIO.OUT)

# Писк 0.2 сек
pwm = GPIO.PWM(BUZZER_PIN, 1000)
pwm.start(50)
time.sleep(0.2)

pwm.stop()

GPIO.cleanup()
