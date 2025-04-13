import RPi.GPIO as GPIO
import time

BUZZER_PIN = 18  # номер GPIO, не физический пин

GPIO.setmode(GPIO.BCM)
GPIO.setup(BUZZER_PIN, GPIO.OUT)

# Писк 0.2 сек
GPIO.output(BUZZER_PIN, GPIO.HIGH)
time.sleep(0.2)
GPIO.output(BUZZER_PIN, GPIO.LOW)

GPIO.cleanup()
