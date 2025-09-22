from flask import Flask, jsonify, request
import RPi.GPIO as GPIO
import time

app = Flask(__name__)

GPIO.setwarnings(False)

# Define GPIO pins for controlling the motor.
IN1 = 17
IN2 = 27
EN1 = 22

# Set GPIO mode and configure the pins as outputs.
GPIO.setmode(GPIO.BCM)
GPIO.setup(IN1, GPIO.OUT)
GPIO.setup(IN2, GPIO.OUT)
GPIO.setup(EN1, GPIO.OUT)

# Set up Pulse Width Modulation (PWM) on EN1 to control motor speed.
pwm = GPIO.PWM(EN1, 1000)
pwm.start(100)

# Function to move the motor forward (close the door).
def move_forward():
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)
    print("Forward")

# Function to move the motor backward (open the door).
def move_backward():
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.HIGH)
    print("Backward")

# Function to stop the motor.
def stop_actuator():
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.LOW)
    print("Stop")

# Function to adjust the motor speed by changing the PWM duty cycle.
def set_speed(duty_cycle):
    pwm.ChangeDutyCycle(duty_cycle)
    print(f"Velocity: {duty_cycle}%")

# Function to define a route for toggling the door open or closed.
@app.route('/toggle_door', methods=['POST'])
def toggle_door():
    data = request.json # Get the JSON data from the POST request.
    is_door_locked = data['isDoorLocked'] # Extract the "isDoorLocked" flag from the request.

    # If the door is locked, move the actuator backward to unlock the door;
    # if the door is unlocked, move the actuator forward to lock the door.

    if is_door_locked == False:
        move_backward()  # Open the door
        time.sleep(2)
    else:
        move_forward()  # Close the door
        time.sleep(2)
        stop_actuator()

    return jsonify({'message': 'Door toggled successfully'}), 200

 # Function to define a route for toggling the gate open and then immediately closing it.
@app.route('/toggle_gate', methods=['POST'])
def toggle_gate(): # Press the gate button
    # The gate, unlike the door, is opened and closed by the same button on the intercom. Pressing the
    # button opens the gate, and pressing the same button again closes it. Therefore, the actuator should
    # behave as if it were a hand pressing the button. This is simulated by moving forward, then stopping and immediately moving backward.
    move_forward()
    time.sleep(2)
    move_backward()

    return jsonify({'message': 'Gate toggled successfully'}), 200

# Function to define a route to get the current status of the door.
@app.route('/get_status', methods=['GET'])
def get_status():
    door_status = False
    return jsonify({
        'isDoorLocked': door_status
    })

@app.route('/')
def index():
    return jsonify({
        "message": "Welcome to gate and door control!"
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
