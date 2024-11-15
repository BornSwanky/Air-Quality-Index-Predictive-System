import tkinter as tk
from tkinter import messagebox, PhotoImage
import joblib
import numpy as np

# Load the trained model and scaler
model = joblib.load('air_quality_model.pkl')
scaler = joblib.load('scaler.pkl')

def interpret_aqi(aqi):
    if aqi <= 50:
        return 'Good'
    elif aqi <= 100:
        return 'Moderate'
    elif aqi <= 150:
        return 'Unhealthy for Sensitive Groups'
    elif aqi <= 200:
        return 'Unhealthy'
    elif aqi <= 300:
        return 'Very Unhealthy'
    else:
        return 'Hazardous'

# Predictions
def predict_aqi(inputs):
    inputs_scaled = scaler.transform(np.array(inputs).reshape(1, -1))
    prediction = model.predict(inputs_scaled)
    return prediction[0]

# Design
def create_labeled_entry(canvas, text, x, y):
    label = tk.Label(canvas, text=text, bg='white', font=('Arial', 10))
    canvas.create_window(x, y-20, window=label, width=180)
    entry = tk.Entry(canvas, fg='black', bg='white', borderwidth=0)
    canvas.create_window(x, y, window=entry, width=180)
    return entry

# For users input and prediction display
def run_app():
    def on_predict():
        try:
            inputs = [
                float(entry_year.get()),
                float(entry_month.get()),
                float(entry_day.get()),
                float(entry_hour.get()),
                float(entry_temperature.get()),
                float(entry_humidity.get()),
                float(entry_wind_speed.get()),
                float(entry_noise_level.get()),
                float(entry_precipitation.get()),
                float(entry_solar_radiation.get())
            ]
        except ValueError:
            messagebox.showerror('Input Error', 'Please ensure all inputs are numerical.')
            return

        try:
            aqi = predict_aqi(inputs)
            aqi_category = interpret_aqi(aqi)
            messagebox.showinfo('Prediction', f'Predicted Air Quality Index: {aqi:.2f}\nCategory: {aqi_category}')
        except Exception as e:
            messagebox.showerror('Prediction Error', f'An error occurred: {e}')

    window = tk.Tk()
    window.title('AQI Prediction System')

    # Background image
    background_image = PhotoImage(file='background.png')
    canvas = tk.Canvas(window, width=350, height=600)
    canvas.pack(fill="both", expand=True)
    canvas.create_image(0, 0, image=background_image, anchor='nw')
    #Title
    title_font = ('Arial', 18, 'bold')
    canvas.create_text(175, 30, text="Air Quality Index", font=title_font, fill='black')

    # Parameters for user entry 
    entry_year = create_labeled_entry(canvas, 'Year', 175, 80)
    entry_month = create_labeled_entry(canvas, 'Month', 175, 130)
    entry_day = create_labeled_entry(canvas, 'Day', 175, 180)
    entry_hour = create_labeled_entry(canvas, 'Hour', 175, 230)
    entry_temperature = create_labeled_entry(canvas, 'Temperature', 175, 280)
    entry_humidity = create_labeled_entry(canvas, 'Humidity', 175, 330)
    entry_wind_speed = create_labeled_entry(canvas, 'Wind Speed', 175, 380)
    entry_noise_level = create_labeled_entry(canvas, 'Noise Level', 175, 430)
    entry_precipitation = create_labeled_entry(canvas, 'Precipitation', 175, 480)
    entry_solar_radiation = create_labeled_entry(canvas, 'Solar Radiation', 175, 530)

    # Generate button
    predict_button = tk.Button(window, text='Generate', command=on_predict, bg='black', fg='white')
    canvas.create_window(175, 580, window=predict_button, width=90)

    # Proportion adjustment to fit everything in the background
    window_width = 350
    window_height = 600
    screen_width = window.winfo_screenwidth()
    screen_height = window.winfo_screenheight()
    x_coordinate = int((screen_width / 2) - (window_width / 2))
    y_coordinate = int((screen_height / 2) - (window_height / 2))
    window.geometry(f"{window_width}x{window_height}+{x_coordinate}+{y_coordinate}")

    window.mainloop()

# Run the app
run_app()

