import requests
from datetime import datetime
from ml.solar_model import SolarMLSystem

API_KEY = ""
LOCATION = "Tehran"  # یا هر شهر

# نمونه گرفتن داده‌های آب و هوا
url = f"http://api.weatherapi.com/v1/current.json?key={API_KEY}&q={LOCATION}&aqi=no"
resp = requests.get(url)
data = resp.json()["current"]

# آماده کردن ورودی مدل
now = datetime.now()
input_data = {
    "Hour": now.hour,
    "Month": now.month,
    "TempOut": data["temp_c"],
    "OutHum": data["humidity"],
    "WindSpeed": data["wind_kph"] / 3.6,  # تبدیل km/h به m/s
    "Bar": data["pressure_mb"]
}

print(input_data)

# استفاده از مدل
ml_system = SolarMLSystem()
ml_system.train("uploads/solar_dataset.csv")  # یا مدل آموزش دیده شما

result = ml_system.full_analysis(input_data)
print(result)
