import { useQuery } from '@tanstack/react-query'
import { Card, CardContent } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Progress } from '@/components/ui/progress'
import { Sun, CloudRain, Cloud, AlertTriangle, CheckCircle, XCircle } from 'lucide-react'

const WEATHER_API_KEY = '2694ef9b0fb54f8b810221511251312'

async function fetchWeather() {
  const res = await fetch(
    `https://api.weatherapi.com/v1/current.json?key=${WEATHER_API_KEY}&q=Tabriz`
  )
  if (!res.ok) throw new Error('Weather fetch failed')
  return res.json()
}

async function fetchPrediction() {
  const res = await fetch('http://localhost:5000/full_analysis', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      Hour: 14,
      Month: 12,
      TempOut: 10,
      OutHum: 80,
      WindSpeed: 5,
      Bar: 1012
    })
  })
  if (!res.ok) throw new Error('Prediction fetch failed')
  return res.json()
}

export default function SolarDashboard() {
  const weatherQuery = useQuery({ queryKey: ['weather'], queryFn: fetchWeather })
  const predictionQuery = useQuery({ queryKey: ['prediction'], queryFn: fetchPrediction })

  if (weatherQuery.isLoading || predictionQuery.isLoading) {
    return (
      <div className="flex items-center justify-center min-h-screen bg-black">
        <span className="loading loading-spinner loading-lg text-warning"></span>
      </div>
    )
  }

  if (weatherQuery.isError || predictionQuery.isError) {
    return (
      <div className="flex items-center justify-center min-h-screen bg-black text-red-400">
        ❌ Error loading data
      </div>
    )
  }

  const weather = weatherQuery.data.current
  const prediction = predictionQuery.data

  const isRainy = weather.condition.text.toLowerCase().includes('rain')
  const isCloudy = weather.condition.text.toLowerCase().includes('cloud')

  // 🌞 Solar suitability logic
  const isSuitable =
    prediction.solar_radiation >= 500 &&
    prediction.sunny_probability >= 0.6 &&
    prediction.is_anomaly === 0

  return (
    <div className="min-h-screen bg-black text-white p-8">
      <div className="flex flex-col grid grid-cols-1 md:grid-cols-2 gap-6">

        {/* WEATHER */}
        <Card className="bg-zinc-900">
          <CardContent className="space-y-4">
            <div className="flex items-center gap-3">
              {isRainy ? (
                <CloudRain className="text-blue-400" />
              ) : isCloudy ? (
                <Cloud className="text-gray-400" />
              ) : (
                <Sun className="text-yellow-400" />
              )}
              <h2 className="text-xl font-bold">Live Weather</h2>
            </div>

            <div className="grid grid-cols-2 gap-4 text-sm">
              <p>🌡 Temp: {weather.temp_c}°C</p>
              <p>💧 Humidity: {weather.humidity}%</p>
              <p>💨 Wind: {weather.wind_kph} km/h</p>
              <p>☁ Condition: {weather.condition.text}</p>
            </div>
          </CardContent>
        </Card>

        {/* <Separator orientation="horizontal" className="hidden md:block bg-white" /> */}

        {/* AI PREDICTION */}
        <Card className="bg-zinc-900">
          <CardContent className="space-y-5">
            <h2 className="text-xl font-bold">AI Solar Analysis</h2>

            <p className="text-3xl font-semibold">
              {prediction.solar_radiation} W/m²
            </p>

            <Progress value={prediction.sunny_probability * 100} />
            <p className="text-sm">
              Sunny Probability: {(prediction.sunny_probability * 100).toFixed(1)}%
            </p>

            <div className="flex gap-2 flex-wrap">
              <Badge variant="secondary">Cluster {prediction.cluster}</Badge>

              {prediction.is_anomaly === 1 && (
                <Badge variant="destructive" className="flex items-center gap-1">
                  <AlertTriangle size={14} /> Anomaly
                </Badge>
              )}
            </div>

            {/* ✅ SUITABILITY RESULT */}
            <div
              className={`mt-4 p-4 rounded-xl flex items-center gap-3 ${
                isSuitable ? 'bg-green-900/40' : 'bg-red-900/40'
              }`}
            >
              {isSuitable ? (
                <CheckCircle className="text-green-400" size={28} />
              ) : (
                <XCircle className="text-red-400" size={28} />
              )}

              <div>
                <p className="font-semibold">
                  {isSuitable
                    ? 'Suitable for Solar Energy Production'
                    : 'Not Ideal for Solar Energy Production'}
                </p>
                <p className="text-sm text-zinc-400">
                  {isSuitable
                    ? 'High radiation and stable conditions detected.'
                    : 'Low efficiency or abnormal conditions detected.'}
                </p>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
