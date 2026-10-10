import { NextRequest, NextResponse } from "next/server";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

/**
 * GET /api/weather?lat=16.5&lon=106.5
 * Fetches real-time weather from Open-Meteo API (free, no key needed).
 * Returns: temp, humidity, wind, rain, daysSinceRain, date
 */
export async function GET(request: NextRequest) {
  const searchParams = request.nextUrl.searchParams;
  const lat = searchParams.get("lat");
  const lon = searchParams.get("lon");

  if (!lat || !lon) {
    return NextResponse.json({ error: "Missing lat/lon params" }, { status: 400 });
  }

  try {
    // Open-Meteo API: current + daily forecast (7 days)
    // Variables: temperature_2m, relative_humidity_2m, wind_speed_10m, precipitation
    const url = `https://api.open-meteo.com/v1/forecast?latitude=${lat}&longitude=${lon}&current=temperature_2m,relative_humidity_2m,wind_speed_10m,precipitation&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,wind_speed_10m_max&timezone=Asia%2FBangkok&forecast_days=7`;

    const response = await fetch(url);
    
    if (!response.ok) {
      throw new Error(`Open-Meteo API returned ${response.status}`);
    }

    const data = await response.json();
    
    const current = data.current;
    const daily = data.daily;
    
    // Current weather
    const temp = current.temperature_2m;
    const humidity = current.relative_humidity_2m;
    const wind = current.wind_speed_10m; // km/h → convert to m/s
    const windMs = Math.round((wind / 3.6) * 10) / 10;
    const rain = current.precipitation; // mm

    // Calculate days since last rain from daily data
    // Open-Meteo daily gives past days + forecast
    let daysSinceRain = 0;
    const today = new Date();
    const todayStr = today.toISOString().slice(0, 10);
    
    // Find today's index in daily.time
    const dailyTimes = daily.time as string[];
    const todayIdx = dailyTimes.indexOf(todayStr);
    
    if (todayIdx >= 0) {
      // Count backwards from today
      for (let i = todayIdx; i >= 0; i--) {
        if (daily.precipitation_sum[i] > 0.5) break;
        daysSinceRain++;
      }
    }

    // Determine region based on latitude
    const latNum = parseFloat(lat);
    const region = latNum >= 18.5 ? "north" : latNum >= 13.5 ? "central" : "south";

    // Day of year
    const doy = Math.floor((today.getTime() - new Date(today.getFullYear(), 0, 0).getTime()) / 86400000);

    return NextResponse.json({
      temp: Math.round(temp * 10) / 10,
      humidity: Math.round(humidity * 10) / 10,
      wind: windMs,
      rain: Math.round(rain * 10) / 10,
      daysSinceRain,
      region,
      doy,
      date: todayStr,
      source: "Open-Meteo",
      lat: parseFloat(lat),
      lon: parseFloat(lon),
      // 7-day forecast for horizon selection
      forecast: dailyTimes.map((date: string, i: number) => ({
        date,
        temp_max: daily.temperature_2m_max[i],
        temp_min: daily.temperature_2m_min[i],
        rain: daily.precipitation_sum[i],
        wind: daily.wind_speed_10m_max[i],
      })),
    });
  } catch (error) {
    const msg = error instanceof Error ? error.message : String(error);
    return NextResponse.json({ error: msg }, { status: 500 });
  }
}
