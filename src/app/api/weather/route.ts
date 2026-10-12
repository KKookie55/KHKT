import { NextRequest, NextResponse } from "next/server";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

/**
 * GET /api/weather?lat=16.5&lon=106.5
 * Fetches weather from Open-Meteo. Falls back to defaults if rate-limited.
 */
export async function GET(request: NextRequest) {
  const searchParams = request.nextUrl.searchParams;
  const lat = searchParams.get("lat");
  const lon = searchParams.get("lon");

  if (!lat || !lon) {
    return NextResponse.json({ error: "Missing lat/lon" }, { status: 400 });
  }

  try {
    const latNum = parseFloat(lat);
    const lonNum = parseFloat(lon);

    // Bangkok timezone
    const bkOffset = 7 * 60 * 60 * 1000;
    const bkTime = new Date(Date.now() + bkOffset);
    const todayStr = bkTime.toISOString().slice(0, 10);
    const yearStart = new Date(Date.UTC(bkTime.getUTCFullYear(), 0, 0));
    const doy = Math.floor((bkTime.getTime() - yearStart.getTime()) / 86400000);

    // Fetch forecast
    const forecastUrl = `https://api.open-meteo.com/v1/forecast?latitude=${lat}&longitude=${lon}&current=temperature_2m,relative_humidity_2m,wind_speed_10m,precipitation&daily=temperature_2m_max,temperature_2m_min,relative_humidity_2m_max,relative_humidity_2m_min,precipitation_sum,wind_speed_10m_max&timezone=Asia%2FBangkok&forecast_days=7`;
    
    let temp = 28, humidity = 70, windMs = 3, rain = 0, daysSinceRain = 5;
    let forecast: any[] = [];
    let source = "default (API unavailable)";

    try {
      const forecastResp = await fetch(forecastUrl, { signal: AbortSignal.timeout(8000) });
      if (forecastResp.ok) {
        const data = await forecastResp.json();
        const current = data.current;
        const daily = data.daily;
        
        temp = current.temperature_2m;
        humidity = current.relative_humidity_2m;
        const wind = current.wind_speed_10m;
        windMs = Math.round((wind / 3.6) * 10) / 10;
        rain = current.precipitation;

        // daysSinceRain from daily forecast (fallback to 7-day window)
        const dailyTimes = daily.time as string[];
        const todayIdx = dailyTimes.indexOf(todayStr);
        daysSinceRain = 0;
        if (todayIdx >= 0) {
          for (let i = todayIdx; i >= 0; i--) {
            if (daily.precipitation_sum[i] > 0.5) break;
            daysSinceRain++;
          }
        }

        forecast = (daily.time as string[]).map((date: string, i: number) => ({
          date,
          temp_max: daily.temperature_2m_max[i],
          temp_min: daily.temperature_2m_min[i],
          temp_avg: Math.round(((daily.temperature_2m_max[i] + daily.temperature_2m_min[i]) / 2) * 10) / 10,
          rh_avg: Math.round(((daily.relative_humidity_2m_max[i] + daily.relative_humidity_2m_min[i]) / 2) * 10) / 10,
          rain: daily.precipitation_sum[i],
          wind: Math.round((daily.wind_speed_10m_max[i] / 3.6) * 10) / 10,
        }));
        source = "Open-Meteo (forecast)";
      }
    } catch {
      // Forecast failed — use defaults
    }

    // Try archive for longer daysSinceRain (non-blocking — if it fails, keep forecast-based value)
    try {
      const startDate = new Date(bkTime.getTime() - 35 * 86400000).toISOString().slice(0, 10);
      const archiveUrl = `https://archive-api.open-meteo.com/v1/archive?latitude=${lat}&longitude=${lon}&start_date=${startDate}&end_date=${todayStr}&daily=precipitation_sum&timezone=Asia%2FBangkok`;
      const archiveResp = await fetch(archiveUrl, { signal: AbortSignal.timeout(10000) });
      if (archiveResp.ok) {
        const archiveData = await archiveResp.json();
        if (archiveData.daily?.precipitation_sum) {
          const archPrecip = archiveData.daily.precipitation_sum;
          let dsr = 0;
          for (let i = archPrecip.length - 1; i >= 0; i--) {
            if (archPrecip[i] !== null && archPrecip[i] >= 0.5) break;
            dsr++;
          }
          daysSinceRain = dsr;
          source = "Open-Meteo (forecast + 35-day archive)";
        }
      }
    } catch {
      // Archive failed — keep forecast-based daysSinceRain
    }

    const region = latNum >= 18.5 ? "north" : latNum >= 13.5 ? "central" : "south";

    return NextResponse.json({
      temp: Math.round(temp * 10) / 10,
      humidity: Math.round(humidity * 10) / 10,
      wind: windMs,
      rain: Math.round(rain * 10) / 10,
      daysSinceRain,
      region,
      doy,
      date: todayStr,
      source,
      lat: latNum,
      lon: lonNum,
      forecast,
    });
  } catch (error) {
    // Return defaults instead of 500 error — map still works
    const msg = error instanceof Error ? error.message : String(error);
    return NextResponse.json({
      temp: 28, humidity: 70, wind: 3, rain: 0,
      daysSinceRain: 5, region: "south", doy: 285,
      date: new Date().toISOString().slice(0, 10),
      source: `fallback (error: ${msg})`,
      lat: parseFloat(lat), lon: parseFloat(lon),
      forecast: [],
    });
  }
}
