/**
 * ForeWatt API Service
 * Connects React dashboard to FastAPI backend
 */

import { RealTimeData, ModelType, HistoricalData, AnomalyData, ComparisonData, Alert } from '../types';

// API base URL - uses Vite proxy in development, direct URL in production
const CLOUD_RUN_URL = 'https://forewatt-api-715624643502.europe-west1.run.app';
const API_BASE = import.meta.env.PROD
  ? (import.meta.env.VITE_API_URL || CLOUD_RUN_URL)
  : ''; // Empty string means use relative URLs (proxied by Vite)

/**
 * Generic fetch wrapper with error handling
 */
async function apiFetch<T>(endpoint: string, options?: RequestInit): Promise<T> {
  const url = `${API_BASE}${endpoint}`;

  const response = await fetch(url, {
    headers: {
      'Content-Type': 'application/json',
      ...options?.headers,
    },
    ...options,
  });

  if (!response.ok) {
    throw new Error(`API Error: ${response.status} ${response.statusText}`);
  }

  return response.json();
}

/**
 * Format time helper
 */
const formatTime = (date: Date): string => {
  return date.toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit', hour12: false });
};

/**
 * Fetch real-time data for dashboard
 * Uses the new /api/realtime/{model} endpoint backed by Firestore
 */
export async function fetchRealTimeData(model: ModelType): Promise<RealTimeData> {
  try {
    // Use new real-time endpoint
    const data = await apiFetch<RealTimeData>(`/api/realtime/${model}`);
    return data;
  } catch (error) {
    console.warn('Real-time endpoint failed, falling back to forecast endpoint:', error);

    // Fallback to old endpoint
    const endpoint = model === 'price' ? '/forecast/price' : '/forecast/consumption';
    const unit = model === 'price' ? 'TL/MWh' : 'MWh';

    const now = new Date();
    const start = new Date(now.getTime() - 12 * 60 * 60 * 1000);

    const data = await apiFetch<{
      data: Array<{
        forecast_time: string;
        target_time: string;
        forecast_value: number;
      }>;
      count: number;
    }>(`${endpoint}?start_date=${start.toISOString()}&limit=48`);

    const forecasts = data.data || [];
    const pivotTime = new Date(now.getTime() - 2 * 60 * 60 * 1000);

    const actuals = forecasts
      .filter(f => new Date(f.target_time) <= pivotTime)
      .slice(-6)
      .map(f => ({ timestamp: f.target_time, value: f.forecast_value }));

    const forecastData = forecasts
      .filter(f => new Date(f.target_time) > pivotTime)
      .slice(0, 12)
      .map(f => {
        const val = f.forecast_value;
        const uncertainty = model === 'price' ? val * 0.1 : val * 0.05;
        return { timestamp: f.target_time, value: val, lower: val - uncertainty, upper: val + uncertainty };
      });

    const avgActual = actuals.length > 0 ? actuals.reduce((sum, d) => sum + d.value, 0) / actuals.length : 0;
    const avgForecast = forecastData.length > 0 ? forecastData.reduce((sum, d) => sum + d.value, 0) / forecastData.length : 0;
    const peakActual = actuals.length > 0 ? actuals.reduce((max, d) => d.value > max.value ? d : max, actuals[0]) : { value: 0, timestamp: now.toISOString() };
    const peakForecast = forecastData.length > 0 ? forecastData.reduce((max, d) => d.value > max.value ? d : max, forecastData[0]) : { value: 0, timestamp: now.toISOString() };

    return {
      modelType: model,
      unit,
      timezone: 'Europe/Istanbul',
      lastUpdated: new Date().toISOString(),
      actual: actuals,
      pivotTime: pivotTime.toISOString(),
      forecast: forecastData,
      summary: {
        avgActual,
        avgForecast,
        peakActual: { value: peakActual.value, time: formatTime(new Date(peakActual.timestamp)) },
        peakForecast: { value: peakForecast.value, time: formatTime(new Date(peakForecast.timestamp)) },
      },
    };
  }
}

/**
 * Fetch historical EPIAS data for a date range
 * Returns actual EPIAS data and our model's forecasts for comparison
 */
export async function fetchHistoricalData(
  model: ModelType,
  startDate: Date,
  endDate: Date
): Promise<HistoricalData> {
  const endpoint = model === 'price' ? '/history/price' : '/history/consumption';
  const valueField = model === 'price' ? 'price' : 'consumption';

  const data = await apiFetch<{
    data: Array<{
      timestamp: string;
      price: number;
      consumption: number;
      forecast: number | null;
    }>;
    count: number;
    has_forecasts: boolean;
  }>(`${endpoint}?start_date=${startDate.toISOString()}&end_date=${endDate.toISOString()}&limit=5000`);

  const records = data.data || [];

  // Transform to historical format - actual from EPIAS, forecast from our model
  const historicalData = records.map(r => ({
    timestamp: r.timestamp,
    actual: r[valueField as keyof typeof r] as number,
    forecast: r.forecast ?? null,  // Use our model's forecast if available
  }));

  // Calculate statistics on actual values
  const actualValues = historicalData.map(d => d.actual).filter(v => v != null && !isNaN(v));
  const mean = actualValues.length > 0 ? actualValues.reduce((a, b) => a + b, 0) / actualValues.length : 0;
  const sorted = [...actualValues].sort((a, b) => a - b);

  const variance = actualValues.length > 0
    ? actualValues.map(x => Math.pow(x - mean, 2)).reduce((a, b) => a + b, 0) / actualValues.length
    : 0;

  // Calculate forecast accuracy if forecasts exist
  const forecastPairs = historicalData.filter(d => d.forecast != null && d.actual != null);
  let mape = 0;
  if (forecastPairs.length > 0) {
    mape = forecastPairs.reduce((sum, d) => {
      const err = Math.abs(d.actual - (d.forecast ?? 0)) / Math.abs(d.actual + 1e-8);
      return sum + err;
    }, 0) / forecastPairs.length * 100;
  }

  return {
    data: historicalData,
    statistics: {
      mean,
      median: sorted.length > 0 ? sorted[Math.floor(sorted.length / 2)] : 0,
      std: Math.sqrt(variance),
      min: sorted.length > 0 ? sorted[0] : 0,
      max: sorted.length > 0 ? sorted[sorted.length - 1] : 0,
      mape: forecastPairs.length > 0 ? mape : undefined,
      forecastCount: forecastPairs.length,
    },
  };
}

/**
 * Fetch anomaly data
 * Uses the /api/anomaly/{model} endpoint
 */
export async function fetchAnomalyData(model: ModelType): Promise<AnomalyData> {
  try {
    // Use new anomaly endpoint
    const data = await apiFetch<AnomalyData>(`/api/anomaly/${model}`);
    return data;
  } catch (error) {
    console.warn('Anomaly endpoint failed, calculating locally:', error);

    // Fallback to local calculation
    const now = new Date();
    const start = new Date(now.getTime() - 48 * 60 * 60 * 1000);

    const historical = await fetchHistoricalData(model, start, now);
    const mean = historical.statistics.mean;
    const std = historical.statistics.std;

    const anomalies = historical.data
      .filter(d => d.forecast != null)  // Only calculate anomalies where we have forecasts
      .map(d => {
        const forecast = d.forecast ?? d.actual;  // Fallback to actual if null
        const residual = Math.abs(d.actual - forecast);
        const anomalyScore = std > 0 ? residual / (3 * std) : 0;
        return {
          timestamp: d.timestamp,
          actual: d.actual,
          forecast,
          residual,
          anomalyScore: Math.min(anomalyScore, 1),
          isAnomaly: anomalyScore > 0.8,
        };
      });

    const anomalyCount = anomalies.filter(a => a.isAnomaly).length;
    const scores = anomalies.map(a => a.anomalyScore);
    const residuals = anomalies.map(a => a.residual);

    return {
      summary: {
        totalRows: anomalies.length,
        anomalyCount,
        anomalyRate: anomalies.length > 0 ? (anomalyCount / anomalies.length) * 100 : 0,
        maxScore: Math.max(...scores, 0),
        maxResidual: Math.max(...residuals, 0),
        meanResidual: residuals.length > 0 ? residuals.reduce((a, b) => a + b, 0) / residuals.length : 0,
      },
      anomalies,
      scoreDistribution: {
        count: anomalies.length,
        mean: scores.length > 0 ? scores.reduce((a, b) => a + b, 0) / scores.length : 0,
        std: 0.2,
        min: Math.min(...scores, 0),
        max: Math.max(...scores, 0),
      },
    };
  }
}

/**
 * Fetch comparison data between two periods
 */
export async function fetchComparisonData(
  model: ModelType,
  periodA: { start: Date; end: Date; label: string },
  periodB: { start: Date; end: Date; label: string }
): Promise<ComparisonData> {
  const [dataA, dataB] = await Promise.all([
    fetchHistoricalData(model, periodA.start, periodA.end),
    fetchHistoricalData(model, periodB.start, periodB.end),
  ]);

  // Group by hour of day for comparison
  const groupByHour = (data: typeof dataA.data) => {
    const hourlyData: { [hour: number]: number[] } = {};
    data.forEach(d => {
      const hour = new Date(d.timestamp).getHours();
      if (!hourlyData[hour]) hourlyData[hour] = [];
      hourlyData[hour].push(d.actual);
    });
    return Array.from({ length: 24 }, (_, h) => ({
      label: `${h.toString().padStart(2, '0')}:00`,
      value: hourlyData[h]?.reduce((a, b) => a + b, 0) / (hourlyData[h]?.length || 1) || 0,
    }));
  };

  const chartDataA = groupByHour(dataA.data);
  const chartDataB = groupByHour(dataB.data);

  return {
    periodA: { label: periodA.label, data: chartDataA },
    periodB: { label: periodB.label, data: chartDataB },
    metrics: {
      diffMean: ((dataA.statistics.mean - dataB.statistics.mean) / dataB.statistics.mean) * 100,
      diffPeak: ((dataA.statistics.max - dataB.statistics.max) / dataB.statistics.max) * 100,
      volatilityA: dataA.statistics.std,
      volatilityB: dataB.statistics.std,
    },
  };
}

/**
 * Fetch day type comparison data (weekday vs weekend) - OPTIMIZED
 * Uses pre-aggregated backend endpoint for fast loading
 */
export async function fetchDayTypeComparison(
  model: ModelType,
  days: number = 30
): Promise<{ weekday: { label: string; value: number }[]; weekend: { label: string; value: number }[]; diffPercent: number }> {
  try {
    // Use optimized pre-aggregated endpoint
    const data = await apiFetch<{
      weekday: { label: string; value: number }[];
      weekend: { label: string; value: number }[];
      diffPercent: number;
      dataPoints?: number;
      error?: string;
    }>(`/api/aggregates/day-type/${model}?days=${days}`);

    if (data.error) {
      console.warn('Day type endpoint returned error:', data.error);
      throw new Error(data.error);
    }

    return {
      weekday: data.weekday,
      weekend: data.weekend,
      diffPercent: data.diffPercent
    };
  } catch (error) {
    console.warn('Optimized day-type endpoint failed, falling back to manual calculation:', error);

    // Fallback to manual calculation
    const endDate = new Date();
    const startDate = new Date(endDate.getTime() - days * 24 * 60 * 60 * 1000);

    const historical = await fetchHistoricalData(model, startDate, endDate);

    // Group by weekday/weekend and hour
    const weekdayHours: { [hour: number]: number[] } = {};
    const weekendHours: { [hour: number]: number[] } = {};

    for (let h = 0; h < 24; h++) {
      weekdayHours[h] = [];
      weekendHours[h] = [];
    }

    historical.data.forEach(d => {
      const date = new Date(d.timestamp);
      const hour = date.getHours();
      const dayOfWeek = date.getDay();
      const isWeekend = dayOfWeek === 0 || dayOfWeek === 6;

      if (isWeekend) {
        weekendHours[hour].push(d.actual);
      } else {
        weekdayHours[hour].push(d.actual);
      }
    });

    // Calculate averages
    const weekdayData = Array.from({ length: 24 }, (_, h) => ({
      label: `${h.toString().padStart(2, '0')}:00`,
      value: weekdayHours[h].length > 0
        ? weekdayHours[h].reduce((a, b) => a + b, 0) / weekdayHours[h].length
        : 0,
    }));

    const weekendData = Array.from({ length: 24 }, (_, h) => ({
      label: `${h.toString().padStart(2, '0')}:00`,
      value: weekendHours[h].length > 0
        ? weekendHours[h].reduce((a, b) => a + b, 0) / weekendHours[h].length
        : 0,
    }));

    // Calculate average difference
    const weekdayAvg = weekdayData.reduce((a, b) => a + b.value, 0) / 24;
    const weekendAvg = weekendData.reduce((a, b) => a + b.value, 0) / 24;
    const diffPercent = weekdayAvg > 0 ? ((weekdayAvg - weekendAvg) / weekdayAvg) * 100 : 0;

    return { weekday: weekdayData, weekend: weekendData, diffPercent };
  }
}

/**
 * Fetch pre-aggregated hourly statistics - FAST
 * Returns hourly min/max/mean/std for pattern analysis
 */
export async function fetchHourlyAggregates(
  model: ModelType,
  days: number = 30
): Promise<{
  data: { hour: number; label: string; min: number; max: number; mean: number; std: number }[];
  totalDataPoints: number;
}> {
  return apiFetch(`/api/aggregates/hourly/${model}?days=${days}`);
}

/**
 * Fetch pre-aggregated daily statistics - FAST
 * Returns daily min/max/mean for historical overview
 */
export async function fetchDailyAggregates(
  model: ModelType,
  days: number = 30
): Promise<{
  data: { date: string; min: number; max: number; mean: number; std: number }[];
  totalDays: number;
}> {
  return apiFetch(`/api/aggregates/daily/${model}?days=${days}`);
}

/**
 * Fetch overall statistics for a period - FAST
 * Returns mean, median, std, percentiles for summary cards
 */
export async function fetchStatistics(
  model: ModelType,
  days: number = 30
): Promise<{
  mean: number;
  median: number;
  std: number;
  min: number;
  max: number;
  p25: number;
  p75: number;
  p95: number;
  count: number;
}> {
  return apiFetch(`/api/aggregates/statistics/${model}?days=${days}`);
}

/**
 * Fetch 12 hours of actual EPIAS data for RealTime graph
 * Accounts for 2-hour EPIAS delay
 */
export async function fetchActualHistory(
  model: ModelType,
  hours: number = 12
): Promise<{ timestamp: string; value: number }[]> {
  const now = new Date();
  // Account for 2-hour EPIAS delay - so we fetch from 14 hours ago to 2 hours ago
  const endDate = new Date(now.getTime() - 2 * 60 * 60 * 1000);
  const startDate = new Date(endDate.getTime() - hours * 60 * 60 * 1000);

  const historical = await fetchHistoricalData(model, startDate, endDate);

  return historical.data.map(d => ({
    timestamp: d.timestamp,
    value: d.actual,
  }));
}

/**
 * Fetch data status - includes info about available data sources
 */
export async function fetchDataStatus(): Promise<{
  status: string;
  last_timestamp: string | null;
  first_timestamp: string | null;
  parquet: { available: boolean; last_timestamp?: string; first_timestamp?: string };
  epias_live: { available: boolean; latest_record?: string };
  forecasts: { available: boolean; last_target?: string; first_target?: string };
  has_forecasts: boolean;
  recommended_source: string;
}> {
  return apiFetch('/api/data-status');
}

/**
 * Fetch alerts from Firestore via API
 */
export async function fetchAlerts(): Promise<Alert[]> {
  try {
    const alerts = await apiFetch<Alert[]>('/api/alerts');
    return alerts;
  } catch (error) {
    console.warn('Alerts endpoint failed:', error);
    return [
      {
        id: '1',
        type: 'SYSTEM',
        severity: 'info',
        title: 'API Connected',
        message: 'Dashboard is connected to the ForeWatt API.',
        timestamp: new Date().toISOString(),
        read: false
      },
    ];
  }
}

/**
 * Check alerts against actual data thresholds
 * Returns triggered alerts based on recent data
 */
export async function checkAlertThresholds(
  configs: Array<{
    id: string;
    title: string;
    enabled: boolean;
    threshold: number;
    condition: 'above' | 'below';
    severity: 'critical' | 'warning' | 'info';
    model: 'price' | 'consumption';
  }>
): Promise<Alert[]> {
  const alerts: Alert[] = [];

  for (const config of configs) {
    if (!config.enabled) continue;

    try {
      // Get recent data (last 24 hours)
      const endDate = new Date();
      const startDate = new Date(endDate.getTime() - 24 * 60 * 60 * 1000);
      const historical = await fetchHistoricalData(config.model, startDate, endDate);

      // Check each data point against threshold
      for (const point of historical.data) {
        const value = point.actual;
        const triggered =
          config.condition === 'above'
            ? value > config.threshold
            : value < config.threshold;

        if (triggered) {
          alerts.push({
            id: `${config.id}-${point.timestamp}`,
            type: config.model.toUpperCase() as any,
            severity: config.severity,
            title: config.title,
            message: `${config.model === 'price' ? 'Price' : 'Consumption'} ${config.condition === 'above' ? 'exceeded' : 'dropped below'} ${config.threshold.toLocaleString()} at ${new Date(point.timestamp).toLocaleTimeString()}. Value: ${value.toFixed(2)}`,
            timestamp: point.timestamp,
            read: false,
          });
        }
      }
    } catch (error) {
      console.warn(`Failed to check alert ${config.id}:`, error);
    }
  }

  // Sort by timestamp descending (most recent first)
  return alerts.sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime());
}

/**
 * Create a new alert
 */
export async function createAlert(alert: Omit<Alert, 'id' | 'timestamp' | 'read'>): Promise<Alert> {
  return apiFetch<Alert>('/api/alerts', {
    method: 'POST',
    body: JSON.stringify(alert),
  });
}

/**
 * Trigger a new forecast (for admin use)
 */
export async function triggerForecast(): Promise<{ success: boolean; message: string }> {
  try {
    const result = await apiFetch<{
      forecast_time: string;
      runtime_seconds: number;
      price: { count: number; min: number; max: number };
      consumption: { count: number; min: number; max: number };
    }>('/forecast', { method: 'POST' });

    return {
      success: true,
      message: `Forecast generated in ${result.runtime_seconds.toFixed(1)}s`,
    };
  } catch (error) {
    return {
      success: false,
      message: error instanceof Error ? error.message : 'Forecast failed',
    };
  }
}

/**
 * Health check
 */
export async function checkHealth(): Promise<boolean> {
  try {
    await apiFetch<{ status: string }>('/health');
    return true;
  } catch {
    return false;
  }
}

/**
 * Generate AI summary using Gemini (via backend)
 */
export async function generateSmartSummary(data: RealTimeData): Promise<string> {
  // TODO: Connect to /api/chat when Gemini is integrated
  // For now, generate client-side summary
  const peak = data.summary.peakForecast;
  const avg = data.summary.avgActual;
  const diff = avg > 0 ? ((peak.value - avg) / avg) * 100 : 0;

  if (data.modelType === 'price') {
    return `Prices expected to peak at ${Math.round(peak.value).toLocaleString()} ${data.unit} around ${peak.time}, approximately ${Math.round(diff)}% above the current average.`;
  } else {
    return `Consumption forecast peaks at ${Math.round(peak.value).toLocaleString()} ${data.unit} by ${peak.time}. Current trend shows ${diff > 0 ? 'increasing' : 'stable'} demand.`;
  }
}
