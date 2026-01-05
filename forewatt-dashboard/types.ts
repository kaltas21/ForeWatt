export type ViewState = 'HOME' | 'REALTIME' | 'HISTORICAL' | 'ANOMALY' | 'COMPARE' | 'ALERTS';
export type ModelType = 'consumption' | 'price';

export interface DataPoint {
  timestamp: string;
  value: number;
  lower?: number;
  upper?: number;
}

export interface RealTimeData {
  modelType: ModelType;
  unit: string;
  timezone: string;
  lastUpdated: string;
  actual: DataPoint[];
  pivotTime: string;
  forecast: DataPoint[];
  summary: {
    avgActual: number;
    avgForecast: number;
    peakActual: { value: number; time: string };
    peakForecast: { value: number; time: string };
  };
}

export interface AnomalyRecord {
  timestamp: string;
  actual: number;
  forecast: number;
  residual: number;
  anomalyScore: number;
  isAnomaly: boolean;
}

export interface AnomalyData {
  summary: {
    totalRows: number;
    anomalyCount: number;
    anomalyRate: number;
    maxScore: number;
    maxResidual: number;
    meanResidual: number;
  };
  anomalies: AnomalyRecord[];
  scoreDistribution: {
    count: number;
    mean: number;
    std: number;
    min: number;
    max: number;
  };
}

export interface HistoricalData {
    data: {timestamp: string; actual: number; forecast: number | null}[];
    statistics: {
        mean: number;
        median: number;
        std: number;
        min: number;
        max: number;
        mape?: number;  // Mean Absolute Percentage Error (if forecasts exist)
        forecastCount?: number;  // Number of data points with forecasts
    }
}

export interface ComparisonData {
  periodA: { label: string; data: { label: string; value: number }[] };
  periodB: { label: string; data: { label: string; value: number }[] };
  metrics: {
    diffMean: number;
    diffPeak: number;
    volatilityA: number;
    volatilityB: number;
  };
}

export interface Alert {
  id: string;
  type: 'PRICE' | 'CONSUMPTION' | 'ANOMALY' | 'SYSTEM';
  severity: 'info' | 'warning' | 'critical';
  title: string;
  message: string;
  timestamp: string;
  read: boolean;
}