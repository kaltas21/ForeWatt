import { RealTimeData, ModelType, AnomalyData, HistoricalData, ComparisonData, Alert } from '../types';

const getExactHour = (date: Date) => {
    const d = new Date(date);
    d.setMinutes(0, 0, 0);
    d.setSeconds(0, 0);
    d.setMilliseconds(0);
    return d;
}

const NOW = getExactHour(new Date());

const formatTime = (date: Date) => {
  return date.toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit', hour12: false });
};

// Helper to simulate time shifting
const addHours = (date: Date, h: number) => {
  const d = new Date(date);
  d.setHours(d.getHours() + h);
  return d;
};

export const generateRealTimeData = (model: ModelType): RealTimeData => {
  const isPrice = model === 'price';
  const baseValue = isPrice ? 1500 : 35000;
  const variance = isPrice ? 300 : 5000;

  // Generate 6h actuals (ending 2h ago)
  // UPDATED: Changed condition to i >= 2 to include the pivot point (T-2h)
  const actuals = [];
  for (let i = 8; i >= 2; i--) {
    const time = addHours(NOW, -i);
    actuals.push({
      timestamp: time.toISOString(),
      value: baseValue + Math.random() * variance - (variance / 2),
    });
  }

  // 12h Forecast
  const forecast = [];
  for (let i = 0; i < 12; i++) {
    const time = addHours(NOW, i);
    const val = baseValue + Math.random() * variance - (variance / 2) + (isPrice ? 200 : 1000); // Slight drift
    const uncertainty = isPrice ? 150 : 2000;
    forecast.push({
      timestamp: time.toISOString(),
      value: val,
      lower: val - uncertainty,
      upper: val + uncertainty,
    });
  }

  const pivotTime = addHours(NOW, -2).toISOString();

  return {
    modelType: model,
    unit: isPrice ? 'TL/MWh' : 'MWh',
    timezone: 'Europe/Istanbul',
    lastUpdated: new Date().toISOString(), // Keep actual wall clock for last updated
    actual: actuals,
    pivotTime: pivotTime,
    forecast: forecast,
    summary: {
      avgActual: actuals.reduce((acc, curr) => acc + curr.value, 0) / actuals.length,
      avgForecast: forecast.reduce((acc, curr) => acc + curr.value, 0) / forecast.length,
      peakActual: { value: Math.max(...actuals.map(d => d.value)), time: formatTime(new Date(actuals[0].timestamp)) },
      peakForecast: { value: Math.max(...forecast.map(d => d.value)), time: formatTime(new Date(forecast[forecast.length - 1].timestamp)) },
    }
  };
};

export const generateAnomalyData = (model: ModelType): AnomalyData => {
    const count = 50;
    const anomalies = [];
    let anomalyCount = 0;
    
    for(let i=0; i<count; i++) {
        const isAnom = Math.random() > 0.8;
        if(isAnom) anomalyCount++;
        anomalies.push({
            timestamp: addHours(NOW, -i).toISOString(),
            actual: 1000 + Math.random() * 500,
            forecast: 1000 + Math.random() * 500,
            residual: Math.random() * 200,
            anomalyScore: Math.random(),
            isAnomaly: isAnom
        });
    }

    return {
        summary: {
            totalRows: count,
            anomalyCount,
            anomalyRate: (anomalyCount/count) * 100,
            maxScore: 0.98,
            maxResidual: 450,
            meanResidual: 120
        },
        anomalies,
        scoreDistribution: {
            count,
            mean: 0.4,
            std: 0.2,
            min: 0.1,
            max: 0.98
        }
    }
}

export const generateHistoricalData = (model: ModelType, startDate: Date, endDate: Date): HistoricalData => {
    const data = [];
    let current = new Date(startDate);
    const end = new Date(endDate);
    
    // Ensure we work with hours
    current.setMinutes(0,0,0);
    end.setMinutes(0,0,0);

    const isPrice = model === 'price';
    const baseVal = isPrice ? 1200 : 30000;
    const volatility = isPrice ? 400 : 8000;

    while (current <= end) {
        // Create simple seasonality
        const hour = current.getHours();
        
        let seasonalFactor = 1;
        // Day pattern (Peak evening)
        if (hour >= 17 && hour <= 21) seasonalFactor = 1.3;
        if (hour >= 2 && hour <= 5) seasonalFactor = 0.7;
        
        // Random component
        const random = (Math.random() - 0.5) * volatility;
        
        const actual = baseVal * seasonalFactor + random;
        // Forecast usually close to actual with some error
        const forecast = actual + (Math.random() - 0.5) * (volatility * 0.2); 

        data.push({
            timestamp: current.toISOString(),
            actual: Math.max(0, actual),
            forecast: Math.max(0, forecast),
        });

        current.setHours(current.getHours() + 1);
    }

    const values = data.map(d => d.actual);
    const mean = values.reduce((a,b) => a+b, 0) / values.length;
    const sorted = [...values].sort((a,b) => a-b);

    return {
        data,
        statistics: {
            mean: mean,
            median: sorted[Math.floor(sorted.length / 2)],
            std: Math.sqrt(values.map(x => Math.pow(x - mean, 2)).reduce((a,b) => a+b) / values.length),
            min: sorted[0],
            max: sorted[sorted.length - 1]
        }
    }
}

export const generateComparisonData = (model: ModelType, labelA: string, labelB: string): ComparisonData => {
    const points = 24; // Compare 24 hours
    const isPrice = model === 'price';
    const baseVal = isPrice ? 1500 : 35000;
    
    const dataA = [];
    const dataB = [];

    for (let i=0; i<points; i++) {
        const hour = i;
        // Seasonal pattern
        let factor = 1;
        if (hour >= 18 && hour <= 21) factor = 1.25;
        if (hour >= 3 && hour <= 6) factor = 0.7;

        const valA = baseVal * factor + (Math.random() * (baseVal * 0.1));
        const valB = baseVal * factor * 0.95 + (Math.random() * (baseVal * 0.15)); // Slightly different period

        dataA.push({ label: `${hour.toString().padStart(2, '0')}:00`, value: valA });
        dataB.push({ label: `${hour.toString().padStart(2, '0')}:00`, value: valB });
    }

    return {
        periodA: { label: labelA, data: dataA },
        periodB: { label: labelB, data: dataB },
        metrics: {
            diffMean: 12.5,
            diffPeak: 5.2,
            volatilityA: 150.4,
            volatilityB: 180.2
        }
    }
}

export const generateAlertHistory = (): Alert[] => {
    return [
        { id: '1', type: 'PRICE', severity: 'critical', title: 'Price Spike Detected', message: 'Price forecast exceeds 2,200 TL/MWh for 20:00.', timestamp: new Date(Date.now() - 1000 * 60 * 30).toISOString(), read: false },
        { id: '2', type: 'SYSTEM', severity: 'info', title: 'Model Updated', message: 'Consumption model V2.4 deployed successfully.', timestamp: new Date(Date.now() - 1000 * 60 * 60 * 5).toISOString(), read: true },
        { id: '3', type: 'ANOMALY', severity: 'warning', title: 'High Anomaly Score', message: 'Unusual consumption dip detected at 14:00.', timestamp: new Date(Date.now() - 1000 * 60 * 60 * 24).toISOString(), read: true },
        { id: '4', type: 'CONSUMPTION', severity: 'warning', title: 'Demand Surge', message: 'Forecast suggests 15% increase vs last week.', timestamp: new Date(Date.now() - 1000 * 60 * 60 * 26).toISOString(), read: true },
    ];
}

export const generateSmartSummary = (data: RealTimeData): string => {
    const peak = data.summary.peakForecast;
    const avg = data.summary.avgActual;
    const diff = ((peak.value - avg) / avg) * 100;
    
    // Natural language templates
    const templates = [
        `Prices expected to peak at ${Math.round(peak.value).toLocaleString()} ${data.unit} around ${peak.time}, approximately ${Math.round(diff)}% above the current average.`,
        `Forecast shows a significant ramp up to ${Math.round(peak.value).toLocaleString()} ${data.unit} by ${peak.time}. Volatility remains moderate.`,
        `Expect stable conditions until ${peak.time}, where values may reach ${Math.round(peak.value).toLocaleString()} ${data.unit}.`
    ];
    
    return templates[0]; // Simple selection for mock
}