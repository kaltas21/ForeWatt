
import React, { useState, useEffect } from 'react';
import { ModelType, HistoricalData } from '../types';
import { fetchHistoricalData, fetchDataStatus } from '../services/api';
import { AnomalyHistoryChart, HourlyPatternsChart } from '../components/Charts';
import { Card, Badge, Button } from '../components/ui';
import { AlertTriangle, Download, RefreshCw, WifiOff, Calendar, ChevronDown, Database } from 'lucide-react';
import { useLanguage } from '../contexts/LanguageContext';

const RANGES = [
    { label: 'Latest 3 Days', value: '3d', days: 3 },
    { label: 'Latest 7 Days', value: '7d', days: 7 },
    { label: 'Latest 15 Days', value: '15d', days: 15 },
    { label: 'Latest 1 Month', value: '1m', days: 30 },
];

export const AnomalyView = ({ model }: { model: ModelType }) => {
    const [data, setData] = useState<HistoricalData | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [rangeType, setRangeType] = useState('7d');
    const [anomalyThreshold, setAnomalyThreshold] = useState(2);
    const [dataStatus, setDataStatus] = useState<{ lastDate: Date | null }>({ lastDate: null });
    const { t } = useLanguage();

    // Fetch data status on mount - use current date as fallback for EPIAS live data
    useEffect(() => {
        const loadDataStatus = async () => {
            try {
                const status = await fetchDataStatus();
                if (status.last_timestamp) {
                    setDataStatus({ lastDate: new Date(status.last_timestamp) });
                } else {
                    // No parquet data - use current date (EPIAS live data available)
                    setDataStatus({ lastDate: new Date() });
                }
            } catch (err) {
                console.warn('Failed to fetch data status:', err);
                // Use current date as fallback (EPIAS live data)
                setDataStatus({ lastDate: new Date() });
            }
        };
        loadDataStatus();
    }, []);

    const loadData = async () => {
        setLoading(true);
        setError(null);
        try {
            const range = RANGES.find(r => r.value === rangeType)!;
            // Use data status last date if available, otherwise use now
            const end = dataStatus.lastDate || new Date();
            const start = new Date(end);
            start.setDate(end.getDate() - range.days);
            const newData = await fetchHistoricalData(model, start, end);
            setData(newData);
        } catch (err) {
            console.error('API error:', err);
            setError(err instanceof Error ? err.message : 'Failed to fetch data');
        }
        setLoading(false);
    };

    useEffect(() => {
        // Only load when we have data status or after a short delay
        if (dataStatus.lastDate) {
            loadData();
        }
    }, [model, rangeType, dataStatus.lastDate]);

    if (loading && !data) {
        return <div className="p-10 flex justify-center h-full items-center"><RefreshCw className="animate-spin text-primary-500 w-10 h-10" /></div>;
    }

    if (error && !data) {
        return (
            <div className="p-10 flex flex-col justify-center h-full items-center gap-4">
                <WifiOff className="text-red-500 w-12 h-12" />
                <p className="text-red-600 dark:text-red-400 font-medium">{error}</p>
                <Button onClick={loadData} disabled={loading}>
                    <RefreshCw size={16} className={loading ? 'animate-spin mr-2' : 'mr-2'} />
                    Retry
                </Button>
            </div>
        );
    }

    if(!data) return null;

    const hasData = data.data && data.data.length > 0;

    // Calculate anomaly statistics from historical data
    const values = data.data.map(d => d.actual).filter(v => v != null);
    const mean = values.length > 0 ? values.reduce((a, b) => a + b, 0) / values.length : 0;
    const std = values.length > 0 ? Math.sqrt(values.map(x => Math.pow(x - mean, 2)).reduce((a, b) => a + b, 0) / values.length) : 0;
    const upperBound = mean + anomalyThreshold * std;
    const lowerBound = mean - anomalyThreshold * std;

    const anomalies = data.data
        .map(d => ({ ...d, isAnomaly: d.actual > upperBound || d.actual < lowerBound }))
        .filter(d => d.isAnomaly);

    const anomalyRate = values.length > 0 ? (anomalies.length / values.length) * 100 : 0;
    const maxDeviation = anomalies.length > 0
        ? Math.max(...anomalies.map(a => Math.abs(a.actual - mean) / std))
        : 0;

    return (
        <div className="space-y-6">
            <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
                <div>
                    <h2 className="text-2xl font-bold text-slate-900 dark:text-white flex items-center gap-2">
                        <AlertTriangle className="text-amber-500" />
                        {t('anomaly.title')}
                        <Badge color="green">Live EPIAS Data</Badge>
                        {!hasData && <Badge color="yellow">Limited Data</Badge>}
                    </h2>
                    <p className="text-slate-500 dark:text-slate-400 text-sm mt-1">
                        Detecting anomalies using {anomalyThreshold}x standard deviation threshold
                    </p>
                </div>
                <RefreshCw
                    size={20}
                    className={`cursor-pointer text-slate-400 hover:text-slate-600 ${loading ? 'animate-spin' : ''}`}
                    onClick={loadData}
                />
            </div>

            {/* Controls */}
            <Card className="p-4 flex flex-wrap items-center gap-6 bg-white dark:bg-slate-900">
                <div className="space-y-2 min-w-[180px]">
                    <label className="text-xs font-bold text-slate-500 dark:text-slate-400 uppercase tracking-wider">Date Range</label>
                    <div className="relative">
                        <select
                            value={rangeType}
                            onChange={(e) => setRangeType(e.target.value)}
                            className="w-full appearance-none bg-slate-50 dark:bg-slate-800 border border-slate-200 dark:border-slate-700 text-slate-900 dark:text-white rounded-lg px-4 py-2.5 pr-8 focus:outline-none focus:ring-2 focus:ring-primary-500 font-medium"
                        >
                            {RANGES.map(r => <option key={r.value} value={r.value}>{r.label}</option>)}
                        </select>
                        <ChevronDown className="absolute right-3 top-3 text-slate-400 pointer-events-none" size={16} />
                    </div>
                </div>
                <div className="space-y-2 min-w-[180px]">
                    <label className="text-xs font-bold text-slate-500 dark:text-slate-400 uppercase tracking-wider">Sensitivity (Sigma)</label>
                    <div className="relative">
                        <select
                            value={anomalyThreshold}
                            onChange={(e) => setAnomalyThreshold(Number(e.target.value))}
                            className="w-full appearance-none bg-slate-50 dark:bg-slate-800 border border-slate-200 dark:border-slate-700 text-slate-900 dark:text-white rounded-lg px-4 py-2.5 pr-8 focus:outline-none focus:ring-2 focus:ring-primary-500 font-medium"
                        >
                            <option value={1.5}>1.5x (High Sensitivity)</option>
                            <option value={2}>2x (Normal)</option>
                            <option value={2.5}>2.5x (Low Sensitivity)</option>
                            <option value={3}>3x (Very Low)</option>
                        </select>
                        <ChevronDown className="absolute right-3 top-3 text-slate-400 pointer-events-none" size={16} />
                    </div>
                </div>
            </Card>

            {hasData ? (
                <>
                    <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                        <Card className="p-5">
                            <div className="text-slate-500 dark:text-slate-400 text-xs font-semibold uppercase tracking-wider mb-2">{t('anomaly.rate')}</div>
                            <div className="text-3xl font-bold text-amber-600">{anomalyRate.toFixed(1)}%</div>
                        </Card>
                        <Card className="p-5">
                            <div className="text-slate-500 dark:text-slate-400 text-xs font-semibold uppercase tracking-wider mb-2">{t('anomaly.count')}</div>
                            <div className="text-3xl font-bold text-red-600">{anomalies.length}</div>
                        </Card>
                        <Card className="p-5">
                            <div className="text-slate-500 dark:text-slate-400 text-xs font-semibold uppercase tracking-wider mb-2">Max Deviation</div>
                            <div className="text-3xl font-bold text-slate-800 dark:text-slate-200">{maxDeviation.toFixed(1)}x</div>
                        </Card>
                        <Card className="p-5">
                            <div className="text-slate-500 dark:text-slate-400 text-xs font-semibold uppercase tracking-wider mb-2">Data Points</div>
                            <div className="text-3xl font-bold text-slate-800 dark:text-slate-200">{values.length}</div>
                        </Card>
                    </div>

                    <div className="grid lg:grid-cols-3 gap-6">
                        <Card className="lg:col-span-2 p-6">
                            <div className="flex justify-between items-center mb-4">
                                <h3 className="font-semibold text-slate-900 dark:text-slate-200">Anomaly Detection on Actual EPIAS Data</h3>
                                <div className="flex gap-2 text-slate-500">
                                    <Download size={16} className="cursor-pointer hover:text-slate-900 dark:hover:text-white" />
                                </div>
                            </div>
                            <AnomalyHistoryChart data={data} anomalyThreshold={anomalyThreshold} />
                        </Card>
                        <Card className="p-6">
                             <h3 className="font-semibold text-slate-900 dark:text-slate-200 mb-4">{t('anomaly.topAnomalies')}</h3>
                             <div className="overflow-y-auto max-h-[420px] space-y-3 pr-2 scrollbar-hide">
                                 {anomalies.length > 0 ? (
                                     anomalies
                                         .sort((a, b) => Math.abs(b.actual - mean) - Math.abs(a.actual - mean))
                                         .slice(0, 10)
                                         .map((a, i) => (
                                         <div key={i} className="flex justify-between items-center p-3 bg-red-50 dark:bg-red-900/10 rounded-md shadow-sm border border-red-100 dark:border-red-900/30">
                                             <div>
                                                 <div className="text-[10px] text-red-600 dark:text-red-400 font-bold uppercase mb-1">
                                                    {new Date(a.timestamp).toLocaleDateString()} {new Date(a.timestamp).toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})}
                                                 </div>
                                                 <div className="text-sm font-bold text-slate-900 dark:text-white">
                                                    Value: {a.actual.toLocaleString(undefined, { maximumFractionDigits: 0 })}
                                                 </div>
                                             </div>
                                             <div className="text-xs font-bold text-red-500">
                                                 {((Math.abs(a.actual - mean) / std)).toFixed(1)}x sigma
                                             </div>
                                         </div>
                                     ))
                                 ) : (
                                     <p className="text-slate-500 text-sm text-center py-8">No anomalies detected with current threshold</p>
                                 )}
                             </div>
                        </Card>
                    </div>

                    {/* Hourly Patterns */}
                    <Card className="p-6">
                        <h4 className="font-semibold text-slate-900 dark:text-white mb-4 flex items-center gap-2">
                            <Calendar size={20} className="text-primary-500" />
                            Hourly Patterns
                        </h4>
                        <HourlyPatternsChart data={data} />
                    </Card>
                </>
            ) : (
                <Card className="p-10 text-center">
                    <AlertTriangle size={48} className="mx-auto text-slate-300 dark:text-slate-600 mb-4" />
                    <h3 className="text-lg font-semibold text-slate-700 dark:text-slate-300 mb-2">No Data Available</h3>
                    <p className="text-slate-500 dark:text-slate-400 max-w-md mx-auto">
                        No historical data available for anomaly detection. Please check your API connection.
                    </p>
                </Card>
            )}
        </div>
    );
};
