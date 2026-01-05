
import React, { useState, useEffect } from 'react';
import { ModelType, HistoricalData } from '../types';
import { fetchHistoricalData, fetchDataStatus } from '../services/api';
import { HistoricalChart, HourlyPatternsChart, ValueDistributionChart } from '../components/Charts';
import { Card, Button, Toggle, Badge } from '../components/ui';
import { Calendar, Download, RefreshCw, Filter, ChevronDown, WifiOff, Database } from 'lucide-react';
import { useLanguage } from '../contexts/LanguageContext';

const RANGES = [
    { label: 'Latest 3 Days', value: '3d', days: 3 },
    { label: 'Latest 7 Days', value: '7d', days: 7 },
    { label: 'Latest 15 Days', value: '15d', days: 15 },
    { label: 'Latest 1 Month', value: '1m', days: 30 },
    { label: 'Latest 3 Months', value: '3m', days: 90 },
    { label: 'Latest 6 Months', value: '6m', days: 180 },
    { label: 'Latest 1 Year', value: '1y', days: 365 },
    { label: 'Custom Range', value: 'custom', days: 0 }
];

export const HistoricalView = ({ model }: { model: ModelType }) => {
    const { t } = useLanguage();
    // State
    const [data, setData] = useState<HistoricalData | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    // Data status (last date in parquet)
    const [dataStatus, setDataStatus] = useState<{ lastDate: Date | null; firstDate: Date | null }>({
        lastDate: null,
        firstDate: null
    });

    // Filters
    const [rangeType, setRangeType] = useState('7d');
    const [startDate, setStartDate] = useState('');
    const [endDate, setEndDate] = useState('');
    const [showActual, setShowActual] = useState(true);
    const [showForecast, setShowForecast] = useState(true);

    // Use current date as reference for "Last X days" - EPIAS provides live data
    useEffect(() => {
        // Always use current time as the end date for "Last X days" selections
        // This ensures "Last 7 days" means the actual last 7 days, not historical parquet dates
        const now = new Date();
        setDataStatus({
            lastDate: now,
            firstDate: new Date(now.getTime() - 365 * 24 * 60 * 60 * 1000)
        });
        console.log('History page using current date as reference:', now);
    }, []);

    // Initial load and range changes - wait for dataStatus to be loaded
    useEffect(() => {
        if (dataStatus.lastDate) {
            handleFetch();
        }
    }, [model, rangeType, dataStatus.lastDate]);

    // Update custom inputs when preset selected (use last parquet date as end)
    useEffect(() => {
        if (!dataStatus.lastDate) return;

        const selectedRange = RANGES.find(r => r.value === rangeType);
        if (selectedRange && selectedRange.value !== 'custom') {
            // Use last parquet date as end date (not current date)
            const end = new Date(dataStatus.lastDate);
            const start = new Date(end);
            start.setDate(end.getDate() - selectedRange.days);

            // Format for datetime-local input: YYYY-MM-DDThh:mm
            setEndDate(end.toISOString().slice(0, 16));
            setStartDate(start.toISOString().slice(0, 16));
        }
    }, [rangeType, dataStatus.lastDate]);

    const handleFetch = async () => {
        setLoading(true);
        setError(null);

        let start: Date, end: Date;

        if (rangeType === 'custom') {
            start = startDate ? new Date(startDate) : new Date(Date.now() - 7 * 24 * 60 * 60 * 1000);
            end = endDate ? new Date(endDate) : (dataStatus.lastDate || new Date());
        } else {
            const range = RANGES.find(r => r.value === rangeType)!;
            // Use last parquet date as end date
            end = dataStatus.lastDate || new Date();
            start = new Date(end);
            start.setDate(end.getDate() - range.days);
        }

        try {
            const newData = await fetchHistoricalData(model, start, end);
            setData(newData);
        } catch (err) {
            console.error('API error:', err);
            setError(err instanceof Error ? err.message : 'Failed to fetch data');
        }
        setLoading(false);
    };

    if (loading && !data) {
        return <div className="p-10 flex justify-center h-full items-center"><RefreshCw className="animate-spin text-primary-500 w-10 h-10" /></div>;
    }

    if (error && !data) {
        return (
            <div className="p-10 flex flex-col justify-center h-full items-center gap-4">
                <WifiOff className="text-red-500 w-12 h-12" />
                <p className="text-red-600 dark:text-red-400 font-medium">{error}</p>
                <Button onClick={handleFetch} disabled={loading}>
                    <RefreshCw size={16} className={loading ? 'animate-spin mr-2' : 'mr-2'} />
                    Retry
                </Button>
            </div>
        );
    }

    if(!data) return null;

    const hasData = data.data && data.data.length > 0;

    return (
        <div className="space-y-6 animate-in fade-in duration-300">
            <div className="flex flex-col xl:flex-row justify-between items-start xl:items-center gap-4">
                <div>
                    <h2 className="text-3xl font-bold text-slate-900 dark:text-white">{t('history.title')}</h2>
                    <div className="flex items-center gap-2 mt-1">
                        <p className="text-slate-500 dark:text-slate-400">{t('history.subtitle')}</p>
                        <Badge color="green">Live Data</Badge>
                        {!hasData && <Badge color="yellow">Limited Data</Badge>}
                    </div>
                </div>
                <div className="flex flex-wrap items-center gap-3">
                    <Button variant="outline" className="hidden sm:flex"><Download size={16} className="mr-2"/> {t('common.export')}</Button>
                </div>
            </div>

            {/* Controls Toolbar */}
            <Card className="p-4 flex flex-col lg:flex-row gap-6 lg:items-end bg-white dark:bg-slate-900 shadow-lg border border-slate-200 dark:border-slate-800">
                {/* Range Selector */}
                <div className="flex-1 space-y-2 min-w-[200px]">
                    <label className="text-xs font-bold text-slate-500 dark:text-slate-400 uppercase tracking-wider">{t('history.dateRange')}</label>
                    <div className="relative">
                        <select 
                            value={rangeType} 
                            onChange={(e) => setRangeType(e.target.value)}
                            className="w-full appearance-none bg-slate-50 dark:bg-slate-800 border border-slate-200 dark:border-slate-700 text-slate-900 dark:text-white rounded-lg px-4 py-2.5 pr-8 focus:outline-none focus:ring-2 focus:ring-primary-500 transition-shadow font-medium"
                        >
                            {RANGES.map(r => <option key={r.value} value={r.value}>{r.label}</option>)}
                        </select>
                        <ChevronDown className="absolute right-3 top-3 text-slate-400 pointer-events-none" size={16} />
                    </div>
                </div>

                {/* Custom Date Inputs */}
                <div className={`flex gap-4 flex-1 transition-opacity duration-300 ${rangeType === 'custom' ? 'opacity-100' : 'opacity-50 pointer-events-none'}`}>
                    <div className="space-y-2 flex-1">
                        <label className="text-xs font-bold text-slate-500 dark:text-slate-400 uppercase tracking-wider">{t('history.startDate')}</label>
                        <input 
                            type="datetime-local" 
                            value={startDate}
                            onChange={(e) => { setStartDate(e.target.value); setRangeType('custom'); }}
                            className="w-full bg-slate-50 dark:bg-slate-800 border border-slate-200 dark:border-slate-700 text-slate-900 dark:text-white rounded-lg px-3 py-2.5 focus:outline-none focus:ring-2 focus:ring-primary-500 font-mono text-sm"
                        />
                    </div>
                    <div className="space-y-2 flex-1">
                        <label className="text-xs font-bold text-slate-500 dark:text-slate-400 uppercase tracking-wider">{t('history.endDate')}</label>
                        <input 
                            type="datetime-local" 
                            value={endDate}
                            onChange={(e) => { setEndDate(e.target.value); setRangeType('custom'); }}
                            className="w-full bg-slate-50 dark:bg-slate-800 border border-slate-200 dark:border-slate-700 text-slate-900 dark:text-white rounded-lg px-3 py-2.5 focus:outline-none focus:ring-2 focus:ring-primary-500 font-mono text-sm"
                        />
                    </div>
                </div>

                {/* Action Button */}
                <div className="flex items-center gap-2 pb-1">
                    <Button onClick={handleFetch} disabled={loading} className="h-[42px] px-6">
                        <RefreshCw size={18} className={`mr-2 ${loading ? 'animate-spin' : ''}`} />
                        {t('history.updateView')}
                    </Button>
                </div>
            </Card>

            {hasData ? (
                <>
                    <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
                         {Object.entries(data.statistics)
                             .filter(([_, val]) => val != null && typeof val === 'number')
                             .map(([key, val]) => (
                             <Card key={key} className="p-4 text-center hover:bg-slate-50 dark:hover:bg-slate-800/50 transition-colors">
                                 <div className="text-xs uppercase text-slate-500 dark:text-slate-400 font-bold mb-1 tracking-wider">{t(`history.${key}`) || key}</div>
                                 <div className="text-xl font-bold text-slate-800 dark:text-white font-mono">
                                    {(val as number).toLocaleString(undefined, { maximumFractionDigits: 1 })}
                                 </div>
                             </Card>
                         ))}
                    </div>

                    <Card className="p-6 border-slate-200 dark:border-slate-800">
                        <div className="flex flex-wrap justify-between items-center mb-6 gap-4">
                            <h3 className="font-bold text-lg text-slate-800 dark:text-white flex items-center gap-2">
                                <Filter size={20} className="text-primary-500" />
                                {t('history.visualization')}
                            </h3>
                            <div className="flex items-center gap-6 bg-slate-50 dark:bg-slate-800/50 p-2 rounded-lg border border-slate-200 dark:border-slate-800">
                                <Toggle label={t('history.showActual')} checked={showActual} onChange={setShowActual} />
                                <div className="w-px h-6 bg-slate-200 dark:bg-slate-700"></div>
                                <Toggle label={t('history.showForecast')} checked={showForecast} onChange={setShowForecast} />
                            </div>
                        </div>

                        {loading ? (
                            <div className="h-[500px] flex items-center justify-center text-slate-400">
                                <RefreshCw className="animate-spin mb-2" size={32} />
                                <span className="sr-only">Loading...</span>
                            </div>
                        ) : (
                            <HistoricalChart data={data} showActual={showActual} showForecast={showForecast} />
                        )}
                    </Card>

                    {/* Detailed Analysis Charts */}
                    <div className="grid md:grid-cols-2 gap-6">
                        <Card className="p-6 bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800">
                            <h4 className="font-semibold text-slate-900 dark:text-white mb-4 flex items-center gap-2">
                                <Calendar size={20} className="text-primary-500" />
                                {t('history.hourlyPatterns')}
                            </h4>
                            <HourlyPatternsChart data={data} />
                        </Card>
                        <Card className="p-6 bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800">
                            <h4 className="font-semibold text-slate-900 dark:text-white mb-4 flex items-center gap-2">
                                <Filter size={20} className="text-primary-500" />
                                {t('history.valueDist')}
                            </h4>
                            <ValueDistributionChart data={data} />
                        </Card>
                    </div>
                </>
            ) : (
                <Card className="p-10 text-center">
                    <Calendar size={48} className="mx-auto text-slate-300 dark:text-slate-600 mb-4" />
                    <h3 className="text-lg font-semibold text-slate-700 dark:text-slate-300 mb-2">No Data Available</h3>
                    <p className="text-slate-500 dark:text-slate-400 max-w-md mx-auto">
                        No forecast data found for the selected date range. Try selecting a shorter range or wait for more forecasts to be generated.
                    </p>
                </Card>
            )}
        </div>
    )
}
