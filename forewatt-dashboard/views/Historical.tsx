
import React, { useState, useEffect } from 'react';
import { ModelType, HistoricalData } from '../types';
import { generateHistoricalData } from '../services/mockData';
import { HistoricalChart } from '../components/Charts';
import { Card, Button, Toggle } from '../components/ui';
import { Calendar, Download, RefreshCw, Filter, ChevronDown } from 'lucide-react';
import { useLanguage } from '../contexts/LanguageContext';

const RANGES = [
    { label: 'Last 3 Days', value: '3d', days: 3 },
    { label: 'Last 7 Days', value: '7d', days: 7 },
    { label: 'Last 15 Days', value: '15d', days: 15 },
    { label: 'Last 1 Month', value: '1m', days: 30 },
    { label: 'Last 3 Months', value: '3m', days: 90 },
    { label: 'Last 6 Months', value: '6m', days: 180 },
    { label: 'Last 1 Year', value: '1y', days: 365 },
    { label: 'Custom Range', value: 'custom', days: 0 }
];

export const HistoricalView = ({ model }: { model: ModelType }) => {
    const { t } = useLanguage();
    // State
    const [data, setData] = useState<HistoricalData | null>(null);
    const [loading, setLoading] = useState(false);
    
    // Filters
    const [rangeType, setRangeType] = useState('7d');
    const [startDate, setStartDate] = useState('');
    const [endDate, setEndDate] = useState('');
    const [showActual, setShowActual] = useState(true);
    const [showForecast, setShowForecast] = useState(true);

    // Initial load and range changes
    useEffect(() => {
        handleFetch();
    }, [model, rangeType]); // Re-fetch when model or preset changes

    // Update custom inputs when preset selected (for visual consistency)
    useEffect(() => {
        const selectedRange = RANGES.find(r => r.value === rangeType);
        if (selectedRange && selectedRange.value !== 'custom') {
            const end = new Date();
            const start = new Date();
            start.setDate(end.getDate() - selectedRange.days);
            
            // Format for datetime-local input: YYYY-MM-DDThh:mm
            setEndDate(end.toISOString().slice(0, 16));
            setStartDate(start.toISOString().slice(0, 16));
        }
    }, [rangeType]);

    const handleFetch = () => {
        setLoading(true);
        // Simulate network request
        setTimeout(() => {
            let start: Date, end: Date;
            
            if (rangeType === 'custom') {
                start = startDate ? new Date(startDate) : new Date(Date.now() - 7 * 24 * 60 * 60 * 1000);
                end = endDate ? new Date(endDate) : new Date();
            } else {
                const range = RANGES.find(r => r.value === rangeType)!;
                end = new Date();
                start = new Date();
                start.setDate(end.getDate() - range.days);
            }

            const newData = generateHistoricalData(model, start, end);
            setData(newData);
            setLoading(false);
        }, 500);
    };

    if(!data) return null;

    return (
        <div className="space-y-6 animate-in fade-in duration-300">
            <div className="flex flex-col xl:flex-row justify-between items-start xl:items-center gap-4">
                <div>
                    <h2 className="text-3xl font-bold text-slate-900 dark:text-white">{t('history.title')}</h2>
                    <p className="text-slate-500 dark:text-slate-400 mt-1">{t('history.subtitle')}</p>
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

            <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
                 {Object.entries(data.statistics).map(([key, val]) => (
                     <Card key={key} className="p-4 text-center hover:bg-slate-50 dark:hover:bg-slate-800/50 transition-colors">
                         <div className="text-xs uppercase text-slate-500 dark:text-slate-400 font-bold mb-1 tracking-wider">{t(`history.${key}`) || key}</div>
                         <div className="text-xl font-bold text-slate-800 dark:text-white font-mono">
                            {val.toLocaleString(undefined, { maximumFractionDigits: 1 })}
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

            {/* Detailed Analysis Placeholders */}
            <div className="grid md:grid-cols-2 gap-6">
                <Card className="p-6 h-[350px] flex flex-col items-center justify-center bg-white dark:bg-slate-900 border-dashed border-2 border-slate-200 dark:border-slate-800">
                    <div className="w-16 h-16 bg-slate-100 dark:bg-slate-800 rounded-full flex items-center justify-center mb-4">
                         <Calendar size={32} className="text-slate-400" />
                    </div>
                    <h4 className="font-semibold text-slate-900 dark:text-white mb-2">{t('history.hourlyPatterns')}</h4>
                    <p className="text-slate-500 text-sm text-center max-w-xs">Average consumption/price distributed by hour of day for the selected period.</p>
                </Card>
                <Card className="p-6 h-[350px] flex flex-col items-center justify-center bg-white dark:bg-slate-900 border-dashed border-2 border-slate-200 dark:border-slate-800">
                    <div className="w-16 h-16 bg-slate-100 dark:bg-slate-800 rounded-full flex items-center justify-center mb-4">
                         <Filter size={32} className="text-slate-400" />
                    </div>
                    <h4 className="font-semibold text-slate-900 dark:text-white mb-2">{t('history.valueDist')}</h4>
                    <p className="text-slate-500 text-sm text-center max-w-xs">Histogram and percentile analysis of the selected data range.</p>
                </Card>
            </div>
        </div>
    )
}
