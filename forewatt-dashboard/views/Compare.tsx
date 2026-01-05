
import React, { useState, useEffect } from 'react';
import { ModelType, ComparisonData } from '../types';
import { fetchComparisonData, fetchDayTypeComparison } from '../services/api';
import { ComparisonChart, DayTypeComparisonChart } from '../components/Charts';
import { Card, Button, Badge } from '../components/ui';
import { ArrowLeftRight, Calendar, TrendingUp, TrendingDown, Percent, Activity, RefreshCw } from 'lucide-react';
import { useLanguage } from '../contexts/LanguageContext';

interface DayTypeData {
    weekday: { label: string; value: number }[];
    weekend: { label: string; value: number }[];
    diffPercent: number;
}

// Helper to check if data is valid for rendering
const isValidDayTypeData = (data: DayTypeData | null): data is DayTypeData => {
    return data !== null &&
           Array.isArray(data.weekday) &&
           data.weekday.length > 0 &&
           data.weekday.some(d => d.value > 0);
};

export const CompareView = ({ model }: { model: ModelType }) => {
    const { t } = useLanguage();
    const [data, setData] = useState<ComparisonData | null>(null);
    const [dayTypeData, setDayTypeData] = useState<DayTypeData | null>(null);
    const [dayTypeError, setDayTypeError] = useState<string | null>(null);
    const [preset, setPreset] = useState('day-over-day');
    const [loading, setLoading] = useState(false);
    const [dayTypeLoading, setDayTypeLoading] = useState(false);

    useEffect(() => {
        const loadData = async () => {
            setLoading(true);
            setDayTypeError(null);
            try {
                const now = new Date();
                let periodA, periodB;

                if (preset === 'day-over-day') {
                    periodA = {
                        start: new Date(now.getTime() - 24 * 60 * 60 * 1000),
                        end: now,
                        label: 'Today'
                    };
                    periodB = {
                        start: new Date(now.getTime() - 48 * 60 * 60 * 1000),
                        end: new Date(now.getTime() - 24 * 60 * 60 * 1000),
                        label: 'Yesterday'
                    };
                } else if (preset === 'week-over-week') {
                    periodA = {
                        start: new Date(now.getTime() - 7 * 24 * 60 * 60 * 1000),
                        end: now,
                        label: 'This Week'
                    };
                    periodB = {
                        start: new Date(now.getTime() - 14 * 24 * 60 * 60 * 1000),
                        end: new Date(now.getTime() - 7 * 24 * 60 * 60 * 1000),
                        label: 'Last Week'
                    };
                } else {
                    periodA = {
                        start: new Date(now.getTime() - 30 * 24 * 60 * 60 * 1000),
                        end: now,
                        label: 'This Month'
                    };
                    periodB = {
                        start: new Date(now.getTime() - 60 * 24 * 60 * 60 * 1000),
                        end: new Date(now.getTime() - 30 * 24 * 60 * 60 * 1000),
                        label: 'Last Month'
                    };
                }

                // Load comparison data first (faster)
                const comparisonResult = await fetchComparisonData(model, periodA, periodB);
                setData(comparisonResult);
                setLoading(false);

                // Load day type data separately with its own loading state
                setDayTypeLoading(true);
                try {
                    const dayTypeResult = await fetchDayTypeComparison(model, 30);
                    console.log('Day type data loaded:', dayTypeResult);
                    setDayTypeData(dayTypeResult);
                } catch (dayTypeErr) {
                    console.error('Failed to fetch day type data:', dayTypeErr);
                    setDayTypeError(dayTypeErr instanceof Error ? dayTypeErr.message : 'Failed to load day type data');
                }
                setDayTypeLoading(false);

            } catch (error) {
                console.error('Failed to fetch comparison data:', error);
                setLoading(false);
            }
        };

        loadData();
    }, [model, preset]);

    if (loading && !data) {
        return (
            <div className="p-10 flex justify-center h-full items-center">
                <RefreshCw className="animate-spin text-primary-500 w-10 h-10" />
            </div>
        );
    }

    if (!data) return null;

    const MetricCard = ({ label, value, subtext, icon: Icon, positive }: any) => (
        <Card className="p-5 flex items-center justify-between hover:bg-slate-50 dark:hover:bg-slate-800/50 transition-colors">
            <div>
                <p className="text-xs font-bold text-slate-500 dark:text-slate-400 uppercase tracking-wider mb-1">{label}</p>
                <div className="flex items-center gap-2">
                    <span className="text-2xl font-bold text-slate-900 dark:text-white">{value}</span>
                    <span className={`text-xs font-bold px-1.5 py-0.5 rounded ${positive ? 'bg-green-100 text-green-700' : 'bg-red-100 text-red-700'}`}>
                        {subtext}
                    </span>
                </div>
            </div>
            <div className="p-3 bg-slate-100 dark:bg-slate-800 rounded-full text-slate-500">
                <Icon size={20} />
            </div>
        </Card>
    );

    return (
        <div className="space-y-6 animate-in fade-in duration-300">
            <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
                <div>
                    <h2 className="text-3xl font-bold text-slate-900 dark:text-white flex items-center gap-3">
                        <ArrowLeftRight className="text-primary-500" />
                        {t('compare.title')}
                    </h2>
                    <p className="text-slate-500 dark:text-slate-400 mt-1">{t('compare.subtitle')}</p>
                </div>
                
                <div className="bg-white dark:bg-slate-900 p-1.5 rounded-xl border border-slate-200 dark:border-slate-800 flex gap-1 shadow-sm">
                    {['day-over-day', 'week-over-week', 'month-over-month'].map(p => (
                        <button
                            key={p}
                            onClick={() => setPreset(p)}
                            className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                                preset === p 
                                ? 'bg-primary-600 text-white shadow-md' 
                                : 'text-slate-600 dark:text-slate-400 hover:bg-slate-100 dark:hover:bg-slate-800'
                            }`}
                        >
                            {p.split('-').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ')}
                        </button>
                    ))}
                </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                <MetricCard 
                    label={t('compare.meanDiff')}
                    value={data.metrics.diffMean.toFixed(1)} 
                    subtext="+5.2%"
                    icon={Activity}
                    positive={true}
                />
                <MetricCard 
                    label={t('compare.peakDiff')}
                    value={data.metrics.diffPeak.toFixed(1)} 
                    subtext="+1.8%"
                    icon={TrendingUp}
                    positive={false}
                />
                <MetricCard 
                    label={`${t('compare.volatility')} A`}
                    value={data.metrics.volatilityA.toFixed(0)} 
                    subtext="Baseline"
                    icon={Percent}
                    positive={true}
                />
                <MetricCard 
                    label={`${t('compare.volatility')} B`}
                    value={data.metrics.volatilityB.toFixed(0)} 
                    subtext="Comparison"
                    icon={Percent}
                    positive={true}
                />
            </div>

            <Card className="p-6">
                <div className="flex justify-between items-center mb-6">
                    <h3 className="font-semibold text-slate-800 dark:text-slate-200">{t('compare.overlay')}</h3>
                    <div className="flex gap-4 text-sm">
                        <div className="flex items-center gap-2">
                            <span className="w-3 h-3 rounded-full bg-primary-500"></span>
                            <span className="text-slate-600 dark:text-slate-400">{data.periodA.label}</span>
                        </div>
                        <div className="flex items-center gap-2">
                            <span className="w-3 h-3 rounded-full bg-slate-400"></span>
                            <span className="text-slate-600 dark:text-slate-400">{data.periodB.label}</span>
                        </div>
                    </div>
                </div>
                <ComparisonChart data={data} />
            </Card>

            <div className="grid md:grid-cols-2 gap-6">
                <Card className="p-6">
                    <div className="flex items-center justify-between mb-4">
                        <h4 className="font-semibold text-slate-900 dark:text-white">{t('compare.dayType')}</h4>
                        {isValidDayTypeData(dayTypeData) && (
                            <Badge color={dayTypeData.diffPercent > 0 ? 'green' : 'blue'}>
                                {dayTypeData.diffPercent > 0 ? '+' : ''}{dayTypeData.diffPercent.toFixed(1)}% weekday
                            </Badge>
                        )}
                    </div>
                    <p className="text-sm text-slate-500 mb-4">
                        {isValidDayTypeData(dayTypeData)
                            ? `Weekday average is ${Math.abs(dayTypeData.diffPercent).toFixed(1)}% ${dayTypeData.diffPercent > 0 ? 'higher' : 'lower'} than weekend average.`
                            : dayTypeError
                            ? 'Failed to load day type comparison data.'
                            : 'Comparing typical weekday profiles vs weekend profiles.'
                        }
                    </p>
                    {dayTypeLoading ? (
                        <div className="h-40 bg-slate-50 dark:bg-slate-800/50 rounded-lg flex items-center justify-center border border-dashed border-slate-200 dark:border-slate-700">
                            <RefreshCw className="animate-spin text-slate-400" size={20} />
                        </div>
                    ) : dayTypeError ? (
                        <div className="h-40 bg-red-50 dark:bg-red-900/20 rounded-lg flex flex-col items-center justify-center border border-dashed border-red-200 dark:border-red-800 text-red-600 dark:text-red-400">
                            <p className="text-sm font-medium">Error loading data</p>
                            <p className="text-xs mt-1">{dayTypeError}</p>
                        </div>
                    ) : isValidDayTypeData(dayTypeData) ? (
                        <DayTypeComparisonChart data={dayTypeData} />
                    ) : (
                        <div className="h-40 bg-amber-50 dark:bg-amber-900/20 rounded-lg flex items-center justify-center border border-dashed border-amber-200 dark:border-amber-800 text-amber-600 dark:text-amber-400">
                            <p className="text-sm">No pattern data available for this period</p>
                        </div>
                    )}
                </Card>
                <Card className="p-6">
                    <h4 className="font-semibold text-slate-900 dark:text-white mb-4">{t('compare.stats')}</h4>
                    <div className="space-y-3">
                        {[
                            { label: t('history.min') + ' Value', a: '1,240', b: '1,180', diff: '+5.1%' },
                            { label: t('history.max') + ' Value', a: '2,450', b: '2,310', diff: '+6.0%' },
                            { label: t('history.std'), a: '150', b: '180', diff: '-16.6%' },
                        ].map((row, i) => (
                            <div key={i} className="flex justify-between items-center text-sm py-2 border-b border-slate-100 dark:border-slate-800 last:border-0">
                                <span className="text-slate-500 dark:text-slate-400">{row.label}</span>
                                <div className="flex gap-4 font-mono">
                                    <span className="text-slate-900 dark:text-white">{row.a}</span>
                                    <span className="text-slate-400">{row.b}</span>
                                    <span className="text-green-600 dark:text-green-400 font-bold">{row.diff}</span>
                                </div>
                            </div>
                        ))}
                    </div>
                </Card>
            </div>
        </div>
    );
};
