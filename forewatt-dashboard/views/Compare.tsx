
import React, { useState, useEffect } from 'react';
import { ModelType, ComparisonData } from '../types';
import { generateComparisonData } from '../services/mockData';
import { ComparisonChart } from '../components/Charts';
import { Card, Button } from '../components/ui';
import { ArrowLeftRight, Calendar, TrendingUp, TrendingDown, Percent, Activity } from 'lucide-react';
import { useLanguage } from '../contexts/LanguageContext';

export const CompareView = ({ model }: { model: ModelType }) => {
    const { t } = useLanguage();
    const [data, setData] = useState<ComparisonData | null>(null);
    const [preset, setPreset] = useState('day-over-day');

    useEffect(() => {
        // Fetch new data based on preset
        const labelA = 'This Period';
        const labelB = preset === 'day-over-day' ? 'Yesterday' 
                     : preset === 'week-over-week' ? 'Last Week' 
                     : 'Last Month';
        
        setData(generateComparisonData(model, labelA, labelB));
    }, [model, preset]);

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
                    <h4 className="font-semibold text-slate-900 dark:text-white mb-4">{t('compare.dayType')}</h4>
                    <p className="text-sm text-slate-500 mb-4">Comparing typical weekday profiles vs weekend profiles shows a 15% drop in consumption.</p>
                    <div className="h-40 bg-slate-50 dark:bg-slate-800/50 rounded-lg flex items-center justify-center border border-dashed border-slate-200 dark:border-slate-700">
                        <span className="text-slate-400 text-sm">Pattern Chart Placeholder</span>
                    </div>
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
