
import React, { useState, useEffect } from 'react';
import { ModelType, AnomalyData } from '../types';
import { generateAnomalyData } from '../services/mockData';
import { AnomalyChart } from '../components/Charts';
import { Card } from '../components/ui';
import { AlertTriangle, Download, RefreshCw } from 'lucide-react';
import { useLanguage } from '../contexts/LanguageContext';

export const AnomalyView = ({ model }: { model: ModelType }) => {
    const [data, setData] = useState<AnomalyData | null>(null);
    const { t } = useLanguage();
    
    useEffect(() => {
        setData(generateAnomalyData(model));
    }, [model]);

    if(!data) return null;

    return (
        <div className="space-y-6">
            <h2 className="text-2xl font-bold text-slate-900 dark:text-white flex items-center gap-2">
                <AlertTriangle className="text-amber-500" />
                {t('anomaly.title')}
            </h2>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <Card className="p-6">
                    <div className="text-slate-500 dark:text-slate-400 text-xs font-semibold uppercase tracking-wider mb-2">{t('anomaly.rate')}</div>
                    <div className="text-3xl font-bold text-slate-300 dark:text-slate-600 opacity-50">--%</div>
                </Card>
                <Card className="p-6">
                    <div className="text-slate-500 dark:text-slate-400 text-xs font-semibold uppercase tracking-wider mb-2">{t('anomaly.count')}</div>
                    <div className="text-4xl font-bold text-red-600">{data.summary.anomalyCount}</div>
                </Card>
                 <Card className="p-6">
                    <div className="text-slate-500 dark:text-slate-400 text-xs font-semibold uppercase tracking-wider mb-2">{t('anomaly.maxResidual')}</div>
                    <div className="text-3xl font-bold text-slate-300 dark:text-slate-600 opacity-50">--</div>
                </Card>
            </div>

            <div className="grid lg:grid-cols-3 gap-6">
                <Card className="lg:col-span-2 p-6">
                    <div className="flex justify-between items-center mb-8">
                        <h3 className="font-semibold text-slate-900 dark:text-slate-200">{t('anomaly.scatter')}</h3>
                        <div className="flex gap-2 text-slate-500">
                            <RefreshCw size={16} className="cursor-pointer hover:text-slate-900 dark:hover:text-white" />
                            <Download size={16} className="cursor-pointer hover:text-slate-900 dark:hover:text-white" />
                        </div>
                    </div>
                    <AnomalyChart data={data} />
                </Card>
                <Card className="p-6">
                     <h3 className="font-semibold text-slate-900 dark:text-slate-200 mb-6">{t('anomaly.topAnomalies')}</h3>
                     <div className="overflow-y-auto max-h-[400px] space-y-3 pr-2 scrollbar-hide">
                         {data.anomalies.filter(a => a.isAnomaly).slice(0, 6).map((a, i) => (
                             <div key={i} className="flex justify-between items-center p-4 bg-red-50 dark:bg-red-900/10 rounded-md shadow-sm border border-red-100 dark:border-red-900/30">
                                 <div>
                                     <div className="text-[10px] text-red-600 dark:text-red-400 font-bold uppercase mb-1">{new Date(a.timestamp).toLocaleDateString()}</div>
                                     <div className="text-sm font-bold text-slate-900 dark:text-white">{t('anomaly.res')}: {a.residual.toFixed(1)}</div>
                                 </div>
                                 <div className="text-xs font-bold text-red-500">
                                     {(a.anomalyScore * 100).toFixed(1)}%
                                 </div>
                             </div>
                         ))}
                     </div>
                </Card>
            </div>
        </div>
    );
};
