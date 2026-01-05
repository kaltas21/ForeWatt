
import React, { useState, useEffect } from 'react';
import { ModelType, RealTimeData } from '../types';
import { fetchRealTimeData, generateSmartSummary } from '../services/api';
import { RealTimeChart } from '../components/Charts';
import { Card, Button, Toggle, Badge } from '../components/ui';
import { RefreshCw, Download, Copy, AlertCircle, Clock, Sparkles, WifiOff } from 'lucide-react';
import { useLanguage } from '../contexts/LanguageContext';

interface Props {
  model: ModelType;
  onDataUpdate: (data: RealTimeData) => void;
}

export const RealTimeView: React.FC<Props> = ({ model, onDataUpdate }) => {
  const { t } = useLanguage();
  const [data, setData] = useState<RealTimeData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [showCI, setShowCI] = useState(true);
  const [lastRefreshed, setLastRefreshed] = useState(new Date());
  const [summary, setSummary] = useState('');

  const fetchData = async () => {
    setLoading(true);
    setError(null);
    try {
      // Backend now returns 12 hours of real EPIAS actuals + 12 hours of forecasts
      const newData = await fetchRealTimeData(model);

      setData(newData);
      setSummary(await generateSmartSummary(newData));
      onDataUpdate(newData);
    } catch (err) {
      console.error('API error:', err);
      setError(err instanceof Error ? err.message : 'Failed to fetch data');
    }
    setLastRefreshed(new Date());
    setLoading(false);
  };

  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 60000);
    return () => clearInterval(interval);
  }, [model]);

  if (loading && !data) {
    return <div className="p-10 flex justify-center h-full items-center"><RefreshCw className="animate-spin text-primary-500 w-10 h-10" /></div>;
  }

  if (error && !data) {
    return (
      <div className="p-10 flex flex-col justify-center h-full items-center gap-4">
        <WifiOff className="text-red-500 w-12 h-12" />
        <p className="text-red-600 dark:text-red-400 font-medium">{error}</p>
        <Button onClick={fetchData} disabled={loading}>
          <RefreshCw size={16} className={loading ? 'animate-spin mr-2' : 'mr-2'} />
          Retry
        </Button>
      </div>
    );
  }

  if (!data) return null;

  return (
    <div className="space-y-6 animate-in fade-in duration-500">

      {/* AI Summary Card */}
      <Card className="bg-gradient-to-r from-indigo-50 to-blue-50 dark:from-indigo-950/30 dark:to-blue-950/30 border-indigo-100 dark:border-indigo-900/50 p-5 flex items-start gap-4 shadow-sm">
         <div className="p-2 bg-white dark:bg-indigo-900/50 rounded-lg shadow-sm text-indigo-500">
             <Sparkles size={24} />
         </div>
         <div className="flex-1">
             <h3 className="text-sm font-bold text-indigo-900 dark:text-indigo-200 uppercase tracking-wide mb-1">{t('realTime.aiInsight')}</h3>
             <p className="text-slate-700 dark:text-slate-300 font-medium leading-relaxed">{summary}</p>
         </div>
      </Card>

      {/* Header Info */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h2 className="text-3xl font-bold text-slate-900 dark:text-white tracking-tight">{t('realTime.title')}</h2>
          <div className="flex items-center gap-2 text-sm text-slate-500 dark:text-slate-400 mt-1">
            <Badge color={model === 'price' ? 'green' : 'blue'}>{t('common.liveConnection')}</Badge>
            <span>•</span>
            <span>{t('common.lastUpdated')}: {lastRefreshed.toLocaleTimeString('en-GB', { hour12: false })}</span>
          </div>
        </div>
        <div className="flex items-center gap-3">
             <div className="bg-amber-50 dark:bg-amber-900/20 border border-amber-200 dark:border-amber-800 text-amber-800 dark:text-amber-300 px-3 py-1.5 rounded-md text-xs flex items-center gap-2">
                <AlertCircle size={14} />
                <span>{t('common.epiasDelay')}</span>
             </div>
             <Button variant="outline" onClick={fetchData} disabled={loading} className="w-10 px-0">
                <RefreshCw size={18} className={loading ? 'animate-spin' : ''} />
             </Button>
        </div>
      </div>

      {/* KPI Cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <Card className="p-4 relative overflow-hidden group">
            <div className="absolute top-0 right-0 p-3 opacity-10 group-hover:opacity-20 transition-opacity">
                <Clock size={40} className="text-slate-900 dark:text-slate-100" />
            </div>
            <p className="text-xs font-bold text-slate-500 dark:text-slate-400 uppercase tracking-wider">{t('realTime.avgActual')}</p>
            <div className="mt-2 flex items-baseline gap-2">
                <span className="text-2xl font-bold text-slate-900 dark:text-white">{Math.round(data.summary.avgActual).toLocaleString()}</span>
                <span className="text-xs text-slate-400">{data.unit}</span>
            </div>
        </Card>
        <Card className="p-4 relative overflow-hidden group">
            <p className="text-xs font-bold text-slate-500 dark:text-slate-400 uppercase tracking-wider">{t('realTime.avgForecast')}</p>
            <div className="mt-2 flex items-baseline gap-2">
                <span className="text-2xl font-bold text-slate-900 dark:text-white">{Math.round(data.summary.avgForecast).toLocaleString()}</span>
                <span className="text-xs text-slate-400">{data.unit}</span>
            </div>
        </Card>
        <Card className="p-4">
            <p className="text-xs font-bold text-slate-500 dark:text-slate-400 uppercase tracking-wider">{t('realTime.peakActual')}</p>
            <div className="mt-2">
                <div className="flex items-baseline gap-2">
                    <span className="text-2xl font-bold text-slate-900 dark:text-white">{Math.round(data.summary.peakActual.value).toLocaleString()}</span>
                    <span className="text-xs text-slate-400">{data.unit}</span>
                </div>
                <div className="text-xs text-slate-500 dark:text-slate-400 mt-1 flex items-center gap-1 font-mono">
                    <Clock size={12} /> {data.summary.peakActual.time}
                </div>
            </div>
        </Card>
        <Card className="p-4 bg-primary-50/50 dark:bg-primary-900/20 border-primary-100 dark:border-primary-800">
            <p className="text-xs font-bold text-primary-700 dark:text-primary-300 uppercase tracking-wider">{t('realTime.peakForecast')}</p>
            <div className="mt-2">
                <div className="flex items-baseline gap-2">
                    <span className="text-2xl font-bold text-primary-900 dark:text-primary-100">{Math.round(data.summary.peakForecast.value).toLocaleString()}</span>
                    <span className="text-xs text-primary-600 dark:text-primary-300">{data.unit}</span>
                </div>
                <div className="text-xs text-primary-600 dark:text-primary-300 mt-1 flex items-center gap-1 font-mono">
                    <Clock size={12} /> {data.summary.peakForecast.time}
                </div>
            </div>
        </Card>
      </div>

      {/* Main Chart */}
      <Card className="p-6">
        <div className="flex justify-between items-center mb-6">
            <h3 className="font-semibold text-slate-800 dark:text-slate-200">{t('realTime.horizon')}</h3>
            <Toggle label={t('realTime.confidence')} checked={showCI} onChange={setShowCI} />
        </div>
        <RealTimeChart data={data} showCI={showCI} />
      </Card>

      {/* Data Table */}
      <Card className="overflow-hidden">
          <div className="p-4 border-b border-slate-100 dark:border-slate-800 flex justify-between items-center bg-slate-50/50 dark:bg-slate-800/50">
              <h3 className="font-semibold text-slate-800 dark:text-slate-200">{t('realTime.tabularData')}</h3>
              <div className="flex gap-2">
                  <Button variant="ghost" className="h-8 text-xs"><Copy size={14} className="mr-1" /> {t('common.copy')}</Button>
                  <Button variant="outline" className="h-8 text-xs"><Download size={14} className="mr-1" /> CSV</Button>
              </div>
          </div>
          <div className="overflow-x-auto">
              <table className="w-full text-sm text-left">
                  <thead className="bg-slate-50 dark:bg-slate-900/50 text-slate-500 dark:text-slate-400 font-medium border-b border-slate-200 dark:border-slate-800">
                      <tr>
                          <th className="px-6 py-3">{t('realTime.time')}</th>
                          <th className="px-6 py-3">{t('realTime.type')}</th>
                          <th className="px-6 py-3 text-right">{t('realTime.value')} ({data.unit})</th>
                          <th className="px-6 py-3 text-right">{t('realTime.lowerCI')}</th>
                          <th className="px-6 py-3 text-right">{t('realTime.upperCI')}</th>
                      </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-100 dark:divide-slate-800">
                      {/* Combine and sort for table */}
                      {[...data.actual.map(d => ({...d, type: 'Actual'})), ...data.forecast.map(d => ({...d, type: 'Forecast'}))]
                        .sort((a,b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime())
                        .map((row, idx) => (
                          <tr key={idx} className={`hover:bg-slate-50 dark:hover:bg-slate-800/50 transition-colors ${row.type === 'Forecast' ? 'bg-primary-50/10 dark:bg-primary-900/5' : ''}`}>
                              <td className="px-6 py-3 text-slate-600 dark:text-slate-300 font-mono">{new Date(row.timestamp).toLocaleTimeString('en-GB', {hour: '2-digit', minute:'2-digit', hour12: false})}</td>
                              <td className="px-6 py-3">
                                  <span className={`inline-flex items-center px-2 py-0.5 rounded text-xs font-medium ${row.type === 'Actual' ? 'bg-slate-100 dark:bg-slate-800 text-slate-800 dark:text-slate-300' : 'bg-blue-100 dark:bg-blue-900/40 text-blue-800 dark:text-blue-300'}`}>
                                      {row.type}
                                  </span>
                              </td>
                              <td className="px-6 py-3 text-right font-medium text-slate-900 dark:text-slate-100">{row.value.toFixed(2)}</td>
                              <td className="px-6 py-3 text-right text-slate-400">{row.lower ? row.lower.toFixed(2) : '-'}</td>
                              <td className="px-6 py-3 text-right text-slate-400">{row.upper ? row.upper.toFixed(2) : '-'}</td>
                          </tr>
                      ))}
                  </tbody>
              </table>
          </div>
      </Card>
    </div>
  );
};
