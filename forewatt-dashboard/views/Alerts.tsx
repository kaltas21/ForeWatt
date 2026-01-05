
import React, { useState, useEffect } from 'react';
import { checkAlertThresholds, fetchDataStatus } from '../services/api';
import { Alert } from '../types';
import { Card, Button, Toggle, Badge } from '../components/ui';
import {
    Bell, AlertTriangle, CheckCircle, Info, Trash2,
    TrendingUp, TrendingDown, Activity, Zap, BarChart3,
    Edit2, Save, X, RefreshCw, Database
} from 'lucide-react';
import { useLanguage } from '../contexts/LanguageContext';

interface AlertConfig {
    id: string;
    title: string;
    description: string;
    enabled: boolean;
    threshold: number;
    unit: string;
    condition: 'above' | 'below';
    severity: 'critical' | 'warning' | 'info';
    model: 'price' | 'consumption';
    icon: any;
}

export const AlertsView = () => {
    const { t } = useLanguage();
    const [activeTab, setActiveTab] = useState<'history' | 'config'>('config');
    const [alerts, setAlerts] = useState<Alert[]>([]);
    const [loading, setLoading] = useState(false);
    const [dataStatus, setDataStatus] = useState<{ lastDate: string | null }>({ lastDate: null });
    
    // Config State - with model field for actual data checking
    const [configs, setConfigs] = useState<AlertConfig[]>([
        {
            id: 'price-high',
            title: 'Price Spike Alert',
            description: 'Triggered when the Day-Ahead Market (PTF) price exceeds the safety threshold.',
            enabled: true,
            threshold: 2200,
            unit: 'TL/MWh',
            condition: 'above',
            severity: 'critical',
            model: 'price',
            icon: TrendingUp
        },
        {
            id: 'price-low',
            title: 'Negative/Low Price',
            description: 'Triggered when market prices drop below operational efficiency levels.',
            enabled: false,
            threshold: 100,
            unit: 'TL/MWh',
            condition: 'below',
            severity: 'warning',
            model: 'price',
            icon: TrendingDown
        },
        {
            id: 'consumption-surge',
            title: 'Demand Surge',
            description: 'Detects unexpected spikes in nationwide electricity consumption.',
            enabled: true,
            threshold: 48000,
            unit: 'MWh',
            condition: 'above',
            severity: 'critical',
            model: 'consumption',
            icon: Zap
        },
        {
            id: 'consumption-low',
            title: 'Low Consumption',
            description: 'Triggered when consumption drops below expected minimum levels.',
            enabled: false,
            threshold: 25000,
            unit: 'MWh',
            condition: 'below',
            severity: 'warning',
            model: 'consumption',
            icon: TrendingDown
        }
    ]);

    // Load data status and check alerts on mount
    useEffect(() => {
        const loadDataStatus = async () => {
            try {
                const status = await fetchDataStatus();
                setDataStatus({ lastDate: status.last_timestamp });
            } catch (err) {
                console.warn('Failed to fetch data status:', err);
            }
        };
        loadDataStatus();
    }, []);

    // Check alerts against real data
    const checkAlerts = async () => {
        setLoading(true);
        try {
            const triggeredAlerts = await checkAlertThresholds(configs);
            // Limit to most recent 50 alerts to avoid UI overload
            setAlerts(triggeredAlerts.slice(0, 50));
        } catch (err) {
            console.error('Failed to check alerts:', err);
        }
        setLoading(false);
    };

    // Auto-check alerts when configs change or on mount
    useEffect(() => {
        if (activeTab === 'history') {
            checkAlerts();
        }
    }, [activeTab]);

    // Editing State for Thresholds
    const [editingId, setEditingId] = useState<string | null>(null);
    const [editValue, setEditValue] = useState<string>('');

    const toggleConfig = (id: string) => {
        setConfigs(configs.map(c => c.id === id ? { ...c, enabled: !c.enabled } : c));
    };

    const startEdit = (config: AlertConfig) => {
        setEditingId(config.id);
        setEditValue(config.threshold.toString());
    };

    const saveEdit = (id: string) => {
        const val = parseFloat(editValue);
        if (!isNaN(val)) {
            setConfigs(configs.map(c => c.id === id ? { ...c, threshold: val } : c));
        }
        setEditingId(null);
    };

    const cancelEdit = () => {
        setEditingId(null);
    };

    // History Actions
    const markAsRead = (id: string) => {
        setAlerts(alerts.map(a => a.id === id ? { ...a, read: true } : a));
    };

    const deleteAlert = (id: string) => {
        setAlerts(alerts.filter(a => a.id !== id));
    };

    const getSeverityColor = (severity: string) => {
        switch (severity) {
            case 'critical': return 'text-red-600 bg-red-100 dark:bg-red-900/30 dark:text-red-400 border-red-200 dark:border-red-800';
            case 'warning': return 'text-amber-600 bg-amber-100 dark:bg-amber-900/30 dark:text-amber-400 border-amber-200 dark:border-amber-800';
            case 'info': return 'text-blue-600 bg-blue-100 dark:bg-blue-900/30 dark:text-blue-400 border-blue-200 dark:border-blue-800';
            default: return 'text-slate-600 bg-slate-100 dark:bg-slate-800 dark:text-slate-400';
        }
    };

    return (
        <div className="space-y-8 animate-in fade-in duration-300 max-w-6xl mx-auto">
            
            {/* Header */}
            <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-6">
                <div>
                    <h2 className="text-3xl font-bold text-slate-900 dark:text-white flex items-center gap-3">
                        <Bell className="text-primary-500" />
                        {t('alerts.title')}
                    </h2>
                    <p className="text-slate-500 dark:text-slate-400 mt-2 text-lg">
                        {t('alerts.subtitle')}
                    </p>
                </div>
                
                <div className="bg-slate-100 dark:bg-slate-900/50 p-1.5 rounded-xl border border-slate-200 dark:border-slate-800 flex gap-1 shadow-inner">
                    <button
                        onClick={() => setActiveTab('config')}
                        className={`px-6 py-2.5 rounded-lg text-sm font-bold transition-all ${
                            activeTab === 'config' 
                            ? 'bg-white dark:bg-slate-800 text-primary-600 dark:text-primary-400 shadow-md transform scale-100' 
                            : 'text-slate-500 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-200'
                        }`}
                    >
                        {t('alerts.config')}
                    </button>
                    <button
                        onClick={() => setActiveTab('history')}
                        className={`px-6 py-2.5 rounded-lg text-sm font-bold transition-all ${
                            activeTab === 'history' 
                            ? 'bg-white dark:bg-slate-800 text-primary-600 dark:text-primary-400 shadow-md transform scale-100' 
                            : 'text-slate-500 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-200'
                        }`}
                    >
                        {t('alerts.history')}
                        {alerts.filter(a => !a.read).length > 0 && (
                            <span className="ml-2 bg-red-500 text-white text-[10px] px-1.5 py-0.5 rounded-full">
                                {alerts.filter(a => !a.read).length}
                            </span>
                        )}
                    </button>
                </div>
            </div>

            {activeTab === 'config' ? (
                <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-6">
                    {configs.map((config) => (
                        <Card 
                            key={config.id} 
                            className={`flex flex-col h-full border-2 transition-all duration-300 ${
                                config.enabled 
                                ? 'border-slate-200 dark:border-slate-800 hover:border-primary-300 dark:hover:border-primary-700 hover:shadow-lg' 
                                : 'border-slate-100 dark:border-slate-800 opacity-75 grayscale-[0.5]'
                            }`}
                        >
                            <div className="p-6 flex flex-col h-full">
                                {/* Card Header */}
                                <div className="flex justify-between items-start mb-4">
                                    <div className={`p-3 rounded-xl ${config.enabled ? 'bg-primary-50 dark:bg-primary-900/20 text-primary-600 dark:text-primary-400' : 'bg-slate-100 dark:bg-slate-800 text-slate-400'}`}>
                                        <config.icon size={24} />
                                    </div>
                                    <Toggle label={config.enabled ? t('common.on') : t('common.off')} checked={config.enabled} onChange={() => toggleConfig(config.id)} />
                                </div>
                                
                                <div className="mb-6 flex-1">
                                    <div className="flex items-center gap-2 mb-2">
                                        <h3 className="font-bold text-lg text-slate-900 dark:text-white">{config.title}</h3>
                                        <span className={`text-[10px] font-bold uppercase px-2 py-0.5 rounded border ${getSeverityColor(config.severity)}`}>
                                            {config.severity}
                                        </span>
                                    </div>
                                    <p className="text-sm text-slate-500 dark:text-slate-400 leading-relaxed">
                                        {config.description}
                                    </p>
                                </div>

                                {/* Controls */}
                                <div className={`pt-6 border-t border-slate-100 dark:border-slate-800 ${!config.enabled ? 'opacity-50 pointer-events-none' : ''}`}>
                                    <div className="flex items-center justify-between">
                                        <div>
                                            <div className="text-[10px] font-bold text-slate-400 uppercase tracking-wider mb-1">{t('alerts.triggerCondition')}</div>
                                            <div className="flex items-center gap-2 font-mono text-sm font-medium text-slate-700 dark:text-slate-300">
                                                <span>Value</span>
                                                <span className="text-primary-500 font-bold">{config.condition === 'above' ? '>' : '<'}</span>
                                                
                                                {editingId === config.id ? (
                                                    <div className="flex items-center gap-1">
                                                        <input 
                                                            type="number" 
                                                            className="w-20 bg-white dark:bg-slate-900 border border-primary-500 rounded px-1.5 py-0.5 text-center focus:outline-none"
                                                            value={editValue}
                                                            onChange={(e) => setEditValue(e.target.value)}
                                                            autoFocus
                                                        />
                                                        <button onClick={() => saveEdit(config.id)} className="p-1 hover:bg-green-100 text-green-600 rounded"><CheckCircle size={14}/></button>
                                                        <button onClick={cancelEdit} className="p-1 hover:bg-red-100 text-red-600 rounded"><X size={14}/></button>
                                                    </div>
                                                ) : (
                                                    <div className="flex items-center gap-2 group cursor-pointer hover:text-primary-600" onClick={() => startEdit(config)}>
                                                        <span className="text-lg font-bold text-slate-900 dark:text-white">{config.threshold.toLocaleString()}</span>
                                                        <Edit2 size={12} className="opacity-0 group-hover:opacity-100 transition-opacity" />
                                                    </div>
                                                )}
                                                
                                                <span className="text-xs text-slate-500">{config.unit}</span>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </Card>
                    ))}
                </div>
            ) : (
                <div className="bg-white dark:bg-slate-900 rounded-2xl border border-slate-200 dark:border-slate-800 shadow-sm overflow-hidden">
                    {/* Data Status & Refresh */}
                    <div className="p-4 bg-slate-50 dark:bg-slate-800/50 border-b border-slate-200 dark:border-slate-700 flex items-center justify-between">
                        <div className="flex items-center gap-3 text-sm text-slate-600 dark:text-slate-400">
                            <Database size={16} />
                            <span>
                                Checking alerts against data up to:{' '}
                                <span className="font-mono font-medium text-slate-900 dark:text-white">
                                    {dataStatus.lastDate ? new Date(dataStatus.lastDate).toLocaleDateString() : 'Loading...'}
                                </span>
                            </span>
                        </div>
                        <Button variant="outline" onClick={checkAlerts} disabled={loading} className="h-8 text-xs">
                            <RefreshCw size={14} className={`mr-1.5 ${loading ? 'animate-spin' : ''}`} />
                            {loading ? 'Checking...' : 'Refresh Alerts'}
                        </Button>
                    </div>
                    <div className="divide-y divide-slate-100 dark:divide-slate-800">
                        {loading ? (
                            <div className="flex flex-col items-center justify-center py-24 text-slate-400">
                                <RefreshCw size={48} className="mb-4 text-primary-500 animate-spin" />
                                <h3 className="text-lg font-semibold text-slate-900 dark:text-white mb-1">Checking Alerts...</h3>
                                <p>Analyzing data against configured thresholds</p>
                            </div>
                        ) : alerts.length > 0 ? (
                            alerts.map(alert => (
                                <div key={alert.id} className={`p-6 transition-colors hover:bg-slate-50 dark:hover:bg-slate-800/50 flex flex-col md:flex-row gap-4 md:items-center justify-between ${!alert.read ? 'bg-primary-50/30 dark:bg-primary-900/10' : ''}`}>
                                    <div className="flex items-start gap-4">
                                        <div className={`mt-1 p-2 rounded-full shrink-0 ${
                                            alert.severity === 'critical' ? 'bg-red-100 text-red-600 dark:bg-red-900/30 dark:text-red-400' :
                                            alert.severity === 'warning' ? 'bg-amber-100 text-amber-600 dark:bg-amber-900/30 dark:text-amber-400' : 
                                            'bg-blue-100 text-blue-600 dark:bg-blue-900/30 dark:text-blue-400'
                                        }`}>
                                            {alert.severity === 'critical' || alert.severity === 'warning' ? <AlertTriangle size={20} /> : <Info size={20} />}
                                        </div>
                                        <div>
                                            <div className="flex items-center gap-2 mb-1">
                                                <h4 className={`text-base font-semibold ${!alert.read ? 'text-slate-900 dark:text-white' : 'text-slate-600 dark:text-slate-400'}`}>
                                                    {alert.title}
                                                </h4>
                                                {!alert.read && <span className="w-2 h-2 rounded-full bg-primary-500 animate-pulse"></span>}
                                            </div>
                                            <p className="text-slate-600 dark:text-slate-400 text-sm mb-2">{alert.message}</p>
                                            <div className="flex items-center gap-3 text-xs text-slate-400 font-medium">
                                                <span className="uppercase tracking-wider border px-1.5 py-0.5 rounded border-slate-200 dark:border-slate-700">{alert.type}</span>
                                                <span>•</span>
                                                <span>{new Date(alert.timestamp).toLocaleString()}</span>
                                            </div>
                                        </div>
                                    </div>
                                    <div className="flex items-center gap-2 self-end md:self-center">
                                        {!alert.read && (
                                            <Button variant="outline" onClick={() => markAsRead(alert.id)} className="h-9 text-xs">
                                                <CheckCircle size={14} className="mr-1.5" /> {t('alerts.markRead')}
                                            </Button>
                                        )}
                                        <button onClick={() => deleteAlert(alert.id)} className="p-2 text-slate-400 hover:text-red-500 hover:bg-red-50 dark:hover:bg-red-900/20 rounded-lg transition-colors" title="Delete Alert">
                                            <Trash2 size={18} />
                                        </button>
                                    </div>
                                </div>
                            ))
                        ) : (
                            <div className="flex flex-col items-center justify-center py-24 text-slate-400">
                                <CheckCircle size={64} className="mb-4 text-green-200 dark:text-green-800" />
                                <h3 className="text-lg font-semibold text-slate-900 dark:text-white mb-1">{t('alerts.allCaughtUp')}</h3>
                                <p>{t('alerts.noAlerts')}</p>
                                <p className="text-xs mt-2">No alerts triggered based on current thresholds</p>
                            </div>
                        )}
                    </div>
                </div>
            )}
        </div>
    );
};
