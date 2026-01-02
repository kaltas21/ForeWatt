
import React, { useEffect, useRef, useState } from 'react';
import * as echarts from 'echarts';
import { RealTimeData, AnomalyData, HistoricalData, ComparisonData } from '../types';

interface RealTimeChartProps {
    data: RealTimeData;
    showCI: boolean;
}

interface HistoricalChartProps {
    data: HistoricalData;
    showActual: boolean;
    showForecast: boolean;
}

interface ComparisonChartProps {
    data: ComparisonData;
}

const isDarkMode = () => document.documentElement.classList.contains('dark');

export const RealTimeChart: React.FC<RealTimeChartProps> = ({ data, showCI }) => {
    const chartRef = useRef<HTMLDivElement>(null);
    const instanceRef = useRef<echarts.EChartsType | null>(null);
    const [theme, setTheme] = useState<'dark' | 'light'>(isDarkMode() ? 'dark' : 'light');

    useEffect(() => {
        const observer = new MutationObserver(() => {
            const newTheme = isDarkMode() ? 'dark' : 'light';
            setTheme(newTheme);
        });
        observer.observe(document.documentElement, { attributes: true, attributeFilter: ['class'] });
        return () => observer.disconnect();
    }, []);

    useEffect(() => {
        if (!chartRef.current) return;
        
        if (instanceRef.current) {
             instanceRef.current.dispose();
        }
        
        instanceRef.current = echarts.init(chartRef.current, theme === 'dark' ? 'dark' : undefined);
        const chart = instanceRef.current;
        const isDark = theme === 'dark';

        // 1. Prepare X-Axis Categories (Hourly Slots) to ensure proper stacking and gap visualization
        const allPoints = [...data.actual, ...data.forecast];
        if (allPoints.length === 0) return;

        const timestamps = allPoints.map(d => new Date(d.timestamp).getTime());
        const minTime = Math.min(...timestamps);
        const maxTime = Math.max(...timestamps);
        
        // Generate complete hourly range to preserve visual gaps
        const categoryData: string[] = [];
        let iterTime = minTime;
        // Round down to nearest hour to be safe
        const startHour = new Date(minTime);
        startHour.setMinutes(0, 0, 0);
        iterTime = startHour.getTime();

        while (iterTime <= maxTime) {
            categoryData.push(new Date(iterTime).toISOString());
            iterTime += 3600 * 1000; // Add 1 hour
        }

        // 2. Map Data to Categories
        // We use a Map for O(1) lookup
        const actualMap = new Map(data.actual.map(d => {
            const t = new Date(d.timestamp);
            t.setMinutes(0,0,0); 
            return [t.toISOString(), d.value];
        }));

        const forecastMap = new Map(data.forecast.map(d => {
            const t = new Date(d.timestamp);
            t.setMinutes(0,0,0);
            return [t.toISOString(), d];
        }));

        const actualSeriesData = categoryData.map(t => actualMap.get(t) ?? null);
        const forecastSeriesData = categoryData.map(t => forecastMap.get(t)?.value ?? null);
        
        // CI Data Preparation (Stacked Area Logic)
        // Base = Lower
        // Top = Upper - Lower
        const ciLowerSeriesData = categoryData.map(t => {
            const f = forecastMap.get(t);
            return f ? (f.lower || 0) : null;
        });

        const ciDiffSeriesData = categoryData.map(t => {
            const f = forecastMap.get(t);
            return f ? ((f.upper || 0) - (f.lower || 0)) : null;
        });
        
        // CI Borders (for the dashed lines)
        const ciUpperBorderData = categoryData.map(t => {
            const f = forecastMap.get(t);
            return f ? (f.upper || 0) : null;
        });

        const option: echarts.EChartsOption = {
            backgroundColor: 'transparent',
            animation: false, 
            toolbox: {
                feature: {
                    saveAsImage: { title: 'Save Image', backgroundColor: isDark ? '#0f172a' : '#ffffff' },
                    dataZoom: { title: { zoom: 'Zoom', back: 'Reset' } },
                    restore: { title: 'Restore' }
                },
                iconStyle: { borderColor: isDark ? '#94a3b8' : '#64748b' },
                top: 0,
                right: 20
            },
            tooltip: {
                trigger: 'axis',
                axisPointer: { type: 'line' },
                backgroundColor: isDark ? 'rgba(15, 23, 42, 0.95)' : 'rgba(255, 255, 255, 0.95)',
                borderColor: isDark ? '#334155' : '#e2e8f0',
                textStyle: { color: isDark ? '#f8fafc' : '#0f172a' },
                formatter: (params: any) => {
                    let res = '';
                    if (Array.isArray(params) && params.length > 0) {
                        const dateStr = params[0].axisValue;
                        const date = new Date(dateStr);
                        res += `<div style="font-weight:bold; margin-bottom: 4px;">${date.getHours().toString().padStart(2, '0')}:00</div>`;
                        params.forEach((param: any) => {
                            // Filter out technical series used for visualization
                            if (param.seriesName.includes('CI Fill') || param.seriesName.includes('CI Border')) return;
                            if (param.value == null) return;

                            const val = param.value;
                            const marker = param.marker;
                            res += `<div>${marker} ${param.seriesName}: <b>${typeof val === 'number' ? val.toFixed(2) : val}</b></div>`;
                        });
                        
                        const fItem = forecastMap.get(dateStr);
                        if (fItem && showCI) {
                             res += `<div style="font-size: 11px; opacity: 0.8; margin-top: 4px; color: ${isDark ? '#94a3b8' : '#64748b'}">
                                CI: ${fItem.lower?.toFixed(2)} - ${fItem.upper?.toFixed(2)}
                             </div>`;
                        }
                    }
                    return res;
                }
            },
            grid: { top: 60, right: 30, bottom: 60, left: 60 },
            dataZoom: [
                { type: 'inside', start: 0, end: 100 },
                { 
                    type: 'slider', 
                    bottom: 0, 
                    height: 20, 
                    borderColor: 'transparent', 
                    backgroundColor: isDark ? '#1e293b' : '#f1f5f9',
                    fillerColor: 'rgba(14, 165, 233, 0.2)',
                    handleStyle: { color: '#0ea5e9' },
                    textStyle: { color: isDark ? '#94a3b8' : '#64748b' } 
                }
            ],
            xAxis: {
                type: 'category',
                data: categoryData,
                boundaryGap: false,
                axisLabel: {
                    formatter: (value: string) => {
                        const date = new Date(value);
                        return date.getHours().toString().padStart(2, '0') + ':00';
                    },
                    color: isDark ? '#94a3b8' : '#64748b'
                },
                splitLine: { show: false }
            },
            yAxis: {
                type: 'value',
                name: data.unit,
                scale: true,
                axisLabel: { color: isDark ? '#94a3b8' : '#64748b' },
                splitLine: { lineStyle: { color: isDark ? '#334155' : '#e2e8f0', type: 'dashed' as const } }
            },
            series: [
                // CI Implementation using Stacking on Category Axis (Robust)
                ...(showCI ? [{
                     name: 'CI Fill Base',
                     type: 'line' as const,
                     data: ciLowerSeriesData,
                     smooth: false, 
                     stack: 'confidence-fill',
                     symbol: 'none',
                     lineStyle: { opacity: 0 },
                     // Important: connectNulls must be false to respect gaps if any, 
                     // but here we want the CI to be continuous where forecast exists.
                     connectNulls: true, 
                     silent: true,
                     z: 0
                 }, {
                    name: 'CI Fill Top',
                    type: 'line' as const,
                    data: ciDiffSeriesData,
                    smooth: false,
                    stack: 'confidence-fill',
                    symbol: 'none',
                    lineStyle: { opacity: 0 },
                    areaStyle: {
                        color: isDark ? 'rgba(56, 189, 248, 0.25)' : 'rgba(14, 165, 233, 0.25)', 
                        opacity: 1
                    },
                    connectNulls: true,
                    silent: true,
                    z: 0
                 },
                 // Separate Border Lines
                 {
                    name: 'CI Border Lower',
                    type: 'line' as const,
                    data: ciLowerSeriesData,
                    smooth: false, 
                    symbol: 'circle',
                    symbolSize: 4,
                    lineStyle: { 
                        opacity: 0.6, 
                        width: 1, 
                        type: 'dashed' as const,
                        color: isDark ? '#38bdf8' : '#0ea5e9'
                    },
                    itemStyle: { color: isDark ? '#38bdf8' : '#0ea5e9' },
                    connectNulls: true,
                    z: 10
                 },
                 {
                    name: 'CI Border Upper',
                    type: 'line' as const,
                    data: ciUpperBorderData,
                    smooth: false, 
                    symbol: 'circle',
                    symbolSize: 4,
                    lineStyle: { 
                        opacity: 0.6, 
                        width: 1, 
                        type: 'dashed' as const,
                        color: isDark ? '#38bdf8' : '#0ea5e9'
                    },
                    itemStyle: { color: isDark ? '#38bdf8' : '#0ea5e9' },
                    connectNulls: true,
                    z: 10
                 }] : []),

                // Actual Data Line
                {
                    name: 'Actual',
                    type: 'line',
                    data: actualSeriesData,
                    smooth: true,
                    symbolSize: (val: number | null) => val ? Math.max(4, Math.min(10, val / (data.modelType === 'price' ? 300 : 5000))) : 0,
                    itemStyle: { color: isDark ? '#f8fafc' : '#0f172a' },
                    lineStyle: { width: 3 },
                    connectNulls: false, // Don't bridge the gap
                    markPoint: {
                        data: data.actual.length > 0 ? [
                            { 
                                name: 'Pivot',
                                coord: [data.actual[data.actual.length-1].timestamp, data.actual[data.actual.length-1].value], 
                                itemStyle: { 
                                    color: '#ffffff',
                                    borderColor: '#ef4444',
                                    borderWidth: 2
                                },
                                label: { show: false }
                            }
                        ] : [],
                        symbol: 'circle',
                        symbolSize: 8,
                        animation: false
                    },
                    z: 20
                },
                // Forecast Data Line
                {
                    name: 'Forecast',
                    type: 'line',
                    data: forecastSeriesData,
                    smooth: true, 
                    lineStyle: { type: 'dashed' as const, width: 3 },
                    itemStyle: { color: '#0ea5e9' },
                    symbolSize: (val: number | null) => val ? Math.max(4, Math.min(10, val / (data.modelType === 'price' ? 300 : 5000))) : 0,
                    connectNulls: false,
                    markLine: {
                        symbol: ['none', 'none'],
                        label: { formatter: 'Pivot (T-2h)', color: '#ef4444' },
                        data: [{ xAxis: new Date(data.pivotTime).toISOString() }],
                        lineStyle: { color: '#ef4444', type: 'solid' as const }
                    },
                    z: 20
                }
            ]
        };

        chart.setOption(option);

        // Add ResizeObserver
        const resizeObserver = new ResizeObserver(() => {
            chart.resize();
        });
        resizeObserver.observe(chartRef.current);

        return () => {
            resizeObserver.disconnect();
            chart.dispose();
            instanceRef.current = null;
        }
    }, [data, showCI, theme]);

    return <div ref={chartRef} className="w-full h-[450px]" />;
};

export const AnomalyChart: React.FC<{ data: AnomalyData }> = ({ data }) => {
    const chartRef = useRef<HTMLDivElement>(null);
    const [theme, setTheme] = useState<'dark' | 'light'>(isDarkMode() ? 'dark' : 'light');

    useEffect(() => {
        const observer = new MutationObserver(() => {
            const newTheme = isDarkMode() ? 'dark' : 'light';
            setTheme(newTheme);
        });
        observer.observe(document.documentElement, { attributes: true, attributeFilter: ['class'] });
        return () => observer.disconnect();
    }, []);

    useEffect(() => {
        if (!chartRef.current) return;
        const isDark = theme === 'dark';
        const chart = echarts.init(chartRef.current, isDark ? 'dark' : undefined);
        
        const option: echarts.EChartsOption = {
             backgroundColor: 'transparent',
             animation: false,
             toolbox: {
                feature: {
                    saveAsImage: { title: 'Save Image', backgroundColor: isDark ? '#0f172a' : '#ffffff' },
                    dataZoom: { title: { zoom: 'Zoom', back: 'Reset' } },
                    restore: { title: 'Restore' }
                },
                iconStyle: { borderColor: isDark ? '#94a3b8' : '#64748b' },
                top: 0,
                right: 0
            },
            grid: { top: 30, right: 30, bottom: 80, left: 50, containLabel: true },
            dataZoom: [
                { type: 'inside', start: 0, end: 100 },
                { 
                    type: 'slider', 
                    bottom: 0, 
                    height: 20, 
                    borderColor: 'transparent', 
                    backgroundColor: isDark ? '#1e293b' : '#f1f5f9',
                    fillerColor: 'rgba(14, 165, 233, 0.2)',
                    handleStyle: { color: '#0ea5e9' },
                    textStyle: { color: isDark ? '#94a3b8' : '#64748b' } 
                }
            ],
             tooltip: { trigger: 'axis' },
             xAxis: { 
                 type: 'category', 
                 data: data.anomalies.map(a => new Date(a.timestamp).toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})),
                 axisLabel: { color: isDark ? '#94a3b8' : '#64748b' }
             },
             yAxis: { 
                 type: 'value',
                 splitLine: { lineStyle: { color: isDark ? '#334155' : '#e2e8f0', type: 'dashed' as const } }
             },
             series: [
                 {
                     type: 'scatter',
                     data: data.anomalies.map(a => ({
                        value: a.residual,
                        itemStyle: {
                            color: Math.abs(a.residual) > 150 ? '#ef4444' : '#22c55e',
                            shadowColor: Math.abs(a.residual) > 150 ? 'rgba(239, 68, 68, 0.5)' : 'rgba(34, 197, 94, 0.5)'
                        }
                     })),
                     symbolSize: (val: number) => Math.min(20, Math.max(6, Math.abs(val) / 20)),
                     itemStyle: {
                         shadowBlur: 10
                     }
                 },
                 {
                     type: 'line',
                     data: data.anomalies.map(() => 0),
                     symbol: 'none',
                     lineStyle: {
                         type: 'dashed' as const,
                         color: isDark ? '#475569' : '#cbd5e1',
                         width: 1
                     },
                     silent: true
                 }
             ]
        };
        chart.setOption(option);
        
        const resizeObserver = new ResizeObserver(() => {
            chart.resize();
        });
        resizeObserver.observe(chartRef.current);

        return () => {
            resizeObserver.disconnect();
            chart.dispose();
        }
    }, [data, theme]);

    return <div ref={chartRef} className="w-full h-[450px]" />;
}

export const HistoricalChart: React.FC<HistoricalChartProps> = ({ data, showActual, showForecast }) => {
    const chartRef = useRef<HTMLDivElement>(null);
    const [theme, setTheme] = useState<'dark' | 'light'>(isDarkMode() ? 'dark' : 'light');

    useEffect(() => {
        const observer = new MutationObserver(() => {
            const newTheme = isDarkMode() ? 'dark' : 'light';
            setTheme(newTheme);
        });
        observer.observe(document.documentElement, { attributes: true, attributeFilter: ['class'] });
        return () => observer.disconnect();
    }, []);

    useEffect(() => {
        if (!chartRef.current) return;
        
        const isDark = theme === 'dark';
        const chart = echarts.init(chartRef.current, isDark ? 'dark' : undefined);
        
        const option: echarts.EChartsOption = {
            backgroundColor: 'transparent',
            animation: false,
            toolbox: {
                feature: {
                    saveAsImage: { title: 'Save Image', backgroundColor: isDark ? '#0f172a' : '#ffffff' },
                    dataZoom: { title: { zoom: 'Zoom', back: 'Reset' } },
                    restore: { title: 'Restore' }
                },
                iconStyle: { borderColor: isDark ? '#94a3b8' : '#64748b' },
                top: 0,
                right: 20
            },
            dataZoom: [
                { type: 'inside', start: 0, end: 100 },
                { 
                    type: 'slider', 
                    bottom: 0, 
                    height: 20, 
                    borderColor: 'transparent', 
                    backgroundColor: isDark ? '#1e293b' : '#f1f5f9',
                    fillerColor: 'rgba(14, 165, 233, 0.2)',
                    handleStyle: { color: '#0ea5e9' },
                    textStyle: { color: isDark ? '#94a3b8' : '#64748b' } 
                }
            ],
            tooltip: { 
                trigger: 'axis',
                backgroundColor: isDark ? 'rgba(15, 23, 42, 0.9)' : 'rgba(255, 255, 255, 0.9)',
                borderColor: isDark ? '#334155' : '#e2e8f0',
                textStyle: { color: isDark ? '#f8fafc' : '#0f172a' }
            },
            legend: {
                top: 0,
                textStyle: { color: isDark ? '#cbd5e1' : '#475569' },
                data: ['Actual', 'Forecast']
            },
            grid: { left: 60, right: 30, bottom: 60, top: 50 },
            xAxis: { 
                type: 'time',
                axisLabel: { color: isDark ? '#94a3b8' : '#64748b' },
                splitLine: { show: false }
            },
            yAxis: { 
                type: 'value',
                scale: true,
                splitLine: { lineStyle: { color: isDark ? '#334155' : '#e2e8f0', type: 'dashed' as const } },
                axisLabel: { color: isDark ? '#94a3b8' : '#64748b' }
            },
            series: [
                { 
                    name: 'Actual',
                    type: 'line', 
                    data: showActual ? data.data.map(d => [d.timestamp, d.actual]) : [], 
                    color: isDark ? '#38bdf8' : '#0284c7',
                    smooth: true,
                    showSymbol: false,
                    lineStyle: { width: 2 }
                },
                { 
                    name: 'Forecast',
                    type: 'line', 
                    data: showForecast ? data.data.map(d => [d.timestamp, d.forecast]) : [], 
                    color: isDark ? '#f472b6' : '#db2777',
                    smooth: true,
                    showSymbol: false,
                    lineStyle: { width: 2, type: 'dashed' as const }
                }
            ]
        };
        chart.setOption(option);

        const resizeObserver = new ResizeObserver(() => {
            chart.resize();
        });
        resizeObserver.observe(chartRef.current);

        return () => {
            resizeObserver.disconnect();
            chart.dispose();
        }
    }, [data, showActual, showForecast, theme]);

    return <div ref={chartRef} className="w-full h-[500px]" />;
}

export const ComparisonChart: React.FC<ComparisonChartProps> = ({ data }) => {
    const chartRef = useRef<HTMLDivElement>(null);
    const [theme, setTheme] = useState<'dark' | 'light'>(isDarkMode() ? 'dark' : 'light');

    useEffect(() => {
        const observer = new MutationObserver(() => {
            const newTheme = isDarkMode() ? 'dark' : 'light';
            setTheme(newTheme);
        });
        observer.observe(document.documentElement, { attributes: true, attributeFilter: ['class'] });
        return () => observer.disconnect();
    }, []);

    useEffect(() => {
        if (!chartRef.current) return;
        const isDark = theme === 'dark';
        const chart = echarts.init(chartRef.current, isDark ? 'dark' : undefined);
        
        const option: echarts.EChartsOption = {
             backgroundColor: 'transparent',
             animation: false,
             toolbox: {
                feature: {
                    saveAsImage: { title: 'Save Image', backgroundColor: isDark ? '#0f172a' : '#ffffff' },
                    dataZoom: { title: { zoom: 'Zoom', back: 'Reset' } },
                    restore: { title: 'Restore' }
                },
                iconStyle: { borderColor: isDark ? '#94a3b8' : '#64748b' },
                top: 0,
                right: 20
            },
            dataZoom: [
                { type: 'inside', start: 0, end: 100 },
                { 
                    type: 'slider', 
                    bottom: 0, 
                    height: 20, 
                    borderColor: 'transparent', 
                    backgroundColor: isDark ? '#1e293b' : '#f1f5f9',
                    fillerColor: 'rgba(14, 165, 233, 0.2)',
                    handleStyle: { color: '#0ea5e9' },
                    textStyle: { color: isDark ? '#94a3b8' : '#64748b' } 
                }
            ],
             tooltip: { 
                 trigger: 'axis',
                 backgroundColor: isDark ? 'rgba(15, 23, 42, 0.9)' : 'rgba(255, 255, 255, 0.9)',
                 borderColor: isDark ? '#334155' : '#e2e8f0',
                 textStyle: { color: isDark ? '#f8fafc' : '#0f172a' }
             },
             legend: {
                top: 0,
                textStyle: { color: isDark ? '#cbd5e1' : '#475569' },
                data: [data.periodA.label, data.periodB.label]
             },
             grid: { left: 50, right: 30, bottom: 60, top: 40 },
             xAxis: { 
                 type: 'category', 
                 data: data.periodA.data.map(d => d.label),
                 axisLabel: { color: isDark ? '#94a3b8' : '#64748b' }
             },
             yAxis: { 
                 type: 'value',
                 scale: true,
                 splitLine: { lineStyle: { color: isDark ? '#334155' : '#e2e8f0', type: 'dashed' as const } },
                 axisLabel: { color: isDark ? '#94a3b8' : '#64748b' }
             },
             series: [
                 {
                     name: data.periodA.label,
                     type: 'line',
                     data: data.periodA.data.map(d => d.value),
                     smooth: true,
                     lineStyle: { width: 3, color: '#0ea5e9' },
                     itemStyle: { color: '#0ea5e9' },
                     areaStyle: { opacity: 0.1, color: '#0ea5e9' }
                 },
                 {
                     name: data.periodB.label,
                     type: 'line',
                     data: data.periodB.data.map(d => d.value),
                     smooth: true,
                     lineStyle: { width: 3, type: 'dashed' as const, color: '#94a3b8' },
                     itemStyle: { color: '#94a3b8' }
                 }
             ]
        };
        chart.setOption(option);
        
        const resizeObserver = new ResizeObserver(() => {
            chart.resize();
        });
        resizeObserver.observe(chartRef.current);

        return () => {
            resizeObserver.disconnect();
            chart.dispose();
        }
    }, [data, theme]);

    return <div ref={chartRef} className="w-full h-[400px]" />;
}
