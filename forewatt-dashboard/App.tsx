
import React, { useState, useEffect } from 'react';
import { ViewState, ModelType, RealTimeData } from './types';
import { HomeView } from './views/Home';
import { RealTimeView } from './views/RealTime';
import { HistoricalView } from './views/Historical';
import { AnomalyView } from './views/Anomaly';
import { CompareView } from './views/Compare';
import { AlertsView } from './views/Alerts';
import { LoginView } from './views/Login';
import { Chatbot } from './components/Chatbot';
import { Card, Toggle } from './components/ui';
import { LanguageProvider, useLanguage } from './contexts/LanguageContext';
import { 
    LayoutDashboard, History, AlertTriangle, Menu, X, Zap, 
    CircleDollarSign, Settings, Moon, Sun, Monitor, Scale, 
    Bell, MessageSquare, Sparkles, LogOut, Globe, Clock, 
    RefreshCw, Mail, Smartphone
} from 'lucide-react';

const SidebarItem = ({ icon: Icon, label, active, onClick }: any) => (
  <button
    onClick={onClick}
    className={`relative w-full flex items-center gap-3 px-4 py-3 rounded-xl transition-all duration-300 group overflow-hidden ${
      active 
      ? 'bg-gradient-to-r from-primary-600 to-primary-500 text-white shadow-lg shadow-primary-500/30' 
      : 'text-slate-500 dark:text-slate-400 hover:bg-slate-100 dark:hover:bg-slate-800 hover:text-slate-900 dark:hover:text-slate-100'
    }`}
  >
    {active && <div className="absolute inset-0 bg-white/10 opacity-0 group-hover:opacity-100 transition-opacity" />}
    <Icon size={20} />
    <span className="font-medium tracking-wide">{label}</span>
  </button>
);

const AppContent = () => {
  const { t, language, setLanguage } = useLanguage();
  // Auth State
  const [isAuthenticated, setIsAuthenticated] = useState(false);

  // Initialize directly to REALTIME view and CONSUMPTION model
  const [view, setView] = useState<ViewState>('REALTIME');
  const [model, setModel] = useState<ModelType>('consumption');
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [currentRealTimeData, setCurrentRealTimeData] = useState<RealTimeData | null>(null);
  
  // Theme & Settings State
  const [darkMode, setDarkMode] = useState(true);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [currentTime, setCurrentTime] = useState(new Date());

  // Additional Settings
  const [notifications, setNotifications] = useState(true);

  // Chat State
  const [isChatOpen, setIsChatOpen] = useState(false);
  const [isChatFullScreen, setIsChatFullScreen] = useState(false);

  useEffect(() => {
    // Clock Update
    const timer = setInterval(() => setCurrentTime(new Date()), 1000);
    return () => clearInterval(timer);
  }, []);

  useEffect(() => {
    // Theme Toggle
    if (darkMode) {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
  }, [darkMode]);

  const handleLogout = () => {
    setIsAuthenticated(false);
    setView('REALTIME');
    setSidebarOpen(true);
  };

  if (!isAuthenticated) {
    return <LoginView onLogin={() => setIsAuthenticated(true)} />;
  }

  const renderContent = () => {
    switch (view) {
      case 'HOME': return <HomeView onSelect={(m) => { setModel(m); setView('REALTIME'); }} />;
      case 'REALTIME': return <RealTimeView model={model} onDataUpdate={setCurrentRealTimeData} />;
      case 'HISTORICAL': return <HistoricalView model={model} />;
      case 'ANOMALY': return <AnomalyView model={model} />;
      case 'COMPARE': return <CompareView model={model} />;
      case 'ALERTS': return <AlertsView />;
      default: return <HomeView onSelect={(m) => { setModel(m); setView('REALTIME'); }} />;
    }
  };

  return (
    <div className="flex h-screen bg-slate-50 dark:bg-black overflow-hidden transition-colors duration-300 font-sans relative">
      
      {/* Sidebar - Glassmorphism & Flashy */}
      <aside 
          className={`${sidebarOpen ? 'w-72' : 'w-0'} bg-white/80 dark:bg-slate-900/80 backdrop-blur-xl border-r border-slate-200 dark:border-slate-800 transition-all duration-500 cubic-bezier(0.4, 0, 0.2, 1) flex flex-col fixed md:relative z-30 h-full overflow-hidden whitespace-nowrap`}
      >
          {/* Logo Area */}
          <div className="p-8 flex items-center gap-3">
              <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-primary-500 to-indigo-600 flex items-center justify-center shadow-lg shadow-primary-500/20 text-white">
                  <Zap size={24} fill="currentColor" />
              </div>
              <div>
                <span className="font-bold text-2xl text-slate-900 dark:text-white tracking-tight block">ForeWatt</span>
                <span className="text-xs text-slate-400 uppercase tracking-widest font-semibold">{t('common.dashboard') || 'Dashboard'}</span>
              </div>
          </div>
          
          <div className="px-4 py-2 flex-1 flex flex-col gap-8">
              {/* Apple-style Model Switcher */}
              <div className="bg-slate-100 dark:bg-slate-800 p-1 rounded-xl flex relative">
                  <div className={`absolute left-1 top-1 bottom-1 w-[calc(50%-4px)] bg-white dark:bg-slate-700 rounded-lg shadow-sm transition-all duration-300 ease-spring ${model === 'price' ? 'translate-x-full' : 'translate-x-0'}`} />
                  
                  <button 
                    onClick={() => setModel('consumption')}
                    className={`flex-1 relative z-10 py-2 text-sm font-semibold text-center transition-colors duration-200 ${model === 'consumption' ? 'text-primary-600 dark:text-white' : 'text-slate-500 dark:text-slate-400'}`}
                  >
                    {t('models.consumption')}
                  </button>
                  <button 
                    onClick={() => setModel('price')}
                    className={`flex-1 relative z-10 py-2 text-sm font-semibold text-center transition-colors duration-200 ${model === 'price' ? 'text-emerald-500 dark:text-emerald-300' : 'text-slate-500 dark:text-slate-400'}`}
                  >
                    {t('models.price')}
                  </button>
              </div>

              {/* Navigation */}
              <nav className="space-y-2">
                  <p className="px-4 text-xs font-bold text-slate-400 uppercase tracking-wider mb-2">{t('nav.analytics')}</p>
                  <SidebarItem icon={LayoutDashboard} label={t('nav.realTime')} active={view === 'REALTIME'} onClick={() => setView('REALTIME')} />
                  <SidebarItem icon={History} label={t('nav.history')} active={view === 'HISTORICAL'} onClick={() => setView('HISTORICAL')} />
                  <SidebarItem icon={AlertTriangle} label={t('nav.anomalies')} active={view === 'ANOMALY'} onClick={() => setView('ANOMALY')} />
                  <SidebarItem icon={Scale} label={t('nav.compare')} active={view === 'COMPARE'} onClick={() => setView('COMPARE')} />
                  <SidebarItem icon={Bell} label={t('nav.alerts')} active={view === 'ALERTS'} onClick={() => setView('ALERTS')} />
              </nav>

              <div className="mt-auto">
                 <p className="px-4 text-xs font-bold text-slate-400 uppercase tracking-wider mb-2">{t('nav.system')}</p>
                 <SidebarItem icon={Settings} label={t('common.settings')} active={false} onClick={() => setSettingsOpen(true)} />
                 <SidebarItem icon={LogOut} label={t('common.signOut')} active={false} onClick={handleLogout} />
              </div>
          </div>

          <div className="p-6 border-t border-slate-200 dark:border-slate-800">
              <div className="flex items-center gap-3">
                 <div className="w-8 h-8 rounded-full bg-slate-200 dark:bg-slate-700 flex items-center justify-center">
                    <Monitor size={16} className="text-slate-500 dark:text-slate-300" />
                 </div>
                 <div>
                     <div className="text-sm font-medium text-slate-900 dark:text-white">{t('common.adminView')}</div>
                     <div className="text-xs text-slate-500 dark:text-slate-400">{t('common.turkeyGrid')}</div>
                 </div>
              </div>
          </div>
      </aside>

      {/* Main Content */}
      <main className="flex-1 flex flex-col h-full overflow-hidden relative w-full bg-slate-50 dark:bg-black/50">
        
        {/* Header - Glassmorphism */}
        <header className="h-20 bg-white/70 dark:bg-slate-900/70 backdrop-blur-md border-b border-slate-200 dark:border-slate-800 flex items-center justify-between px-8 shrink-0 sticky top-0 z-20 transition-all duration-300">
            <div className="flex items-center gap-4">
                <button onClick={() => setSidebarOpen(!sidebarOpen)} className="p-2 hover:bg-slate-200 dark:hover:bg-slate-800 rounded-lg text-slate-600 dark:text-slate-300 transition-colors">
                    {sidebarOpen ? <X size={24} /> : <Menu size={24} />}
                </button>
                <div className="hidden md:flex items-center gap-3 pl-4 border-l border-slate-200 dark:border-slate-700">
                    <span className="relative flex h-3 w-3">
                      <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-green-400 opacity-75"></span>
                      <span className="relative inline-flex rounded-full h-3 w-3 bg-green-500"></span>
                    </span>
                    <span className="text-sm font-medium text-slate-600 dark:text-slate-300 tracking-wide">{t('common.systemOp')}</span>
                </div>
            </div>
            
            <div className="flex items-center gap-6">
                 {/* Clock */}
                 <div className="text-right hidden sm:block">
                     <div className="text-2xl font-bold font-mono text-slate-900 dark:text-white leading-none">
                        {currentTime.toLocaleTimeString('en-GB', { hour12: false })}
                     </div>
                     <div className="text-[10px] text-slate-500 dark:text-slate-400 uppercase tracking-widest font-bold">Europe/Istanbul</div>
                 </div>

                 <div className="flex items-center gap-3">
                     <button 
                        onClick={() => setIsChatOpen(!isChatOpen)}
                        className={`flex items-center gap-2 px-4 py-2 rounded-xl transition-all duration-300 border ${
                            isChatOpen 
                            ? 'bg-primary-600 border-primary-500 text-white shadow-lg shadow-primary-500/30' 
                            : 'bg-white dark:bg-slate-800 border-slate-200 dark:border-slate-700 text-slate-700 dark:text-slate-200 hover:border-primary-500/50 hover:bg-slate-50 dark:hover:bg-slate-700'
                        }`}
                     >
                        <Sparkles size={18} className={isChatOpen ? 'fill-white animate-pulse' : 'text-primary-500'} />
                        <span className="text-sm font-bold tracking-wide">{t('common.askAI')}</span>
                     </button>

                     <button onClick={() => setView('ALERTS')} className="relative p-2.5 bg-slate-100 dark:bg-slate-800 rounded-full text-slate-600 dark:text-slate-300 hover:bg-slate-200 dark:hover:bg-slate-700 transition-colors">
                         <Bell size={20} />
                         <span className="absolute top-1 right-1 h-2.5 w-2.5 bg-red-500 rounded-full border-2 border-white dark:border-slate-900"></span>
                     </button>
                     <button onClick={() => setSettingsOpen(true)} className="p-2.5 bg-slate-100 dark:bg-slate-800 rounded-full text-slate-600 dark:text-slate-300 hover:bg-slate-200 dark:hover:bg-slate-700 transition-colors">
                         <Settings size={20} />
                     </button>
                 </div>
            </div>
        </header>

        {/* Content Wrapper for Side-by-Side Layout */}
        <div className="flex-1 flex overflow-hidden relative">
            <div className="flex-1 overflow-auto p-6 md:p-10 scrollbar-hide relative transition-all duration-300">
                {renderContent()}
            </div>
            
            {/* Chatbot is now a sibling, allowing it to push content when it expands */}
            <Chatbot 
                isOpen={isChatOpen} 
                onClose={() => setIsChatOpen(false)} 
                isFullScreen={isChatFullScreen}
                onToggleFullScreen={() => setIsChatFullScreen(!isChatFullScreen)}
                contextData={currentRealTimeData} 
            />
        </div>

        {/* Settings Modal */}
        {settingsOpen && (
            <div className="fixed inset-0 z-[60] flex items-center justify-center p-4 bg-black/40 backdrop-blur-sm animate-in fade-in duration-200">
                <div className="bg-white dark:bg-slate-900 w-full max-w-lg rounded-2xl shadow-2xl border border-slate-200 dark:border-slate-700 overflow-hidden">
                    <div className="p-6 border-b border-slate-100 dark:border-slate-800 flex justify-between items-center bg-slate-50 dark:bg-slate-800/50">
                        <h3 className="text-xl font-bold text-slate-900 dark:text-white flex items-center gap-2">
                            <Settings className="text-primary-500" size={24} />
                            {t('settings.title')}
                        </h3>
                        <button onClick={() => setSettingsOpen(false)} className="text-slate-400 hover:text-slate-900 dark:hover:text-white transition-colors">
                            <X size={24} />
                        </button>
                    </div>
                    
                    <div className="p-6 space-y-6 max-h-[70vh] overflow-y-auto custom-scrollbar">
                        
                        {/* Section: Appearance */}
                        <div className="space-y-4">
                            <h4 className="text-xs font-bold text-slate-400 uppercase tracking-wider">{t('settings.appearance')}</h4>
                            
                            <div className="flex items-center justify-between p-3 rounded-lg bg-slate-50 dark:bg-slate-800/50 border border-slate-100 dark:border-slate-800">
                                <div className="flex items-center gap-3">
                                    <div className={`p-2 rounded-lg ${darkMode ? 'bg-indigo-500 text-white' : 'bg-amber-400 text-white'}`}>
                                        {darkMode ? <Moon size={20} /> : <Sun size={20} />}
                                    </div>
                                    <div>
                                        <div className="font-medium text-slate-900 dark:text-white">{t('settings.themeMode')}</div>
                                        <div className="text-xs text-slate-500 dark:text-slate-400">{t('settings.themeDesc')}</div>
                                    </div>
                                </div>
                                <Toggle label="" checked={darkMode} onChange={setDarkMode} />
                            </div>

                            <div className="flex items-center justify-between p-3 rounded-lg bg-slate-50 dark:bg-slate-800/50 border border-slate-100 dark:border-slate-800">
                                <div className="flex items-center gap-3">
                                    <div className="p-2 rounded-lg bg-emerald-500/10 text-emerald-500">
                                        <Globe size={20} />
                                    </div>
                                    <div>
                                        <div className="font-medium text-slate-900 dark:text-white">{t('settings.language')}</div>
                                        <div className="text-xs text-slate-500 dark:text-slate-400">{t('settings.langDesc')}</div>
                                    </div>
                                </div>
                                <select 
                                    value={language} 
                                    onChange={(e) => setLanguage(e.target.value as any)}
                                    className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-700 text-slate-900 dark:text-white text-sm rounded-lg p-2 focus:ring-2 focus:ring-primary-500 outline-none"
                                >
                                    <option value="en">English (US)</option>
                                    <option value="tr">Türkçe (TR)</option>
                                </select>
                            </div>
                        </div>

                        {/* Section: Data */}
                        <div className="space-y-4 pt-4 border-t border-slate-100 dark:border-slate-800">
                            <h4 className="text-xs font-bold text-slate-400 uppercase tracking-wider">{t('settings.dataConn')}</h4>

                             <div className="flex items-center justify-between p-3 rounded-lg bg-slate-50 dark:bg-slate-800/50 border border-slate-100 dark:border-slate-800">
                                <div className="flex items-center gap-3">
                                    <div className="p-2 rounded-lg bg-green-500/20 text-green-600 dark:text-green-400">
                                        <Zap size={20} />
                                    </div>
                                    <div>
                                        <div className="font-medium text-slate-900 dark:text-white">{t('settings.liveStream')}</div>
                                        <div className="text-xs text-slate-500 dark:text-slate-400">{t('settings.streamDesc')}</div>
                                    </div>
                                </div>
                                 <div className="flex items-center gap-2">
                                    <span className="relative flex h-2 w-2">
                                      <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-green-400 opacity-75"></span>
                                      <span className="relative inline-flex rounded-full h-2 w-2 bg-green-500"></span>
                                    </span>
                                    <span className="text-xs font-bold text-green-600 dark:text-green-400">ACTIVE</span>
                                </div>
                            </div>
                        </div>

                         {/* Section: Notifications */}
                         <div className="space-y-4 pt-4 border-t border-slate-100 dark:border-slate-800">
                            <h4 className="text-xs font-bold text-slate-400 uppercase tracking-wider">{t('settings.notifications')}</h4>
                            
                            <div className="flex items-center justify-between p-3 rounded-lg bg-slate-50 dark:bg-slate-800/50 border border-slate-100 dark:border-slate-800">
                                <div className="flex items-center gap-3">
                                    <div className="p-2 rounded-lg bg-purple-500/10 text-purple-500">
                                        <Bell size={20} />
                                    </div>
                                    <div>
                                        <div className="font-medium text-slate-900 dark:text-white">{t('settings.sysAlerts')}</div>
                                        <div className="text-xs text-slate-500 dark:text-slate-400">{t('settings.alertDesc')}</div>
                                    </div>
                                </div>
                                <Toggle label="" checked={notifications} onChange={setNotifications} />
                            </div>
                        </div>

                    </div>
                    <div className="p-4 border-t border-slate-100 dark:border-slate-800 bg-slate-50 dark:bg-slate-900/50 flex justify-end">
                        <button onClick={() => setSettingsOpen(false)} className="px-6 py-2.5 bg-primary-600 hover:bg-primary-700 text-white rounded-xl font-semibold transition-colors shadow-lg shadow-primary-500/20">
                            {t('settings.saveChanges')}
                        </button>
                    </div>
                </div>
            </div>
        )}
      </main>
    </div>
  );
};

const App = () => (
  <LanguageProvider>
    <AppContent />
  </LanguageProvider>
);

export default App;
