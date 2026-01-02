
import React, { useState } from 'react';
import { Card, Button } from '../components/ui';
import { Zap, Lock, User, ArrowRight } from 'lucide-react';
import { useLanguage } from '../contexts/LanguageContext';

interface LoginProps {
  onLogin: () => void;
}

export const LoginView: React.FC<LoginProps> = ({ onLogin }) => {
  const { t } = useLanguage();
  const [loading, setLoading] = useState(false);
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setTimeout(() => {
      onLogin();
    }, 800);
  };

  return (
    <div className="min-h-screen bg-slate-50 dark:bg-black flex items-center justify-center p-4 relative overflow-hidden">
        {/* Background Effects */}
        <div className="absolute top-[-10%] right-[-5%] w-[500px] h-[500px] bg-primary-500/10 rounded-full blur-3xl" />
        <div className="absolute bottom-[-10%] left-[-5%] w-[500px] h-[500px] bg-indigo-500/10 rounded-full blur-3xl" />

        <Card className="w-full max-w-md p-8 bg-white/80 dark:bg-slate-900/80 backdrop-blur-xl border-slate-200 dark:border-slate-800 shadow-2xl relative z-10">
            <div className="flex flex-col items-center mb-8">
                <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-primary-500 to-indigo-600 flex items-center justify-center shadow-lg shadow-primary-500/20 text-white mb-4">
                    <Zap size={28} fill="currentColor" />
                </div>
                <h1 className="text-2xl font-bold text-slate-900 dark:text-white">{t('common.welcome')}</h1>
                <p className="text-slate-500 dark:text-slate-400 text-sm mt-1">{t('common.subtitle')}</p>
            </div>

            <form onSubmit={handleSubmit} className="space-y-4">
                <div className="space-y-1.5">
                    <label className="text-xs font-bold text-slate-500 dark:text-slate-400 uppercase tracking-wider ml-1">{t('common.username')}</label>
                    <div className="relative">
                        <div className="absolute left-3 top-2.5 text-slate-400">
                            <User size={18} />
                        </div>
                        <input 
                            type="text" 
                            value={username}
                            onChange={(e) => setUsername(e.target.value)}
                            className="w-full pl-10 pr-4 py-2.5 bg-slate-50 dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-500 text-slate-900 dark:text-white transition-all"
                            placeholder={t('common.username')}
                            required
                        />
                    </div>
                </div>

                <div className="space-y-1.5">
                    <label className="text-xs font-bold text-slate-500 dark:text-slate-400 uppercase tracking-wider ml-1">{t('common.password')}</label>
                    <div className="relative">
                        <div className="absolute left-3 top-2.5 text-slate-400">
                            <Lock size={18} />
                        </div>
                        <input 
                            type="password"
                            value={password}
                            onChange={(e) => setPassword(e.target.value)} 
                            className="w-full pl-10 pr-4 py-2.5 bg-slate-50 dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-500 text-slate-900 dark:text-white transition-all"
                            placeholder="••••••••"
                            required
                        />
                    </div>
                </div>

                <Button 
                    type="submit" 
                    disabled={loading}
                    className="w-full py-3 mt-6 bg-primary-600 hover:bg-primary-500 text-white shadow-lg shadow-primary-500/25"
                >
                    {loading ? (
                        <span className="animate-pulse">{t('common.signingIn')}</span>
                    ) : (
                        <>
                            {t('common.signIn')} <ArrowRight size={18} />
                        </>
                    )}
                </Button>
            </form>

            <div className="mt-8 pt-6 border-t border-slate-100 dark:border-slate-800 text-center">
                <p className="text-xs text-slate-400">
                    {t('common.accessNote')}
                </p>
            </div>
        </Card>
    </div>
  );
};
