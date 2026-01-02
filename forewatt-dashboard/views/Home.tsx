
import React from 'react';
import { ModelType } from '../types';
import { Card, Button } from '../components/ui';
import { Zap, CircleDollarSign, TrendingUp, BarChart3 } from 'lucide-react';
import { useLanguage } from '../contexts/LanguageContext';

interface Props {
  onSelect: (model: ModelType) => void;
}

export const HomeView: React.FC<Props> = ({ onSelect }) => {
  const { t } = useLanguage();
  return (
    <div className="max-w-5xl mx-auto py-12 px-6">
      <div className="text-center mb-16 space-y-4">
        <h1 className="text-5xl font-bold tracking-tight text-slate-900 dark:text-white">{t('home.title')}</h1>
        <p className="text-xl text-slate-500 dark:text-slate-400 max-w-2xl mx-auto">
          {t('home.tagline')}
        </p>
      </div>

      <div className="grid md:grid-cols-2 gap-8 mb-20">
        {/* Consumption Card */}
        <Card className="group relative overflow-hidden border-2 border-transparent hover:border-primary-100 transition-all hover:shadow-xl">
          <div className="absolute top-0 right-0 p-4 opacity-10 group-hover:opacity-20 transition-opacity">
            <Zap size={120} />
          </div>
          <div className="p-8 flex flex-col h-full">
            <div className="w-12 h-12 bg-blue-100 rounded-lg flex items-center justify-center mb-6 text-blue-600">
              <Zap size={24} />
            </div>
            <h2 className="text-2xl font-bold text-slate-900 dark:text-white mb-2">{t('home.consCardTitle')}</h2>
            <p className="text-slate-500 dark:text-slate-400 mb-6 flex-1">
              {t('home.consCardDesc')}
            </p>
            <div className="bg-slate-50 dark:bg-slate-800 rounded-lg p-4 mb-6">
              <div className="flex items-center gap-2 text-sm text-slate-600 dark:text-slate-300">
                <TrendingUp size={16} className="text-green-500" />
                <span>{t('home.bestMAE')}: <strong>892 MWh</strong></span>
              </div>
            </div>
            <Button onClick={() => onSelect('consumption')} className="w-full">
              {t('home.viewForecasts')}
            </Button>
          </div>
        </Card>

        {/* Price Card */}
        <Card className="group relative overflow-hidden border-2 border-transparent hover:border-emerald-100 transition-all hover:shadow-xl">
          <div className="absolute top-0 right-0 p-4 opacity-10 group-hover:opacity-20 transition-opacity">
            <CircleDollarSign size={120} />
          </div>
          <div className="p-8 flex flex-col h-full">
            <div className="w-12 h-12 bg-emerald-100 rounded-lg flex items-center justify-center mb-6 text-emerald-600">
              <CircleDollarSign size={24} />
            </div>
            <h2 className="text-2xl font-bold text-slate-900 dark:text-white mb-2">{t('home.priceCardTitle')}</h2>
            <p className="text-slate-500 dark:text-slate-400 mb-6 flex-1">
               {t('home.priceCardDesc')}
            </p>
            <div className="bg-slate-50 dark:bg-slate-800 rounded-lg p-4 mb-6">
              <div className="flex items-center gap-2 text-sm text-slate-600 dark:text-slate-300">
                <BarChart3 size={16} className="text-emerald-500" />
                <span>{t('home.bestMAE')}: <strong>78.5 TL/MWh</strong></span>
              </div>
            </div>
            <Button onClick={() => onSelect('price')} className="w-full bg-emerald-600 hover:bg-emerald-700">
              {t('home.viewForecasts')}
            </Button>
          </div>
        </Card>
      </div>

      <div className="border-t border-slate-200 dark:border-slate-800 pt-8 text-center text-slate-400 text-sm">
        <p>{t('home.footer')}</p>
      </div>
    </div>
  );
};
