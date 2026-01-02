import React from 'react';

export const Card = ({ children, className = '' }: { children: React.ReactNode; className?: string }) => (
  <div className={`bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-800 shadow-sm ${className}`}>
    {children}
  </div>
);

export const Button = ({ 
  children, 
  variant = 'primary', 
  className = '',
  ...props
}: { 
  children: React.ReactNode; 
  variant?: 'primary' | 'secondary' | 'outline' | 'ghost'; 
} & React.ButtonHTMLAttributes<HTMLButtonElement>) => {
  const base = "px-4 py-2 rounded-lg font-medium transition-all duration-200 flex items-center justify-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed";
  const variants = {
    primary: "bg-primary-600 hover:bg-primary-500 text-white shadow-lg shadow-primary-500/20",
    secondary: "bg-slate-100 dark:bg-slate-800 hover:bg-slate-200 dark:hover:bg-slate-700 text-slate-900 dark:text-slate-100",
    outline: "border border-slate-300 dark:border-slate-700 hover:bg-slate-50 dark:hover:bg-slate-800 text-slate-700 dark:text-slate-200",
    ghost: "hover:bg-slate-100 dark:hover:bg-slate-800 text-slate-600 dark:text-slate-300"
  };

  return (
    <button className={`${base} ${variants[variant]} ${className}`} {...props}>
      {children}
    </button>
  );
};

export const Badge = ({ children, color = 'blue' }: { children: React.ReactNode; color?: 'blue' | 'green' | 'red' | 'yellow' }) => {
    const colors = {
        blue: 'bg-blue-100 dark:bg-blue-900/30 text-blue-800 dark:text-blue-300 border border-blue-200 dark:border-blue-800',
        green: 'bg-green-100 dark:bg-green-900/30 text-green-800 dark:text-green-300 border border-green-200 dark:border-green-800',
        red: 'bg-red-100 dark:bg-red-900/30 text-red-800 dark:text-red-300 border border-red-200 dark:border-red-800',
        yellow: 'bg-amber-100 dark:bg-amber-900/30 text-amber-800 dark:text-amber-300 border border-amber-200 dark:border-amber-800',
    }
    return (
        <span className={`px-2 py-0.5 rounded-full text-xs font-semibold ${colors[color]}`}>
            {children}
        </span>
    )
}

export const Toggle = ({ label, checked, onChange }: { label: string; checked: boolean; onChange: (v: boolean) => void }) => (
    <div className="flex items-center gap-2 cursor-pointer group" onClick={() => onChange(!checked)}>
        <div className={`w-11 h-6 rounded-full p-1 transition-all duration-300 ${checked ? 'bg-primary-600 shadow-inner' : 'bg-slate-300 dark:bg-slate-600'}`}>
            <div className={`bg-white w-4 h-4 rounded-full shadow-sm transform transition-all duration-300 ${checked ? 'translate-x-5' : ''}`} />
        </div>
        <span className="text-sm font-medium text-slate-700 dark:text-slate-300 select-none group-hover:text-primary-600 dark:group-hover:text-primary-400 transition-colors">{label}</span>
    </div>
);