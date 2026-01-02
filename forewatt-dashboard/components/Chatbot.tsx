import React, { useState, useEffect, useRef } from 'react';
import { geminiService } from '../services/geminiService';
import { RealTimeData } from '../types';
import { 
    X, Send, Bot, User, Loader2, Maximize2, Minimize2, 
    Code, Copy, Check, Terminal, Sparkles, ChevronRight
} from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { Button } from './ui';

interface Message {
  id: string;
  role: 'user' | 'assistant';
  text: string;
}

interface ChatbotProps {
    isOpen: boolean;
    onClose: () => void;
    isFullScreen: boolean;
    onToggleFullScreen: () => void;
    contextData: RealTimeData | null;
}

// Custom Code Block Component
const CodeBlock = ({ language, code }: { language: string, code: string }) => {
    const [activeTab, setActiveTab] = useState<'code' | 'preview'>('code');
    const [copied, setCopied] = useState(false);

    const handleCopy = () => {
        navigator.clipboard.writeText(code);
        setCopied(true);
        setTimeout(() => setCopied(false), 2000);
    };

    return (
        <div className="my-4 rounded-lg overflow-hidden border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-900 shadow-sm">
            <div className="flex items-center justify-between px-3 py-2 bg-slate-100 dark:bg-slate-800 border-b border-slate-200 dark:border-slate-700">
                <div className="flex gap-1 bg-slate-200 dark:bg-slate-700 p-0.5 rounded-lg">
                    <button 
                        onClick={() => setActiveTab('code')}
                        className={`px-3 py-1 text-xs font-medium rounded-md transition-all ${activeTab === 'code' ? 'bg-white dark:bg-slate-600 shadow-sm text-slate-900 dark:text-white' : 'text-slate-500 dark:text-slate-400 hover:text-slate-700'}`}
                    >
                        Code
                    </button>
                    <button 
                        onClick={() => setActiveTab('preview')}
                        className={`px-3 py-1 text-xs font-medium rounded-md transition-all ${activeTab === 'preview' ? 'bg-white dark:bg-slate-600 shadow-sm text-slate-900 dark:text-white' : 'text-slate-500 dark:text-slate-400 hover:text-slate-700'}`}
                    >
                        Preview
                    </button>
                </div>
                <div className="flex items-center gap-2">
                    <span className="text-xs text-slate-500 uppercase font-mono hidden sm:block">{language || 'text'}</span>
                    <button onClick={handleCopy} className="p-1.5 hover:bg-slate-200 dark:hover:bg-slate-700 rounded text-slate-500 transition-colors">
                        {copied ? <Check size={14} className="text-green-500" /> : <Copy size={14} />}
                    </button>
                </div>
            </div>
            
            <div className="relative group">
                {activeTab === 'code' ? (
                    <div className="p-4 overflow-x-auto bg-[#1e1e1e] text-blue-100 font-mono text-sm leading-relaxed">
                        <pre>{code}</pre>
                    </div>
                ) : (
                    <div className="p-8 bg-white dark:bg-slate-950 flex items-center justify-center min-h-[150px] border-t border-slate-100 dark:border-slate-800">
                        <div className="text-center">
                             <Terminal size={32} className="mx-auto text-slate-300 mb-2" />
                             <p className="text-sm text-slate-500">Preview Canvas</p>
                             <p className="text-xs text-slate-400 mt-1">Mock runtime environment active</p>
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
};

export const Chatbot = ({ isOpen, onClose, isFullScreen, onToggleFullScreen, contextData }: ChatbotProps) => {
  const [messages, setMessages] = useState<Message[]>([
    { id: '1', role: 'assistant', text: "Hello! I'm the ForeWatt AI. \n\nI can analyze **price trends**, detect **anomalies**, or help you compare **historical data**.\n\nHow can I assist you today?" }
  ]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (isOpen) {
        geminiService.startChat(contextData);
        scrollToBottom();
    }
  }, [isOpen, contextData]);

  const scrollToBottom = () => {
      messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSend = async () => {
    if (!input.trim() || isLoading) return;

    const userMsg: Message = { id: Date.now().toString(), role: 'user', text: input };
    setMessages(prev => [...prev, userMsg]);
    setInput('');
    setIsLoading(true);

    try {
      const responseText = await geminiService.sendMessage(userMsg.text);
      const aiMsg: Message = { id: (Date.now() + 1).toString(), role: 'assistant', text: responseText };
      setMessages(prev => [...prev, aiMsg]);
    } finally {
      setIsLoading(false);
    }
  };

  // Custom Markdown Components
  const MarkdownComponents = {
      code(props: any) {
          const {children, className, node, ...rest} = props
          const match = /language-(\w+)/.exec(className || '')
          return match ? (
              <CodeBlock language={match[1]} code={String(children).replace(/\n$/, '')} />
          ) : (
              <code {...rest} className={className}>
                  {children}
              </code>
          )
      }
  };

  // Determine container styles based on mode (FullScreen Overlay vs Sidebar Push)
  const containerClasses = isFullScreen
    ? "fixed inset-0 z-50 w-full"
    : `relative h-full transition-all duration-300 ease-in-out border-l border-slate-200 dark:border-slate-800 ${isOpen ? 'w-full md:w-[450px] lg:w-[500px] border-l' : 'w-0 border-none'}`;
  
  const innerClasses = isFullScreen
    ? "w-full"
    : "min-w-[450px] lg:min-w-[500px]"; // Ensure inner content doesn't wrap during width transition

  return (
    <div className={`${containerClasses} bg-white dark:bg-slate-900 shadow-2xl flex flex-col overflow-hidden shrink-0`}>
        <div className={`flex flex-col h-full ${innerClasses}`}>
            {/* Header */}
            <div className="flex items-center justify-between px-6 py-4 border-b border-slate-100 dark:border-slate-800 bg-white/50 dark:bg-slate-900/50 backdrop-blur-sm shrink-0">
                <div className="flex items-center gap-3">
                    <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-primary-500 to-indigo-600 flex items-center justify-center shadow-lg shadow-primary-500/20 text-white">
                        <Bot size={20} />
                    </div>
                    <div>
                        <h3 className="font-bold text-slate-900 dark:text-white leading-tight">ForeWatt Assistant</h3>
                        <div className="flex items-center gap-1.5">
                            <span className="w-2 h-2 rounded-full bg-green-500 animate-pulse"></span>
                            <span className="text-xs font-medium text-slate-500 dark:text-slate-400">Gemini 3.0 Flash</span>
                        </div>
                    </div>
                </div>
                <div className="flex items-center gap-1">
                    <button 
                        onClick={onToggleFullScreen}
                        className="p-2 text-slate-500 hover:text-slate-900 dark:text-slate-400 dark:hover:text-white hover:bg-slate-100 dark:hover:bg-slate-800 rounded-lg transition-colors"
                        title={isFullScreen ? "Exit Full Screen" : "Full Screen"}
                    >
                        {isFullScreen ? <Minimize2 size={20} /> : <Maximize2 size={20} />}
                    </button>
                    <button 
                        onClick={onClose}
                        className="p-2 text-slate-500 hover:text-slate-900 dark:text-slate-400 dark:hover:text-white hover:bg-slate-100 dark:hover:bg-slate-800 rounded-lg transition-colors"
                    >
                        <ChevronRight size={24} />
                    </button>
                </div>
            </div>

            {/* Chat Area */}
            <div className="flex-1 overflow-y-auto p-6 space-y-6 bg-slate-50/50 dark:bg-black/20 scrollbar-thin scrollbar-thumb-slate-300 dark:scrollbar-thumb-slate-700">
                {messages.map((msg) => (
                    <div key={msg.id} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'} animate-in fade-in slide-in-from-bottom-2 duration-300`}>
                        <div className={`flex max-w-[90%] md:max-w-[85%] gap-4 ${msg.role === 'user' ? 'flex-row-reverse' : 'flex-row'}`}>
                            {/* Avatar */}
                            <div className={`w-8 h-8 rounded-full flex items-center justify-center shrink-0 mt-1 ${
                                msg.role === 'user' 
                                ? 'bg-slate-200 dark:bg-slate-700 text-slate-600 dark:text-slate-300' 
                                : 'bg-primary-100 dark:bg-primary-900/30 text-primary-600 dark:text-primary-400'
                            }`}>
                                {msg.role === 'user' ? <User size={16} /> : <Sparkles size={16} />}
                            </div>

                            {/* Message Bubble */}
                            <div className={`rounded-2xl p-4 shadow-sm text-sm leading-relaxed markdown-body ${
                                msg.role === 'user'
                                ? 'bg-primary-600 text-white rounded-tr-none'
                                : 'bg-white dark:bg-slate-800 text-slate-800 dark:text-slate-200 rounded-tl-none border border-slate-100 dark:border-slate-700'
                            }`}>
                                {msg.role === 'user' ? (
                                    <p>{msg.text}</p>
                                ) : (
                                    <ReactMarkdown components={MarkdownComponents} remarkPlugins={[remarkGfm]}>
                                        {msg.text}
                                    </ReactMarkdown>
                                )}
                            </div>
                        </div>
                    </div>
                ))}
                
                {isLoading && (
                    <div className="flex justify-start animate-in fade-in">
                        <div className="flex max-w-[85%] gap-4">
                            <div className="w-8 h-8 rounded-full bg-primary-100 dark:bg-primary-900/30 text-primary-600 dark:text-primary-400 flex items-center justify-center shrink-0">
                                <Sparkles size={16} />
                            </div>
                            <div className="bg-white dark:bg-slate-800 px-4 py-3 rounded-2xl rounded-tl-none border border-slate-100 dark:border-slate-700 shadow-sm flex items-center gap-2">
                                <Loader2 size={16} className="animate-spin text-primary-500" />
                                <span className="text-xs text-slate-500 dark:text-slate-400">Analyzing patterns...</span>
                            </div>
                        </div>
                    </div>
                )}
                <div ref={messagesEndRef} />
            </div>

            {/* Input Area */}
            <div className="p-4 bg-white dark:bg-slate-900 border-t border-slate-100 dark:border-slate-800 shrink-0">
                <div className="relative flex items-end gap-2 p-2 bg-slate-50 dark:bg-slate-800/50 border border-slate-200 dark:border-slate-700 rounded-xl focus-within:ring-2 focus-within:ring-primary-500/20 focus-within:border-primary-500 transition-all">
                    <textarea
                        value={input}
                        onChange={(e) => setInput(e.target.value)}
                        onKeyDown={(e) => {
                            if (e.key === 'Enter' && !e.shiftKey) {
                                e.preventDefault();
                                handleSend();
                            }
                        }}
                        placeholder="Ask about electricity prices, anomalies, or forecasts..."
                        className="w-full bg-transparent border-none focus:ring-0 resize-none max-h-32 min-h-[44px] py-2.5 px-3 text-sm text-slate-900 dark:text-white placeholder:text-slate-400"
                        rows={1}
                    />
                    <div className="pb-1 pr-1">
                        <Button 
                            onClick={handleSend} 
                            disabled={isLoading || !input.trim()} 
                            className={`w-9 h-9 p-0 rounded-lg transition-all ${
                                !input.trim() 
                                ? 'bg-slate-200 dark:bg-slate-700 text-slate-400 dark:text-slate-500 shadow-none' 
                                : 'bg-primary-600 hover:bg-primary-500 text-white shadow-md'
                            }`}
                        >
                            <Send size={18} />
                        </Button>
                    </div>
                </div>
                <div className="mt-2 text-center">
                    <p className="text-[10px] text-slate-400 dark:text-slate-500">
                        AI can make mistakes. Please verify critical forecast data.
                    </p>
                </div>
            </div>
        </div>
    </div>
  );
};