import { GoogleGenAI } from "@google/genai";
import { RealTimeData } from '../types';

const SYSTEM_INSTRUCTION = `You are the ForeWatt AI Assistant, an expert helper integrated into Turkey's electricity forecasting dashboard.
Your role is to help users understand forecasts, analyze patterns, and make data-driven decisions.

## Platform Context
ForeWatt is an open-source machine learning platform for 24-hour ahead electricity forecasting in Turkey.
- Consumption Forecasting: Predicts hourly electricity demand in MWh.
- Price Forecasting: Predicts hourly PTF (Day-Ahead Market) prices in TL/MWh.

## Data Infrastructure
- Source: EPIAŞ (Turkish Energy Exchange).
- Real-time data arrives with a 2-hour delay.
- Dashboard shows: 6h actual -> 2h gap -> 12h forecast.
- Times are Europe/Istanbul (UTC+3).

## Response Guidelines
1. Be Concise: Users need quick insights.
2. Use Data: Reference specific numbers provided in the context.
3. Acknowledge Uncertainty: If data is missing, say so.
4. Provide Context: Relate current values to typical patterns (e.g., Evening peaks, Morning ramps).

## Knowledge Base
- Consumption Peaks: 18:00-21:00. Min: 02:00-06:00.
- Price Patterns: Morning ramp (08-10), Afternoon dip (13-15 solar), Evening peak (18-21).
`;

// Get API key from Vite environment variables
const API_KEY = import.meta.env.VITE_GEMINI_API_KEY || import.meta.env.GEMINI_API_KEY;

export class GeminiService {
  private ai: GoogleGenAI | null = null;
  private model: string;
  private chat: any;
  private initialized: boolean = false;

  constructor() {
    this.model = 'gemini-2.0-flash';
    if (API_KEY && API_KEY !== 'PLACEHOLDER_API_KEY') {
      this.ai = new GoogleGenAI({ apiKey: API_KEY });
      this.initialized = true;
    } else {
      console.warn('Gemini API key not configured. Chat will use mock responses.');
    }
  }

  public isAvailable(): boolean {
    return this.initialized;
  }

  public async startChat(currentContextData: RealTimeData | null) {
    if (!this.ai) return;

    let contextString = "No live data available currently.";

    if (currentContextData) {
      contextString = `
      CURRENT DASHBOARD STATE:
      Model: ${currentContextData.modelType}
      Unit: ${currentContextData.unit}
      Last Actual: ${currentContextData.actual[currentContextData.actual.length - 1]?.value.toFixed(2)} at ${currentContextData.actual[currentContextData.actual.length - 1]?.timestamp}
      Next Forecast: ${currentContextData.forecast[0]?.value.toFixed(2)} at ${currentContextData.forecast[0]?.timestamp}
      Summary:
      - Avg Actual: ${currentContextData.summary.avgActual.toFixed(2)}
      - Avg Forecast: ${currentContextData.summary.avgForecast.toFixed(2)}
      `;
    }

    this.chat = this.ai.chats.create({
      model: this.model,
      config: {
        systemInstruction: SYSTEM_INSTRUCTION + "\n\n" + contextString,
      },
    });
  }

  public async sendMessage(message: string): Promise<string> {
    // If API is not configured, return mock response
    if (!this.ai) {
      return this.getMockResponse(message);
    }

    if (!this.chat) {
      await this.startChat(null);
    }

    try {
      const response = await this.chat.sendMessage({ message });
      return response.text || "I couldn't generate a response.";
    } catch (error) {
      console.error("Gemini Error:", error);
      return "I'm having trouble connecting to the forecasting brain right now. Please try again.";
    }
  }

  private getMockResponse(message: string): string {
    const lowerMsg = message.toLowerCase();

    if (lowerMsg.includes('price') || lowerMsg.includes('fiyat')) {
      return `Based on current patterns, electricity prices typically follow these trends:
- **Morning ramp**: 08:00-10:00 (prices increase as demand rises)
- **Afternoon dip**: 13:00-15:00 (solar generation peaks)
- **Evening peak**: 18:00-21:00 (highest prices of the day)

To get live AI analysis, please configure your Gemini API key in the .env.local file.`;
    }

    if (lowerMsg.includes('consumption') || lowerMsg.includes('demand') || lowerMsg.includes('tüketim')) {
      return `Electricity consumption in Turkey typically shows:
- **Peak hours**: 18:00-21:00 (evening household + industrial demand)
- **Low hours**: 02:00-06:00 (minimal activity)
- **Weekday vs Weekend**: ~15% lower on weekends

To get live AI analysis, please configure your Gemini API key in the .env.local file.`;
    }

    if (lowerMsg.includes('anomaly') || lowerMsg.includes('anomali')) {
      return `Anomaly detection monitors for unusual patterns:
- Sudden spikes or drops in consumption/price
- Deviations from historical hourly patterns
- Unexpected divergence between forecast and actual values

The dashboard flags data points with anomaly scores > 0.8 as potential anomalies.`;
    }

    return `I'm the ForeWatt AI Assistant. I can help you with:
- **Price forecasts** and market analysis
- **Consumption patterns** and demand trends
- **Anomaly detection** and unusual events
- **Historical comparisons** between time periods

Note: For full AI capabilities, configure your Gemini API key in .env.local:
\`VITE_GEMINI_API_KEY=your_key_here\``;
  }
}

export const geminiService = new GeminiService();
