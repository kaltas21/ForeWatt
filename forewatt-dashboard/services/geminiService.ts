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

export class GeminiService {
  private ai: GoogleGenAI;
  private model: any;
  private chat: any;

  constructor() {
    this.ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
    this.model = 'gemini-3-flash-preview';
  }

  public async startChat(currentContextData: RealTimeData | null) {
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
}

export const geminiService = new GeminiService();
