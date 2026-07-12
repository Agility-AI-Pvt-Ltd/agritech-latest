from typing import TypedDict, Dict, Any, List
from langgraph.graph import StateGraph, START, END

from core.config import settings
from pipeline.llm_factory import get_llm

class AdvisoryState(TypedDict):
    """State schema for multi-agent workflow"""
    user_query: str
    sowing_date: str
    latitude: float
    longitude: float
    crop_stage: str
    weather_current: Dict[str, Any]
    weather_forecast: List[Dict[str, Any]]
    db_context: str
    technical_response: str
    final_hinglish_response: str

class LangGraphAdvisoryGenerator:
    def __init__(self):
        self._graph = self._build_advisory_graph()

    def _agent_technical_advisor(self, state: AdvisoryState) -> AdvisoryState:
        llm = get_llm(temperature=settings.llm_temperature)
        
        if state['weather_forecast']:
            forecast_text = "\n".join([
                f"Day {f['day']} ({f['date']}): Max {f['temp_max']}°C, Min {f['temp_min']}°C, Rain {f['rain_mm']}mm, Humidity {f['humidity']}%"
                for f in state['weather_forecast']
            ])
        else:
            forecast_text = "[Weather data unavailable - using general seasonal guidelines]"
        
        prompt = f"""
        You are an agronomy expert. Answer the farmer's specific question precisely.
        Use 3-day weather data to guide your practical advice. Keep it SHORT but COMPLETE (150-200 words max).
        
        Crop Stage: {state['crop_stage']}
        Current Weather: {state['weather_current']['temp']}°C
        3-Day Forecast: {forecast_text}
        Knowledge Base: {state['db_context']}
        
        QUESTION: {state['user_query']}
        
        PROVIDE:
        - Exact quantities/dosages (if applicable)
        - Best timing based on weather
        - Specific action steps (2-3 key points)
        - Weather windows or alerts
        - Skip background explanation, be direct
        """
        
        response = llm.invoke(prompt)
        state['technical_response'] = response.content
        return state

    # ==========================================
    # AGENT 2: HINGLISH TRANSLATOR (UP Farmer Friendly)
    # ==========================================
    def _agent_hinglish_translator(self, state: AdvisoryState) -> AdvisoryState:
        """
        Agent 2: Converts technical response to (Uttar-Pradesh India)-style Hinglish for farmers
        Input: technical_response, weather_forecast
        Output: final_hinglish_response
        """
        
        llm = get_llm(temperature=0.1)
        
        # Format forecast for prompt
        if state['weather_forecast']:
            forecast_text = "\n".join([
                f"Day {f['day']} ({f['date']}): Max {f['temp_max']}°C, Min {f['temp_min']}°C, Rain {f['rain_mm']}mm"
                for f in state['weather_forecast']
            ])
        else:
            forecast_text = "[Weather data unavailable]"
        
        prompt = f"""
        You are 'Kisan Mitra'. Convert the technical advice into concise, practical Uttar-Pradesh style Hinglish.
        Write like an experienced local agri advisor speaking directly to a farmer in everyday speech.

        ADVICE TO CONVERT:
        {state['technical_response']}

        WEATHER: {forecast_text}
        CROP STAGE: {state['crop_stage']}

        UP-STYLE GLOSSARY:
        - Use: "aaj", "kal", "abhi", "iss samay", "dhyaan rakhein", "agar", "toh", "khet", "fasal", "patti", "jad", "dawai", "chhidkaav", "sinchai", "matra"
        - Prefer: "Kya karein", "Kab karein", "Kitna daalein", "Baarish aaye toh rok dein"
        - Prefer simple farmer words over technical jargon when possible
        - If a technical word is necessary, explain it in easy Hinglish

        TONE EXAMPLES:
        Example 1:
        "Aaj halki sinchai rakhein. Zyada paani mat dein, warna jad dab sakti hai."

        Example 2:
        "Kal ya parson chhidkaav tabhi karein jab baarish na ho. Subah ya shaam ka samay better rahega."

        Example 3:
        "Patti pe peelepan dikh raha ho toh matra dekhkar khaad dein, andaze se zyada mat daalein."

        FORMAT (संक्षिप्त लेकिन पूरा - CONCISE BUT COMPLETE) :
        SALAH: [Main advice]
        KYUN: [Why it matters + weather connection]
        KYA KAREIN: [Exact steps + quantities]
        SAHI SAMAY: [Which day/time based on forecast]
        BAARISH ALERT: [If rain affects this]

        RULES:
        - Avoid formal Hindi and textbook language
        - Prefer common farmer speech from Uttar Pradesh
        - Keep sentences short and direct
        - Practical, actionable steps only
        - Include exact quantities/timing if available
        - Address weather directlye
        - 120-180 words total
        - No stories, no over-explanation
        - Do not sound robotic or overly polished
        """
        
        response = llm.invoke(prompt)
        state['final_hinglish_response'] = response.content
        return state



    def _build_advisory_graph(self):
        builder = StateGraph(AdvisoryState)
        builder.add_node("technical_advisor", self._agent_technical_advisor)
        builder.add_node("hinglish_translator", self._agent_hinglish_translator)
        
        builder.set_entry_point("technical_advisor")
        builder.add_edge("technical_advisor", "hinglish_translator")
        builder.add_edge("hinglish_translator", END)
        
        return builder.compile()

    def generate_advisory(self, user_query: str, crop_stage: str, weather_current: Dict[str, Any], weather_forecast: List[Dict[str, Any]], context: str) -> str:
        state = AdvisoryState(
            user_query=user_query,
            sowing_date="",
            latitude=0,
            longitude=0,
            crop_stage=crop_stage,
            weather_current=weather_current,
            weather_forecast=weather_forecast,
            db_context=context,
            technical_response="",
            final_hinglish_response=""
        )
        
        result = self._graph.invoke(state)
        return result['final_hinglish_response']
