"""
CLI Interface for Kisan Mitra Advisory System
Run this from command line instead of starting the FastAPI server
"""

from datetime import datetime
from api.dependencies import (
    get_vector_store,
    get_pageindex_provider,
    get_weather_provider,
    get_advisory_generator,
    get_crop_service
)
from core.config import settings

def start_cli():
    """Start CLI interface"""
    
    # Load dependencies
    vector_store = get_vector_store()
    pageindex_provider = get_pageindex_provider()
    weather_provider = get_weather_provider()
    advisory_generator = get_advisory_generator()
    crop_service = get_crop_service()

    mode = settings.retrieval_mode.strip().lower()
    if mode == "pageindex":
        if not pageindex_provider.is_loaded():
            print("[!] PageIndex failed to load. Check PAGEINDEX_TREE_PATH and PAGEINDEX_PDF_PATH.")
            return
    else:
        if not vector_store.is_loaded():
            print("[!] Knowledge base failed to load. Please run the ingest script.")
            return

    print("\n" + "="*50)
    print("    KISAN MITRA - SMART UP ADVISORY")
    print("="*50)
    print(f"Retrieval mode: {mode}")

    # Date Validation
    while True:
        sowing_dt = input("\nBijai ki tarikh (YYYY-MM-DD): ")
        try:
            if datetime.strptime(sowing_dt, "%Y-%m-%d") > datetime.now():
                print("Future date nahi chalegi! Sahi purani tarikh dalein.")
                continue
            break
        except:
            print("Format sahi karein (YYYY-MM-DD)")

    # Latitude/Longitude input
    try:
        lat = float(input(f"Latitude (default {settings.default_latitude} for UP): ") or str(settings.default_latitude))
        lon = float(input(f"Longitude (default {settings.default_longitude} for UP): ") or str(settings.default_longitude))
    except:
        lat, lon = settings.default_latitude, settings.default_longitude
        print(f"Default coordinates use karenge: {lat}, {lon}")

    # Main loop
    while True:
        print("\n" + "-"*50)
        print("--- SAWAL CHUNEIN (0 to Exit) ---")
        questions = settings.predefined_questions
        for k, v in questions.items():
            print(f"{k}. {v}")
        
        choice = input("\nChoice: ")
        
        if choice == "0":
            print("\nPhir milenge! Khush rahiye! 🙏")
            break
        
        question = questions.get(choice)
        if not question:
            print(f"[!] Choice '{choice}' invalid hai!")
            continue
        
        # Get advisory
        print("\n[*] Expert dimaag laga raha hai...")
        
        try:
            # Calculate crop stage
            stage = crop_service.calculate_crop_stage(sowing_dt)
            
            # Fetch weather (3-day forecast)
            weather_current, weather_forecast = weather_provider.fetch_weather(lat, lon)
            
            # Get context (RAG/PageIndex)
            if mode == "pageindex":
                context = pageindex_provider.search(question)
                if not context:
                    print("[!] PageIndex se context nahi mila.")
                    continue
            else:
                docs = vector_store.search(f"Schedule and chemicals for {stage} maize")
                context = "\n".join([d.page_content for d in docs])
            
            # Generate advisory
            advisory = advisory_generator.generate_advisory(question, stage, weather_current, weather_forecast, context)
            
            # Display result
            print("\n" + "-"*50)
            print(advisory)
            print("-"*50)
            
        except Exception as e:
            print(f"\n[!] Error: {str(e)}")

if __name__ == "__main__":
    start_cli()
