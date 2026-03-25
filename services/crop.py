from datetime import datetime

class CropService:
    @staticmethod
    def calculate_crop_stage(sowing_date: str) -> str:
        """Calculate crop stage based on days since sowing"""
        sowing = datetime.strptime(sowing_date, "%Y-%m-%d")
        days = (datetime.now() - sowing).days
        
        if days <= 15: 
            return "Establishment/Seedling"
        elif 20 <= days <= 35: 
            return "Knee-High"
        elif 40 <= days <= 55: 
            return "Tasseling/Silking"
        else: 
            return "Maturity"
