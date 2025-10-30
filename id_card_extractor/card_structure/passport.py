import re
import json
from langchain.schema import HumanMessage

class PassportExtractor:
    def __init__(self, llm):
        self.llm = llm
        self.patterns = {
            'passport_number': r'\b[A-Z][0-9]{7}\b',
            'name': r'(?:Name|NAME)[\s:]*([A-Z][A-Za-z\s]+?)(?:\n|$)',
            'surname': r'(?:Surname|SURNAME)[\s:]*([A-Z][A-Za-z\s]+?)(?:\n|$)',
            'nationality': r'(?:Nationality|NATIONALITY)[\s:]*([A-Z][A-Za-z\s]+?)(?:\n|$)',
            'date_of_birth': r'\b\d{1,2}[/-]\d{1,2}[/-]\d{4}\b',
            'place_of_birth': r'(?:Place of Birth|PLACE OF BIRTH)[\s:]*([A-Z][A-Za-z\s]+?)(?:\n|$)',
            'date_of_issue': r'(?:Date of Issue|DATE OF ISSUE)[\s:]*(\d{1,2}[/-]\d{1,2}[/-]\d{4})(?:\n|$)',
            'date_of_expiry': r'(?:Date of Expiry|DATE OF EXPIRY)[\s:]*(\d{1,2}[/-]\d{1,2}[/-]\d{4})(?:\n|$)',
        }
        self.keywords = [
            'passport',
            'government of india',
            'republic of india',
            'ministry of external affairs',
            'indian passport'
        ]

    def preprocess_text(self, text):
        text = ' '.join(text.split())
        text = re.sub(r'[^\w\s/-]', '', text)  # Remove special chars except / and -
        return text

    def extract_patterns(self, text):
        found_data = {}
        for pattern_name, pattern in self.patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                if pattern_name == 'passport_number':
                    found_data[pattern_name] = matches[0].upper()
                elif pattern_name in ['name', 'surname', 'nationality', 'place_of_birth']:
                    name = ' '.join([word for word in matches[0].split() if len(word) > 1])
                    if 2 < len(name) < 50:
                        found_data[pattern_name] = name.title()
                else:
                    found_data[pattern_name] = matches[0]
        return found_data

    def llm_extract(self, text):
        prompt = f"""Extract passport details from:
{text}

Return JSON with:
- passport_number (format A1234567)
- name
- surname
- nationality
- date_of_birth
- place_of_birth
- date_of_issue
- date_of_expiry

Rules:
- Passport number must start with letter then 7 digits
- Clean names from OCR errors
- Return empty fields if not found"""
        
        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            return json.loads(response.content)
        except Exception as e:
            print(f"⚠️ LLM extraction error: {str(e)}")
            return None

    def determine_confidence(self, data):
        if data.get('passport_number') and data.get('name'):
            return "high"
        elif data.get('passport_number'):
            return "medium"
        return "low"

    def extract(self, raw_text):
        try:
            cleaned_text = self.preprocess_text(raw_text)
            
            # First try strict pattern matching
            passport_match = re.search(self.patterns['passport_number'], cleaned_text)
            if passport_match:
                pattern_data = self.extract_patterns(cleaned_text)
                if pattern_data.get('name'):
                    return {
                        "document_type": "Passport",
                        "confidence": "high",
                        "data": pattern_data
                    }
            
            # Fallback to LLM if strict matching fails
            llm_result = self.llm_extract(cleaned_text)
            if llm_result and llm_result.get('passport_number'):
                return {
                    "document_type": "Passport",
                    "confidence": self.determine_confidence(llm_result),
                    "data": {k: v for k, v in llm_result.items() if v}
                }
            
            return {
                "document_type": "Passport",
                "confidence": "low",
                "data": {},
                "note": "No valid Passport data found"
            }
            
        except Exception as e:
            return {
                "document_type": "Passport",
                "confidence": "low",
                "error": str(e)
            }