import re
import json
from langchain.schema import HumanMessage

class PANExtractor:
    def __init__(self, llm):
        self.llm = llm
        self.patterns = {
            'pan_number': r'\b[A-Z]{5}[0-9]{4}[A-Z]\b',
            'name': r'(?:Name|NAME|Permanent Account Number Card|Permanent Account Number)[\s:]*([A-Z][A-Za-z\s]+?)(?:\n|$)',
            'father_name': r'(?:Father\'?s Name|FATHER\'?S NAME)[\s:]*([A-Z][A-Za-z\s]+?)(?:\n|$)',
            'date_of_birth': r'\b\d{1,2}[/-]\d{1,2}[/-]\d{4}\b',
        }
        self.pan_keywords = [
            'income tax',
            'permanent account number',
            'government of india',
            'pan card',
            'incometaxdepartment'
        ]

    def preprocess_text(self, text):
        text = ' '.join(text.split())
        corrections = {
            'INCOMETAX': 'INCOME TAX',
            'Goverment': 'Government',
            'Pemranent': 'Permanent',
            'Accont': 'Account',
            'Nunber': 'Number',
            'FATHER:': 'FATHER',
            'NAME:': 'NAME',
            'DOB:': 'Date of Birth:'
        }
        for wrong, correct in corrections.items():
            text = text.replace(wrong, correct)
        return text

    def extract_patterns(self, text):
        found_data = {}
        for pattern_name, pattern in self.patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                if pattern_name == 'pan_number':
                    found_data[pattern_name] = matches[0].upper()
                elif pattern_name in ['name', 'father_name']:
                    name = ' '.join([word for word in matches[0].split() if len(word) > 1])
                    if 2 < len(name) < 50:
                        found_data[pattern_name] = name.title()
                else:
                    found_data[pattern_name] = matches[0]
        return found_data

    def llm_extract(self, text):
        prompt = f"""Extract PAN card details from this text. Handle OCR errors:

{text}

Return JSON with:
- name (title case)
- pan_number (format: ABCDE1234F)
- father_name (if found)
- date_of_birth (if found)
- signature_present (boolean)
- photo_present (boolean)

Rules:
- PAN must be 10 chars (5 letters, 4 digits, 1 letter)
- Clean names from OCR errors
- Return empty fields if not found"""
        
        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            return json.loads(response.content)
        except Exception as e:
            print(f"⚠️ LLM extraction error: {str(e)}")
            return None

    def determine_confidence(self, data):
        if data.get('pan_number') and data.get('name'):
            return "high"
        elif data.get('pan_number'):
            return "medium"
        return "low"

    def extract(self, raw_text):
        try:
            cleaned_text = self.preprocess_text(raw_text)
            
            # First try strict pattern matching
            pan_match = re.search(self.patterns['pan_number'], cleaned_text)
            if pan_match:
                pattern_data = self.extract_patterns(cleaned_text)
                if pattern_data.get('name'):
                    return {
                        "document_type": "PAN Card",
                        "confidence": "high",
                        "data": pattern_data
                    }
            
            # Fallback to LLM if strict matching fails
            llm_result = self.llm_extract(cleaned_text)
            if llm_result and llm_result.get('pan_number'):
                return {
                    "document_type": "PAN Card",
                    "confidence": self.determine_confidence(llm_result),
                    "data": {k: v for k, v in llm_result.items() if v}
                }
            
            return {
                "document_type": "PAN Card",
                "confidence": "low",
                "data": {},
                "note": "No valid PAN data found"
            }
            
        except Exception as e:
            return {
                "document_type": "PAN Card",
                "confidence": "low",
                "error": str(e)
            }