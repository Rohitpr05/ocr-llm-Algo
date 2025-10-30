import re
import json
from langchain.schema import HumanMessage

class AadhaarExtractor:
    def __init__(self, llm):
        """Initialize Aadhaar extractor with LLM instance"""
        self.llm = llm
        
        # Aadhaar-specific patterns
        self.patterns = {
            'aadhaar_number': r'\b\d{4}\s*\d{4}\s*\d{4}\b',
            'phone_number': r'\b[6-9]\d{9}\b',
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'pincode': r'\b\d{6}\b',
            'date_of_birth': r'\b\d{1,2}[/-]\d{1,2}[/-]\d{4}\b',
            'vid_number': r'\b\d{16}\b',  # Virtual ID
        }
        
        # Common Aadhaar keywords and phrases
        self.aadhaar_keywords = [
            'government of india',
            'unique identification authority',
            'aadhaar',
            'aadhar',
            'uid',
            'enrollment',
            'resident',
            'address',
            'father',
            'mother',
            'husband',
            'wife',
            'son',
            'daughter',
            'date of birth',
            'dob',
            'gender',
            'male',
            'female',
            'pin',
            'enrolment',
            'issued',
            'govt',
            'india'
        ]

    def preprocess_text(self, text):
        """Clean and preprocess OCR text for better extraction"""
        # Remove extra whitespace and normalize
        text = ' '.join(text.split())
        
        # Fix common OCR errors for Aadhaar
        corrections = {
            'Govemment': 'Government',
            'Govemment': 'Government',
            'Goverment': 'Government',
            'Umque': 'Unique',
            'Identfication': 'Identification',
            'Authonty': 'Authority',
            'Authonty': 'Authority',
            'Aadhar': 'Aadhaar',
            'Aadhar': 'Aadhaar',
            'Bom': 'Born',
            'Bom:': 'Born:',
            'DOB:': 'Date of Birth:',
            'D.O.B:': 'Date of Birth:',
            'Enrolment': 'Enrollment',
            'Enrolment': 'Enrollment',
        }
        
        for wrong, correct in corrections.items():
            text = text.replace(wrong, correct)
        
        return text

    def extract_patterns(self, text):
        """Extract data using regex patterns"""
        found_data = {}
        
        for pattern_name, pattern in self.patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                if pattern_name == 'aadhaar_number':
                    # Clean and format Aadhaar number
                    aadhaar = re.sub(r'\s+', '', matches[0])
                    found_data[pattern_name] = f"{aadhaar[:4]} {aadhaar[4:8]} {aadhaar[8:]}"
                elif pattern_name == 'phone_number':
                    found_data[pattern_name] = matches[0]
                elif pattern_name == 'email':
                    found_data[pattern_name] = matches[0].lower()
                elif pattern_name == 'pincode':
                    # Only include if it's exactly 6 digits
                    if len(matches[0]) == 6:
                        found_data[pattern_name] = matches[0]
                else:
                    found_data[pattern_name] = matches[0]
        
        return found_data

    def extract_name_from_text(self, text):
        """Extract name using common patterns in Aadhaar cards"""
        # Look for name patterns
        name_patterns = [
            r'(?:Name|NAME)[\s:]+([A-Z][A-Za-z\s]+?)(?:\n|$|[A-Z]{2,})',
            r'^([A-Z][A-Za-z\s]+?)\s+(?:S/O|D/O|W/O|C/O)',
            r'(?:S/O|D/O|W/O|C/O)[\s:]+([A-Z][A-Za-z\s]+?)(?:\n|$)',
        ]
        
        for pattern in name_patterns:
            match = re.search(pattern, text, re.MULTILINE)
            if match:
                name = match.group(1).strip()
                # Clean up the name
                name = re.sub(r'\s+', ' ', name)
                if len(name) > 2 and len(name) < 50:
                    return name
        
        return None

    def extract_address_from_text(self, text):
        """Extract address from text"""
        # Look for address patterns
        lines = text.split('\n')
        address_lines = []
        
        # Find lines that might contain address
        for line in lines:
            line = line.strip()
            # Skip lines with common non-address content
            if (len(line) > 5 and 
                not re.search(r'government|aadhaar|unique|identification|authority', line, re.IGNORECASE) and
                not re.search(r'^\d{4}\s*\d{4}\s*\d{4}$', line) and  # Not Aadhaar number
                not re.search(r'^[A-Z]{2,}$', line)):  # Not all caps words
                
                # Check if it contains address-like content
                if (re.search(r'\d', line) or  # Contains numbers
                    any(addr_word in line.lower() for addr_word in 
                        ['road', 'street', 'nagar', 'colony', 'village', 'district', 'state', 'pin'])):
                    address_lines.append(line)
        
        if address_lines:
            return ' '.join(address_lines[:3])  # Take first 3 relevant lines
        
        return None

    def llm_extract(self, text):
        """Use LLM for intelligent extraction"""
        prompt = f"""Extract information from this Aadhaar card OCR text. Handle OCR errors intelligently.

OCR Text: {text}

Extract and return ONLY JSON format:
{{
  "document_type": "Aadhaar Card",
  "confidence": "high/medium/low",
  "data": {{
    "name": "full name",
    "aadhaar_number": "formatted as XXXX XXXX XXXX",
    "date_of_birth": "DD/MM/YYYY or DD-MM-YYYY",
    "gender": "Male/Female",
    "father_name": "father's name if found",
    "mother_name": "mother's name if found",
    "husband_name": "husband's name if found",
    "address": "full address",
    "phone_number": "10 digit number",
    "email": "email if found",
    "pincode": "6 digit pincode",
    "enrollment_number": "enrollment ID if found",
    "vid_number": "16 digit VID if found"
  }}
}}

Rules:
- Only include fields that are clearly identifiable
- Format Aadhaar number with spaces: XXXX XXXX XXXX
- Handle common OCR errors (O->0, I->1, etc.)
- Set confidence based on text quality and data completeness
- Leave fields empty if not found rather than guessing"""

        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            response_text = response.content.strip()
            
            # Extract JSON from response
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start != -1 and json_end > json_start:
                json_text = response_text[json_start:json_end]
                return json.loads(json_text)
                
        except Exception as e:
            print(f"⚠️  LLM extraction failed: {e}")
            return None

    def determine_confidence(self, extracted_data):
        """Determine extraction confidence based on data quality"""
        data = extracted_data.get('data', {})
        
        # Count successfully extracted key fields
        key_fields = ['name', 'aadhaar_number', 'date_of_birth', 'address']
        found_key_fields = sum(1 for field in key_fields if data.get(field))
        
        # Calculate confidence
        if found_key_fields >= 3 and data.get('aadhaar_number'):
            return "high"
        elif found_key_fields >= 2:
            return "medium"
        else:
            return "low"

    def extract(self, raw_text):
        """Main extraction method for Aadhaar cards"""
        try:
            # Preprocess text
            cleaned_text = self.preprocess_text(raw_text)
            
            # Try LLM extraction first
            llm_result = self.llm_extract(cleaned_text)
            
            if llm_result and llm_result.get('data'):
                # Filter out empty values
                filtered_data = {k: v for k, v in llm_result['data'].items() if v}
                llm_result['data'] = filtered_data
                llm_result['confidence'] = self.determine_confidence(llm_result)
                return llm_result
            
            # Fallback to pattern matching
            print("📋 Using pattern matching fallback")
            pattern_data = self.extract_patterns(cleaned_text)
            
            # Try to extract name and address manually
            name = self.extract_name_from_text(cleaned_text)
            if name:
                pattern_data['name'] = name
                
            address = self.extract_address_from_text(cleaned_text)
            if address:
                pattern_data['address'] = address
            
            return {
                "document_type": "Aadhaar Card",
                "confidence": "medium" if pattern_data.get('aadhaar_number') else "low",
                "data": pattern_data,
                "extraction_method": "pattern_matching"
            }
            
        except Exception as e:
            print(f"❌ Aadhaar extraction failed: {e}")
            return {
                "document_type": "Aadhaar Card",
                "confidence": "low",
                "data": {},
                "error": str(e)
            }

# Export all extractor classes
__all__ = ['AadhaarExtractor', 'PANExtractor', 'PassportExtractor']