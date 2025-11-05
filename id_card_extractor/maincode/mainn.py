import os
import sys
import base64
import requests
from io import BytesIO
from PIL import Image
import pytesseract
import fitz  # PyMuPDF
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import json
import re
import platform
import cv2
import numpy as np
from urllib.parse import urlparse
import time
from pathlib import Path

# Append project root to import custom modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# ✅ Import extractors and form filler
from card_structure.adhaar import AadhaarExtractor
from card_structure.passport import PassportExtractor
from card_structure.pan import PANExtractor
from form_filler.filler_llm import fill_itr_with_llm

# Configure Tesseract path for Windows (optional)
if platform.system() == "Windows":
    possible_paths = [
        r'C:\Program Files\Tesseract-OCR\tesseract.exe',
        r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
        rf'C:\Users\{os.getenv("USERNAME")}\AppData\Local\Tesseract-OCR\tesseract.exe'
    ]
    for path in possible_paths:
        if os.path.exists(path):
            pytesseract.pytesseract.tesseract_cmd = path
            break

class OptimizedDocumentExtractor:
    def __init__(self):
        self.llm = ChatOpenAI(
            model="anthropic/claude-3-haiku",
            openai_api_base="https://openrouter.ai/api/v1",
            openai_api_key="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",
            temperature=0.1,
            max_tokens=4096,
        )
        self.aadhaar_extractor = AadhaarExtractor(self.llm)
        self.passport_extractor = PassportExtractor(self.llm)
        self.pan_extractor = PANExtractor(self.llm)
        self.ocr_configs = [
            '--psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz /,-.',
            '--psm 4',
            '--psm 6',
            '--psm 11'
        ]
        self.detection_patterns = {
            'aadhar_number': r'\b\d{4}\s*\d{4}\s*\d{4}\b',
            'pan_number': r'\b[A-Z]{5}\d{4}[A-Z]\b',
            'passport_number': r'\b[A-Z]\d{7}\b',
            'driving_license': r'\b[A-Z]{2}\d{2}\s*\d{11}\b',
            'voter_id': r'\b[A-Z]{3}\d{7}\b',
        }

    def download_image_from_url(self, url):
        try:
            print(f"📥 Downloading image from URL...")
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            return Image.open(BytesIO(response.content))
        except Exception as e:
            print(f"❌ Error downloading image: {e}")
            return None

    def is_url(self, path):
        try:
            result = urlparse(path)
            return all([result.scheme, result.netloc])
        except:
            return False

    def fast_preprocess_image(self, image):
        try:
            img_array = np.array(image)
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY) if len(img_array.shape) == 3 else img_array
            denoised = cv2.medianBlur(gray, 3)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(denoised)
            thresh = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY, 11, 2)
            return Image.fromarray(thresh)
        except Exception as e:
            print(f"⚠️ Preprocessing failed: {e}")
            return image

    def smart_ocr_extract(self, image):
        best_text = ""
        if image.mode != 'RGB':
            image = image.convert('RGB')
        for config in self.ocr_configs:
            try:
                text = pytesseract.image_to_string(image, config=config).strip()
                if len(text) > len(best_text):
                    best_text = text
                    if len(text) > 50:
                        break
            except:
                continue
        if len(best_text) < 30:
            processed_image = self.fast_preprocess_image(image)
            for config in self.ocr_configs[:2]:
                try:
                    text = pytesseract.image_to_string(processed_image, config=config).strip()
                    if len(text) > len(best_text):
                        best_text = text
                except:
                    continue
        return best_text

    def detect_document_type(self, raw_text):
        text_lower = raw_text.lower()
        if re.search(self.detection_patterns['aadhar_number'], raw_text) or 'aadhaar' in text_lower:
            return 'aadhaar'
        elif re.search(self.detection_patterns['pan_number'], raw_text) or 'pan' in text_lower:
            return 'pan'
        elif re.search(self.detection_patterns['passport_number'], raw_text) or 'passport' in text_lower:
            return 'passport'
        elif 'driving' in text_lower or 'licence' in text_lower:
            return 'driving_license'
        elif 'voter' in text_lower or 'election' in text_lower:
            return 'voter_id'
        return 'unknown'

    def extract_document_data(self, raw_text, doc_type):
        try:
            if doc_type == 'aadhaar':
                return self.aadhaar_extractor.extract(raw_text)
            elif doc_type == 'pan':
                return self.pan_extractor.extract(raw_text)
            elif doc_type == 'passport':
                return self.passport_extractor.extract(raw_text)
            else:
                return {
                    "document_type": doc_type,
                    "confidence": "low",
                    "data": {},
                    "note": f"Extractor for {doc_type} not implemented"
                }
        except Exception as e:
            return {
                "document_type": doc_type,
                "confidence": "low",
                "data": {},
                "error": str(e)
            }

    def process_pdf_fast(self, pdf_path):
        pdf = fitz.open(pdf_path)
        results = []
        for page_num in range(min(len(pdf), 5)):
            page = pdf.load_page(page_num)
            mat = fitz.Matrix(2, 2)
            pix = page.get_pixmap(matrix=mat)
            img_pil = Image.open(BytesIO(pix.tobytes("png")))
            result = self.process_single_image(img_pil, f"PDF Page {page_num + 1}")
            if result and 'error' not in result:
                results.append(result)
        pdf.close()
        return results if results else [{"error": "No extractable content"}]

    def process_single_image(self, image, source_name="Image"):
        print(f"🔍 Processing {source_name}...")
        raw_text = self.smart_ocr_extract(image)
        if not raw_text or len(raw_text.strip()) < 5:
            return {"error": "Insufficient text extracted"}
        doc_type = self.detect_document_type(raw_text)
        structured_data = self.extract_document_data(raw_text, doc_type)
        return {
            "source": source_name,
            "raw_text": raw_text,
            "extracted_data": structured_data
        }

    def process_input(self, input_path):
        if self.is_url(input_path):
            image = self.download_image_from_url(input_path)
            if not image:
                return {"error": "Failed to download image"}
            return self.process_single_image(image, "URL Image")
        elif not os.path.exists(input_path):
            return {"error": f"File not found: {input_path}"}
        else:
            ext = Path(input_path).suffix.lower()
            if ext == '.pdf':
                return self.process_pdf_fast(input_path)
            elif ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif', '.webp']:
                image = Image.open(input_path)
                return self.process_single_image(image, Path(input_path).name)
            else:
                return {"error": f"Unsupported file format: {ext}"}

# ✅ RUN THIS FUNCTION
def main_run(path):
    extractor = OptimizedDocumentExtractor()
    result = extractor.process_input(path)

    # Extract the structured data from the result
    if isinstance(result, list):
        result = result[0]

    if 'extracted_data' in result:
        extracted_info = result['extracted_data']['data']
        os.makedirs("extracted_data", exist_ok=True)
        with open("extracted_data/extracted_data.json", "w") as f:
            json.dump(extracted_info, f, indent=2)
        print("✅ Extracted data saved to extracted_data/extracted_data.json")

        # Trigger form filling using LLM
        fill_itr_with_llm(
            template_path=os.path.join(os.path.dirname(__file__), "..", "forms", "ITR TEST FORM.docx"),
            output_path=os.path.join(os.path.dirname(__file__), "..", "filled_forms", "filled_itr.docx"),
            json_data=extracted_info
        )

    else:
        print("❌ No extracted data found")

if __name__ == "__main__":
    path = input("📁 Enter image or PDF path: ").strip()
    main_run(path)
