"""
Optimized Main Entry Point for ID Card Extractor
Replaces: maincode/mainn.py
"""
import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any, Union, List
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from PIL import Image
import pytesseract
import cv2
import numpy as np
import fitz
import requests
from io import BytesIO
import platform

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from card_structure.adhaar import AadhaarExtractor
from card_structure.pan import PANExtractor
from card_structure.passport import PassportExtractor

# Load environment variables
load_dotenv(Path(__file__).parent.parent / "ocr_env" / ".env")

class OptimizedDocumentExtractor:
    """Optimized document extraction engine"""
    
    def __init__(self):
        """Initialize extractor with LLM and utilities"""
        # Setup Tesseract path for Windows
        if platform.system() == "Windows":
            tesseract_paths = [
                r'C:\Program Files\Tesseract-OCR\tesseract.exe',
                r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
            ]
            for path in tesseract_paths:
                if os.path.exists(path):
                    pytesseract.pytesseract.tesseract_cmd = path
                    break
        
        # Initialize LLM
        api_key = "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"
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY not found in environment variables")
        
        self.llm = ChatOpenAI(
            model=os.getenv("LLM_MODEL", "anthropic/claude-3-haiku"),
            openai_api_base="https://openrouter.ai/api/v1",
            openai_api_key=api_key,
            temperature=0.1,
            max_tokens=4096,
        )
        
        # Initialize extractors
        self.aadhaar_extractor = AadhaarExtractor(self.llm)
        self.pan_extractor = PANExtractor(self.llm)
        self.passport_extractor = PassportExtractor(self.llm)
        
        print("✅ Document extractor initialized successfully")
    
    def preprocess_image(self, image: Image.Image) -> Image.Image:
        """Preprocess image for better OCR"""
        try:
            img_array = np.array(image)
            
            # Convert to grayscale
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array
            
            # Denoise
            denoised = cv2.medianBlur(gray, 3)
            
            # Enhance contrast
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(denoised)
            
            # Adaptive thresholding
            thresh = cv2.adaptiveThreshold(
                enhanced, 255, 
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
            )
            
            return Image.fromarray(thresh)
        except Exception as e:
            print(f"⚠️  Preprocessing failed: {e}, using original")
            return image
    
    def extract_text_ocr(self, image: Image.Image) -> str:
        """Extract text using OCR with multiple strategies"""
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        ocr_configs = [
            '--psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz /,-.',
            '--psm 4',
            '--psm 6',
        ]
        
        best_text = ""
        max_length = 0
        
        # Try original image first
        for config in ocr_configs:
            try:
                text = pytesseract.image_to_string(image, config=config).strip()
                if len(text) > max_length:
                    best_text = text
                    max_length = len(text)
                
                if max_length > 50:
                    break
            except:
                continue
        
        # Try preprocessed if needed
        if max_length < 30:
            processed = self.preprocess_image(image)
            for config in ocr_configs[:2]:
                try:
                    text = pytesseract.image_to_string(processed, config=config).strip()
                    if len(text) > max_length:
                        best_text = text
                        max_length = len(text)
                except:
                    continue
        
        print(f"📝 Extracted {max_length} characters")
        return best_text
    
    def detect_document_type(self, text: str) -> str:
        """Detect document type from text"""
        import re
        
        text_lower = text.lower()
        
        # Aadhaar patterns
        if (re.search(r'\b\d{4}\s*\d{4}\s*\d{4}\b', text) or 
            any(k in text_lower for k in ['aadhaar', 'aadhar', 'uid'])):
            return 'aadhaar'
        
        # PAN patterns
        if (re.search(r'\b[A-Z]{5}\d{4}[A-Z]\b', text) or 
            'pan' in text_lower or 'permanent account' in text_lower):
            return 'pan'
        
        # Passport patterns
        if (re.search(r'\b[A-Z]\d{7}\b', text) or 
            'passport' in text_lower or 'republic of india' in text_lower):
            return 'passport'
        
        return 'unknown'
    
    def extract_structured_data(self, text: str, doc_type: str) -> Dict[str, Any]:
        """Extract structured data based on document type"""
        try:
            if doc_type == 'aadhaar':
                return self.aadhaar_extractor.extract(text)
            elif doc_type == 'pan':
                return self.pan_extractor.extract(text)
            elif doc_type == 'passport':
                return self.passport_extractor.extract(text)
            else:
                return {
                    "document_type": doc_type,
                    "confidence": "low",
                    "data": {}
                }
        except Exception as e:
            print(f"❌ Extraction failed: {e}")
            return {
                "document_type": doc_type,
                "confidence": "low",
                "data": {},
                "error": str(e)
            }
    
    def load_image_from_url(self, url: str) -> Optional[Image.Image]:
        """Load image from URL"""
        try:
            print(f"🔗 Downloading from URL...")
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            return Image.open(BytesIO(response.content))
        except Exception as e:
            print(f"❌ URL download failed: {e}")
            return None
    
    def load_image_from_file(self, file_path: str) -> Optional[Image.Image]:
        """Load image from file"""
        try:
            return Image.open(file_path)
        except Exception as e:
            print(f"❌ File load failed: {e}")
            return None
    
    def load_images_from_pdf(self, pdf_path: str, max_pages: int = 5) -> List[Image.Image]:
        """Extract images from PDF"""
        images = []
        try:
            pdf = fitz.open(pdf_path)
            pages_to_process = min(len(pdf), max_pages)
            
            print(f"📄 Processing {pages_to_process} PDF pages...")
            mat = fitz.Matrix(2.0, 2.0)
            
            for page_num in range(pages_to_process):
                page = pdf.load_page(page_num)
                pix = page.get_pixmap(matrix=mat)
                img = Image.open(BytesIO(pix.tobytes("png")))
                images.append(img)
            
            pdf.close()
            print(f"✅ Extracted {len(images)} images from PDF")
        except Exception as e:
            print(f"❌ PDF processing failed: {e}")
        
        return images
    
    def is_url(self, path: str) -> bool:
        """Check if path is URL"""
        return path.startswith(('http://', 'https://'))
    
    def process_single_image(self, image: Image.Image, source: str = "Image") -> Dict[str, Any]:
        """Process single image"""
        print(f"\n{'='*60}")
        print(f"🔍 Processing: {source}")
        print(f"{'='*60}")
        
        # Resize if too large
        width, height = image.size
        if width > 2000 or height > 2000:
            ratio = min(2000 / width, 2000 / height)
            new_size = (int(width * ratio), int(height * ratio))
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        
        # Extract text
        raw_text = self.extract_text_ocr(image)
        
        if not raw_text or len(raw_text) < 5:
            print("❌ Insufficient text extracted")
            return {"error": "Insufficient text", "raw_text": raw_text}
        
        # Detect type
        doc_type = self.detect_document_type(raw_text)
        print(f"📋 Document Type: {doc_type.upper()}")
        
        # Extract structured data
        structured = self.extract_structured_data(raw_text, doc_type)
        
        confidence = structured.get("confidence", "unknown")
        print(f"📊 Confidence: {confidence.upper()}")
        
        if structured.get("data"):
            print(f"✅ Extracted {len(structured['data'])} fields")
        
        print(f"{'='*60}\n")
        
        return {
            "source": source,
            "document_type": doc_type,
            "raw_text": raw_text,
            "extracted_data": structured
        }
    
    def process_input(self, input_path: str) -> Dict[str, Any]:
        """Process any input type"""
        # Load image(s)
        if self.is_url(input_path):
            image = self.load_image_from_url(input_path)
            if image is None:
                return {"error": "Failed to load from URL"}
            return self.process_single_image(image, "URL")
        
        path = Path(input_path)
        if not path.exists():
            return {"error": f"File not found: {input_path}"}
        
        ext = path.suffix.lower()
        
        # PDF
        if ext == '.pdf':
            images = self.load_images_from_pdf(str(path))
            if not images:
                return {"error": "No images from PDF"}
            
            # Process all pages, return best
            results = []
            for i, img in enumerate(images):
                result = self.process_single_image(img, f"PDF Page {i+1}")
                if "error" not in result:
                    results.append(result)
            
            if not results:
                return {"error": "No successful extractions"}
            
            # Return highest confidence
            best = max(results, key=lambda x: {
                "high": 3, "medium": 2, "low": 1
            }.get(x.get("extracted_data", {}).get("confidence", "low"), 0))
            
            return best
        
        # Image file
        elif ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif']:
            image = self.load_image_from_file(str(path))
            if image is None:
                return {"error": "Failed to load image"}
            return self.process_single_image(image, path.name)
        
        else:
            return {"error": f"Unsupported file type: {ext}"}


def main_run(input_path: str, save_output: bool = True):
    """Main function to run extraction"""
    try:
        print("\n" + "="*60)
        print("🚀 ID CARD EXTRACTOR - STARTING")
        print("="*60 + "\n")
        
        # Create extractor
        extractor = OptimizedDocumentExtractor()
        
        # Process input
        result = extractor.process_input(input_path)
        
        # Check for errors
        if "error" in result:
            print(f"\n❌ ERROR: {result['error']}\n")
            return result
        
        # Save output if requested
        if save_output and "extracted_data" in result:
            output_dir = Path(__file__).parent / "extracted_data"
            output_dir.mkdir(exist_ok=True)
            
            output_file = output_dir / "extracted_data.json"
            
            import json
            data_to_save = result["extracted_data"].get("data", {})
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data_to_save, f, indent=2, ensure_ascii=False)
            
            print(f"\n💾 Data saved to: {output_file.name}")
        
        print("\n" + "="*60)
        print("✅ EXTRACTION COMPLETED SUCCESSFULLY")
        print("="*60 + "\n")
        
        return result
        
    except Exception as e:
        print(f"\n❌ FATAL ERROR: {e}\n")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Command line argument provided
        input_path = sys.argv[1]
        main_run(input_path)
    else:
        # Interactive mode - ask for file path
        print("\n" + "="*60)
        print("🚀 ID CARD EXTRACTOR - INTERACTIVE MODE")
        print("="*60 + "\n")
        
        input_path = input("📁 Drop or paste your file path here: ").strip()
        
        # Remove quotes if user dragged file (Windows adds quotes)
        input_path = input_path.strip('"').strip("'")
        
        if not input_path:
            print("❌ No file path provided")
        else:
            main_run(input_path)