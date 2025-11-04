"""
Optimized Form Filler
Replaces: form_filler/filler_llm.py
"""
import os
import json
from pathlib import Path
from datetime import datetime
from docx import Document
from docx.shared import Inches
from typing import Dict, Any, Optional


def find_signature() -> Optional[Path]:
    """Find signature file"""
    base_dir = Path(__file__).parent.parent
    signature_folders = [
        base_dir / "signatures",
        base_dir / "signature"
    ]
    
    for folder in signature_folders:
        if not folder.exists():
            continue
        
        for ext in ['png', 'jpg', 'jpeg', 'PNG', 'JPG', 'JPEG']:
            sig_file = folder / f"signature.{ext}"
            if sig_file.exists():
                print(f"✅ Found signature: {sig_file.name}")
                return sig_file
    
    print("⚠️  No signature file found")
    return None


def prepare_replacements(data: Dict[str, Any]) -> Dict[str, str]:
    """Prepare placeholder replacements"""
    current_date = datetime.now().strftime("%d-%m-%Y")
    
    replacements = {
        "{{NAME}}": data.get("name", ""),
        "{{PAN}}": data.get("pan_number", data.get("pan", "")),
        "{{FATHER_NAME}}": data.get("father_name", ""),
        "{{ACK_NO}}": data.get("ack_no", data.get("acknowledgement_number", "")),
        "{{CAPACITY}}": data.get("capacity", "Self"),
        "{{DATE}}": current_date,
        "{{SIGNATURE}}": "[SIGNATURE_PLACEHOLDER]",
        "{{DOB}}": data.get("date_of_birth", ""),
        "{{ADDRESS}}": data.get("address", ""),
        "{{PHONE}}": data.get("phone_number", ""),
        "{{EMAIL}}": data.get("email", ""),
        "{{PASSPORT_NUMBER}}": data.get("passport_number", ""),
        "{{AADHAAR_NUMBER}}": data.get("aadhaar_number", ""),
    }
    
    # Convert to strings
    replacements = {k: str(v) if v else "" for k, v in replacements.items()}
    
    print(f"📝 Prepared {len(replacements)} placeholder replacements")
    return replacements


def replace_text_in_paragraph(paragraph, replacements: Dict[str, str]) -> int:
    """Replace text in paragraph"""
    if not paragraph.runs:
        return 0
    
    full_text = ''.join(run.text for run in paragraph.runs)
    updated_text = full_text
    count = 0
    
    for placeholder, replacement in replacements.items():
        if placeholder in updated_text:
            updated_text = updated_text.replace(placeholder, replacement)
            count += 1
    
    if updated_text != full_text:
        # Clear and rebuild
        for run in paragraph.runs[::-1]:
            paragraph._element.remove(run._element)
        paragraph.add_run(updated_text)
    
    return count


def handle_signature(paragraph, signature_path: Optional[Path]) -> bool:
    """Handle signature placeholder"""
    full_text = ''.join(run.text for run in paragraph.runs)
    
    if "{{SIGNATURE}}" not in full_text:
        return False
    
    # Save formatting
    try:
        original_alignment = paragraph.alignment
        original_space_before = paragraph.paragraph_format.space_before
        original_space_after = paragraph.paragraph_format.space_after
    except:
        original_alignment = None
        original_space_before = None
        original_space_after = None
    
    # Clear paragraph
    paragraph.clear()
    
    # Insert signature or placeholder
    if signature_path and signature_path.exists():
        try:
            run = paragraph.add_run()
            run.add_picture(str(signature_path), width=Inches(1.2), height=Inches(0.6))
            print("✅ Signature inserted")
        except Exception as e:
            print(f"⚠️  Signature insertion failed: {e}")
            paragraph.add_run("[SIGNATURE MISSING]")
    else:
        paragraph.add_run("[SIGNATURE MISSING]")
    
    # Restore formatting
    try:
        if original_alignment:
            paragraph.alignment = original_alignment
        if original_space_before:
            paragraph.paragraph_format.space_before = original_space_before
        if original_space_after:
            paragraph.paragraph_format.space_after = original_space_after
    except:
        pass
    
    return True


def process_table(table, replacements: Dict[str, str], signature_path: Optional[Path]) -> int:
    """Process table cells"""
    count = 0
    
    for row in table.rows:
        for cell in row.cells:
            for paragraph in cell.paragraphs:
                if not handle_signature(paragraph, signature_path):
                    count += replace_text_in_paragraph(paragraph, replacements)
            
            for nested_table in cell.tables:
                count += process_table(nested_table, replacements, signature_path)
    
    return count


def optimize_layout(doc: Document):
    """Optimize document layout"""
    try:
        for paragraph in doc.paragraphs:
            try:
                if paragraph.paragraph_format.space_before and paragraph.paragraph_format.space_before.pt > 12:
                    paragraph.paragraph_format.space_before = None
                if paragraph.paragraph_format.space_after and paragraph.paragraph_format.space_after.pt > 12:
                    paragraph.paragraph_format.space_after = None
            except:
                pass
        
        for table in doc.tables:
            try:
                table.alignment = 0
            except:
                pass
            
            for row in table.rows:
                try:
                    row.height_rule = 1
                except:
                    pass
                
                for cell in row.cells:
                    for paragraph in cell.paragraphs:
                        try:
                            if paragraph.paragraph_format.space_before and paragraph.paragraph_format.space_before.pt > 6:
                                paragraph.paragraph_format.space_before = None
                            if paragraph.paragraph_format.space_after and paragraph.paragraph_format.space_after.pt > 6:
                                paragraph.paragraph_format.space_after = None
                        except:
                            pass
    except Exception as e:
        print(f"⚠️  Layout optimization failed: {e}")


def fill_itr_with_llm(template_path: str, output_path: str, json_data: Dict[str, Any]) -> bool:
    """Fill ITR form with data"""
    try:
        print("\n" + "="*60)
        print("✍️  FORM FILLING - STARTING")
        print("="*60 + "\n")
        
        # Check template
        template_path = Path(template_path)
        if not template_path.exists():
            print(f"❌ Template not found: {template_path}")
            return False
        
        print(f"📄 Template: {template_path.name}")
        
        # Load document
        doc = Document(template_path)
        
        # Find signature
        signature_path = find_signature()
        
        # Prepare replacements
        replacements = prepare_replacements(json_data)
        
        # Process document
        replacement_count = 0
        
        # Main paragraphs
        for paragraph in doc.paragraphs:
            if not handle_signature(paragraph, signature_path):
                replacement_count += replace_text_in_paragraph(paragraph, replacements)
        
        # Tables
        for table in doc.tables:
            replacement_count += process_table(table, replacements, signature_path)
        
        # Headers and footers
        for section in doc.sections:
            if section.header:
                for paragraph in section.header.paragraphs:
                    replace_text_in_paragraph(paragraph, replacements)
                for table in section.header.tables:
                    process_table(table, replacements, signature_path)
            
            if section.footer:
                for paragraph in section.footer.paragraphs:
                    replace_text_in_paragraph(paragraph, replacements)
                for table in section.footer.tables:
                    process_table(table, replacements, signature_path)
        
        # Optimize layout
        optimize_layout(doc)
        
        # Save
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        doc.save(output_path)
        
        print(f"💾 Form saved: {output_path.name}")
        print(f"✅ Filled {replacement_count} placeholders")
        
        print("\n" + "="*60)
        print("✅ FORM FILLING COMPLETED")
        print("="*60 + "\n")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Form filling failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def fill_itr_form(template_name: str = "ITR TEST FORM.docx", 
                  data_file: Optional[str] = None,
                  output_name: Optional[str] = None) -> bool:
    """Fill ITR form - convenience function"""
    try:
        base_dir = Path(__file__).parent.parent
        
        # Template path
        template_path = base_dir / "forms" / template_name
        
        # Data file
        if data_file is None:
            data_file = base_dir / "maincode" / "extracted_data" / "extracted_data.json"
        else:
            data_file = Path(data_file)
        
        # Output file
        if output_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_name = f"filled_itr_{timestamp}.docx"
        
        output_path = base_dir / "filled_forms" / output_name
        
        # Load data
        if not data_file.exists():
            print(f"❌ Data file not found: {data_file}")
            return False
        
        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"📊 Loaded data from: {data_file.name}")
        
        # Fill form
        return fill_itr_with_llm(str(template_path), str(output_path), data)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        template = sys.argv[1]
        fill_itr_form(template)
    else:
        fill_itr_form()