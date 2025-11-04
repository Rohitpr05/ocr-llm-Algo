"""
Simple Run Script - ID Card Extractor
Usage: python run.py
"""
import sys
from pathlib import Path

def main():
    """Interactive mode for easy usage"""
    print("\n" + "="*60)
    print("🚀 ID CARD EXTRACTOR")
    print("="*60)
    
    print("\nChoose operation:")
    print("1. Extract data from document")
    print("2. Fill form with extracted data")
    print("3. Extract and fill (both operations)")
    
    choice = input("\nEnter choice (1/2/3): ").strip()
    
    if choice == "1":
        # Extract data
        input_path = input("Enter document path: ").strip()
        if not input_path:
            print("❌ No path provided")
            return
        
        print()
        from maincode.mainn import main_run
        main_run(input_path)
    
    elif choice == "2":
        # Fill form
        template = input("Enter template name (or press Enter for default): ").strip()
        if not template:
            template = "ITR TEST FORM.docx"
        
        print()
        from form_filler.filler_llm import fill_itr_form
        fill_itr_form(template)
    
    elif choice == "3":
        # Both
        input_path = input("Enter document path: ").strip()
        if not input_path:
            print("❌ No path provided")
            return
        
        template = input("Enter template name (or press Enter for default): ").strip()
        if not template:
            template = "ITR TEST FORM.docx"
        
        print()
        from maincode.mainn import main_run
        from form_filler.filler_llm import fill_itr_form
        
        # Extract
        result = main_run(input_path)
        
        # Fill if extraction successful
        if "error" not in result:
            fill_itr_form(template)
    
    else:
        print("❌ Invalid choice")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user\n")
    except Exception as e:
        print(f"\n❌ Error: {e}\n")