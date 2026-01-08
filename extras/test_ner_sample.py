"""
Test NER on specific sample text from TEST_SAMPLES_WITH_OUTPUTS.md
"""
import sys
from pathlib import Path
import io

# Fix encoding for Windows console
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from Scripts.ner.ner_extractor import MultilingualNER

# Test text from the document
test_text = """In Lahore, Dr. Muhammad Ali contacted the Pakistan Red Crescent at 
+92-42-6263-2200 requesting 500 tents, 1000 blankets, and medical supplies. 
The United Nations Humanitarian Office is coordinating with NDMA in Islamabad."""

print("=" * 80)
print("NER TEST - Sample from TEST_SAMPLES_WITH_OUTPUTS.md")
print("=" * 80)
print(f"\nInput Text:\n{test_text}\n")

try:
    # Initialize NER
    print("Loading NER model...")
    ner = MultilingualNER()
    
    # Extract entities
    print("Extracting entities...\n")
    entities = ner.extract_all(test_text)
    
    print("EXTRACTION RESULTS:")
    print("-" * 80)
    
    print("\n1. PERSONS:")
    if entities['persons']:
        for person in entities['persons']:
            print(f"   - {person}")
    else:
        print("   (None found)")
    
    print("\n2. ORGANIZATIONS:")
    if entities['organizations']:
        for org in entities['organizations']:
            print(f"   - {org}")
    else:
        print("   (None found)")
    
    print("\n3. LOCATIONS:")
    if entities['locations']:
        for loc in entities['locations']:
            print(f"   - {loc}")
    else:
        print("   (None found)")
    
    print("\n4. PHONE NUMBERS:")
    if entities['phone_numbers']:
        for phone in entities['phone_numbers']:
            print(f"   - {phone}")
    else:
        print("   (None found)")
    
    print("\n5. RESOURCES:")
    if entities['resources']:
        for res in entities['resources']:
            print(f"   - {res}")
    else:
        print("   (None found)")
    
    print("\n" + "=" * 80)
    print("EXPECTED vs ACTUAL COMPARISON:")
    print("=" * 80)
    
    expected = {
        'persons': ['Muhammad Ali'],
        'organizations': ['Pakistan Red Crescent', 'United Nations Humanitarian Office', 'NDMA'],
        'locations': ['Lahore', 'Islamabad'],
        'phone_numbers': ['+92-42-6263-2200'],
        'resources': ['500 tents', '1000 blankets', 'medical supplies']
    }
    
    print("\nPERSONS:")
    print(f"  Expected: {expected['persons']}")
    print(f"  Actual:   {entities['persons']}")
    match = set(e.lower() for e in expected['persons']) == set(e.lower() for e in entities['persons'])
    print(f"  Match: {'✓' if match else '✗'}")
    
    print("\nORGANIZATIONS:")
    print(f"  Expected: {expected['organizations']}")
    print(f"  Actual:   {entities['organizations']}")
    match = set(e.lower() for e in expected['organizations']) == set(e.lower() for e in entities['organizations'])
    print(f"  Match: {'✓' if match else '✗'}")
    
    print("\nLOCATIONS:")
    print(f"  Expected: {expected['locations']}")
    print(f"  Actual:   {entities['locations']}")
    match = set(e.lower() for e in expected['locations']) == set(e.lower() for e in entities['locations'])
    print(f"  Match: {'✓' if match else '✗'}")
    
    print("\nPHONE NUMBERS:")
    print(f"  Expected: {expected['phone_numbers']}")
    print(f"  Actual:   {entities['phone_numbers']}")
    match = set(expected['phone_numbers']) == set(entities['phone_numbers'])
    print(f"  Match: {'✓' if match else '✗'}")
    
    print("\nRESOURCES:")
    print(f"  Expected: {expected['resources']}")
    print(f"  Actual:   {entities['resources']}")
    # Resources are partial matches (we look for keywords)
    print(f"  Match: (Partial - keyword-based extraction)")
    
    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
