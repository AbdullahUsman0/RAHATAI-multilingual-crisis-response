#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
NER Extraction Test - Multiple Sample Texts
Tests NER extraction on various crisis-related text samples
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Scripts'))

from ner.ner_extractor import MultilingualNER

# Enable UTF-8 console output on Windows
if sys.platform.startswith('win'):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def test_samples():
    """Test NER on multiple sample texts"""
    
    print("\n" + "="*80)
    print("NER EXTRACTION TEST - Multiple Samples")
    print("="*80 + "\n")
    
    # Initialize NER
    print("Loading NER model...")
    ner = MultilingualNER()
    print("Model loaded.\n")
    
    # Test samples
    samples = [
        {
            "text": "In Lahore, Dr. Muhammad Ali contacted the Pakistan Red Crescent at +92-42-6263-2200 requesting 500 tents, 1000 blankets, and medical supplies. The United Nations Humanitarian Office is coordinating with NDMA in Islamabad.",
            "name": "Crisis Response - Main Sample"
        },
        {
            "text": "Dr. Sarah Khan from WHO office in Karachi reported 200 patients at Shaukat Khanum Hospital. Contact: +92-21-3456-7890",
            "name": "Medical Emergency"
        },
        {
            "text": "The earthquake in Mingora caused severe damage. Mr. Hassan Ahmed coordinated with Edhi Foundation and Save the Children. Distributed 300 sleeping bags and food supplies.",
            "name": "Natural Disaster Response"
        },
        {
            "text": "NDMA emergency team headed by Dr. Fatima Ahmed responded to flood in Sukkur. Contact Aga Khan Hospital at +92-71-5555-1111.",
            "name": "Flood Response"
        },
        {
            "text": "Prof. Ahmed from UNICEF office in Islamabad requested 500 water containers, 200 medicines, and medical kits for the IDP camp.",
            "name": "Humanitarian Aid Request"
        },
    ]
    
    # Run tests
    for i, sample in enumerate(samples, 1):
        print(f"\n{'='*80}")
        print(f"TEST {i}: {sample['name']}")
        print(f"{'='*80}")
        print(f"\nText: {sample['text']}\n")
        
        # Extract entities
        entities = ner.extract_all(sample['text'])
        
        # Display results
        print("RESULTS:")
        print("-" * 80)
        
        print("\n1. PERSONS:")
        if entities['persons']:
            for person in entities['persons']:
                print(f"   - {person}")
        else:
            print("   [None extracted]")
        
        print("\n2. ORGANIZATIONS:")
        if entities['organizations']:
            for org in entities['organizations']:
                print(f"   - {org}")
        else:
            print("   [None extracted]")
        
        print("\n3. LOCATIONS:")
        if entities['locations']:
            for loc in entities['locations']:
                print(f"   - {loc}")
        else:
            print("   [None extracted]")
        
        print("\n4. PHONE NUMBERS:")
        if entities['phone_numbers']:
            for phone in entities['phone_numbers']:
                print(f"   - {phone}")
        else:
            print("   [None extracted]")
        
        print("\n5. RESOURCES:")
        if entities['resources']:
            for resource in entities['resources']:
                print(f"   - {resource}")
        else:
            print("   [None extracted]")
    
    print(f"\n{'='*80}")
    print("TEST COMPLETE")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    test_samples()
