"""
Multilingual Named Entity Recognition (NER)
Extracts: locations, phone numbers, resources, persons, organizations
Supports: English, Urdu, Roman-Urdu
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import re
import torch

# Configure DirectML for AMD GPU support
# Note: torch-directml may not be available as a standard package
DIRECTML_AVAILABLE = False
try:
    import torch_directml
    DIRECTML_AVAILABLE = True
except ImportError:
    DIRECTML_AVAILABLE = False

from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from typing import List, Dict
import pandas as pd

from Scripts.utils import load_config

class MultilingualNER:
    """
    Multilingual NER for crisis text
    """
    
    def __init__(self, model_name: str = "xlm-roberta-base", device: str = None):
        """
        Initialize NER model
        
        Args:
            model_name: HuggingFace model name
            device: Device to use ('cuda' or 'cpu')
        """
        # Set device - prioritize DirectML for AMD GPU
        # Note: transformers pipeline uses device index, not torch.device
        if device is None:
            if DIRECTML_AVAILABLE:
                try:
                    # For DirectML, we use device 0 (DirectML device)
                    self.device = 0
                except:
                    self.device = 0 if torch.cuda.is_available() else -1
            elif torch.cuda.is_available():
                self.device = 0
            else:
                self.device = -1
        else:
            if device == 'directml' and DIRECTML_AVAILABLE:
                self.device = 0
            elif device == 'cuda':
                self.device = 0
            else:
                self.device = -1
        
        # Initialize NER pipeline
        try:
            self.ner_pipeline = pipeline(
                "ner",
                model=model_name,
                aggregation_strategy="simple",
                device=self.device
            )
        except:
            # Fallback to a different model
            print("Warning: Using default NER model")
            self.ner_pipeline = pipeline(
                "ner",
                model="dbmdz/bert-large-cased-finetuned-conll03-english",
                aggregation_strategy="simple",
                device=self.device
            )
    
    def extract_entities(self, text: str) -> List[Dict]:
        """
        Extract named entities from text
        
        Args:
            text: Input text
            
        Returns:
            list: List of entities with 'word', 'score', 'entity_group'
        """
        if not text or pd.isna(text):
            return []
        
        try:
            entities = self.ner_pipeline(text)
            return entities
        except Exception as e:
            print(f"Error in NER: {e}")
            return []
    
    def extract_phone_numbers(self, text: str) -> List[str]:
        """
        Extract phone numbers using regex
        
        Args:
            text: Input text
            
        Returns:
            list: List of phone numbers
        """
        # Pakistani phone number patterns
        patterns = [
            r'\+92-\d{1,2}-\d{4}-\d{4}',                        # +92-42-6263-2200
            r'\+92\s\d{1,2}\s\d{4}\s\d{4}',                     # +92 42 6263 2200
            r'\+92[.\s]?\d{2}[.\s]?\d{4}[.\s]?\d{4}',           # +92.42.6263.2200
            r'\+92\s?\d{2}\s?\d{7}',                            # +92 XX XXXXXXX
            r'03\d{2}-?\d{4}-?\d{4}',                           # 0300-1234-5678
            r'0\d{2}[\s-]?\d{7}',                               # 0XX-XXXXXXX
            r'\d{4}[\s-]?\d{7}',                                # XXXX-XXXXXXX
        ]
        
        phone_numbers = []
        for pattern in patterns:
            matches = re.findall(pattern, text)
            phone_numbers.extend(matches)
        
        # Deduplicate while preserving original format (with dashes)
        seen = set()
        cleaned_numbers = []
        for phone in phone_numbers:
            # Normalize for comparison (remove spaces/dashes/dots for dedup)
            normalized = re.sub(r'[\s\-.]', '', phone)
            if normalized not in seen:
                seen.add(normalized)
                cleaned_numbers.append(phone)
        
        return cleaned_numbers
    
    def extract_locations(self, text: str) -> List[str]:
        """
        Extract location entities using both NER model and regex patterns
        
        Args:
            text: Input text
            
        Returns:
            list: List of locations
        """
        locations = []
        
        # Method 1: Use NER model
        entities = self.extract_entities(text)
        ner_locations = [
            e['word'] for e in entities 
            if e.get('entity_group', '').upper() in ['LOC', 'LOCATION', 'GPE', 'MISC']
        ]
        locations.extend(ner_locations)
        
        # Method 2: Regex patterns for common location indicators
        # Pakistani cities list
        pakistani_cities = [
            'Karachi', 'Lahore', 'Islamabad', 'Rawalpindi', 'Faisalabad', 'Multan', 'Hyderabad', 
            'Peshawar', 'Quetta', 'Sialkot', 'Gujranwala', 'Bahawalpur', 'Sargodha', 'Sukkur', 
            'Larkana', 'Sheikhupura', 'Rahim Yar Khan', 'Gujrat', 'Kasur', 'Mardan', 'Mingora', 
            'Nawabshah', 'Chiniot', 'Kotri', 'Khanpur', 'Hafizabad', 'Kohat', 'Jacobabad', 
            'Shikarpur', 'Muzaffargarh', 'Khanewal', 'Gojra', 'Bahawalnagar', 'Abbottabad', 
            'Muridke', 'Pakpattan', 'Chishtian', 'Daska', 'Mandi Bahauddin', 'Ahmadpur East', 
            'Kamalia', 'Tando Adam', 'Khairpur', 'Dera Ghazi Khan', 'Kot Addu', 'Vehari', 
            'Nowshera', 'Charsadda', 'Jhelum', 'Mianwali', 'Sadiqabad', 'Okara', 'Sahiwal', 
            'Mirpur Khas', 'Chaman', 'Kandhkot', 'Burewala', 'Jaranwala', 'Hasilpur', 'Attock', 
            'Muzaffarabad', 'Gwadar', 'Gilgit', 'Skardu'
        ]
        
        # Check for Pakistani cities
        text_lower = text.lower()
        for city in pakistani_cities:
            if city.lower() in text_lower:
                locations.append(city)
        
        # Pattern: "in/at/near [Location]" - but stop at "and" or comma
        pattern1 = r'\b(in|at|near|from|to)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)(?:\s+and|\s*[,.]|$)'
        matches = re.finditer(pattern1, text, re.IGNORECASE)
        for match in matches:
            if match.lastindex and match.lastindex >= 2:
                loc = match.group(2)
                if loc and loc.strip() and len(loc.strip()) > 2:
                    # Split if contains "and" - extract individual cities
                    if ' and ' in loc:
                        parts = loc.split(' and ')
                        for part in parts:
                            if part.strip() and len(part.strip()) > 2:
                                locations.append(part.strip())
                    else:
                        locations.append(loc.strip())
        
        # Pattern: "[Location] city/town/village/district"
        pattern2 = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(city|town|village|district|province|area|region)'
        matches = re.finditer(pattern2, text, re.IGNORECASE)
        for match in matches:
            if match.lastindex and match.lastindex >= 1:
                loc = match.group(1)
                if loc and loc.strip():
                    locations.append(loc.strip())
        
        # Pattern: "[Location] Pakistan"
        pattern3 = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+Pakistan\b'
        matches = re.finditer(pattern3, text, re.IGNORECASE)
        for match in matches:
            if match.lastindex and match.lastindex >= 1:
                loc = match.group(1)
                if loc and loc.strip():
                    locations.append(loc.strip())
        
        # Method 3: Common location keywords followed by place names
        location_keywords = ['emergency in', 'flood in', 'earthquake in', 'disaster in', 'crisis in', 
                           'located in', 'situated in', 'based in']
        for keyword in location_keywords:
            pattern = rf'\b{keyword}\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)'
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                if match.lastindex and match.lastindex >= 1:
                    loc = match.group(1)
                    if loc and loc.strip():
                        locations.append(loc.strip())
        
        # Clean and deduplicate
        cleaned_locations = []
        seen = set()
        
        for loc in locations:
            if not loc or len(loc.strip()) < 2:
                continue
            
            loc = loc.strip()
            
            # Split locations joined by "and" or comma
            if ' and ' in loc.lower():
                parts = re.split(r'\s+and\s+', loc, flags=re.IGNORECASE)
                for part in parts:
                    part = part.strip()
                    if part and len(part) > 2:
                        # Remove trailing words
                        part = re.sub(r'\s+(at|in|on|to|from|near)\s*$', '', part, flags=re.IGNORECASE).strip()
                        if part and len(part) > 2:
                            part_lower = part.lower()
                            if part_lower not in seen:
                                seen.add(part_lower)
                                cleaned_locations.append(part)
                continue
            
            # Handle comma-separated locations
            if ',' in loc:
                parts = loc.split(',')
                for part in parts:
                    part = part.strip()
                    if part and len(part) > 2:
                        part = re.sub(r'\s+(at|in|on|to|from|near)\s*$', '', part, flags=re.IGNORECASE).strip()
                        if part and len(part) > 2:
                            part_lower = part.lower()
                            if part_lower not in seen:
                                seen.add(part_lower)
                                cleaned_locations.append(part)
                continue
            
            # Remove trailing words like "at", "and", "in", etc.
            loc = re.sub(r'\s+(at|and|in|on|to|from|near)\s*$', '', loc, flags=re.IGNORECASE)
            loc = loc.strip()
            
            # Skip if too short or empty after cleaning
            if len(loc) < 2:
                continue
            
        # Remove common false positives (words that aren't really locations)
        false_positives = [
            'Contact', 'Need', 'Emergency', 'Help', 'Food', 'Water', 'Medical', 
            'Dr', 'Doctor', 'Mr', 'Mrs', 'Ms', 'Phone', 'Number', 
            'Muhammad', 'Ali', 'Ahmed', 'Khan', 'Sarah', 'John', 'Maria', 'Hassan',
            'Contacted', 'The', 'And', 'The', 'Coordinating', 'Requesting', 'With',
            'Providing', 'Managing', 'Heading', 'Hospital', 'Organization', 'Agency'
        ]
        
        # More aggressive false positive filtering based on context
        cleaned_locations = []
        seen = set()
        
        for loc in locations:
            if not loc or len(loc.strip()) < 2:
                continue
            
            loc = loc.strip()
            
            # Skip if it's in false positives list
            if loc in false_positives:
                continue            # Skip multi-word phrases that look like person names or actions
            words = loc.split()
            if len(words) > 2 and any(word.lower() in ['contacted', 'requesting', 'coordinating', 'distributing'] for word in words):
                continue
            
            # Normalize and add
            loc_lower = loc.lower()
            if loc_lower not in seen:
                seen.add(loc_lower)
                cleaned_locations.append(loc)
        
        # Sort for consistency
        cleaned_locations.sort()
        
        return cleaned_locations
    
    def extract_persons(self, text: str) -> List[str]:
        """
        Extract person names using regex patterns
        
        Args:
            text: Input text
            
        Returns:
            list: List of person names
        """
        persons = []
        
        # Pattern 1: Dr./Mr./Mrs./Ms. followed by full name (FirstName LastName or FirstName LastName Suffix)
        pattern1 = r'\b(?:Dr\.?|Mr\.?|Mrs\.?|Ms\.?|Prof\.?)\s+([A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)'
        matches = re.finditer(pattern1, text, re.IGNORECASE)
        for match in matches:
            if match.lastindex and match.lastindex >= 1:
                name = match.group(1).strip()
                # Remove trailing words that shouldn't be part of name
                name = re.sub(r'\s+(contacted|at|is|are|from|to|for|in|on|with)\b.*$', '', name, flags=re.IGNORECASE)
                name = name.strip()
                # Check that this doesn't look like an organization
                if name and len(name) > 3 and not any(word in name for word in ['Hospital', 'Center', 'Office', 'Foundation', 'University', 'Department']):
                    persons.append(name)
        
        # Pattern 2: Name pattern in context like "coordinator [Name]", "led by [Name]", etc.
        contextual_patterns = [
            r'coordinator\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)(?=\s+(?:is|from|at|to|for|in|with)|\s*[,.]|$)',
            r'led\s+by\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)(?=\s+(?:is|from|at|to|for|in|with)|\s*[,.]|$)',
            r'from\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?(?:\s+[A-Z][a-z]+)?)\s+(?:is|from)',
        ]
        
        for pattern in contextual_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                if match.lastindex and match.lastindex >= 1:
                    name = match.group(1).strip()
                    # Check for organization keywords in the name
                    org_words = ['Hospital', 'Foundation', 'Council', 'Committee', 'Agency', 'Center', 
                                'Team', 'Department', 'Office', 'Group', 'Unit', 'Service', 'Board']
                    if name and len(name) > 2 and not any(word in name for word in org_words):
                        persons.append(name)
        
        # Deduplicate while preserving order
        seen = set()
        cleaned_persons = []
        for person in persons:
            person_lower = person.lower()
            if person_lower not in seen:
                seen.add(person_lower)
                cleaned_persons.append(person)
        
        return cleaned_persons
    
    def extract_organizations(self, text: str) -> List[str]:
        """
        Extract organization names using regex patterns
        
        Args:
            text: Input text
            
        Returns:
            list: List of organization names
        """
        orgs = []
        
        # Common organizations/NGOs - with full names for better matching
        known_orgs = {
            'Pakistan Red Crescent': ['pakistan red crescent'],
            'Red Crescent': ['red crescent'],
            'Red Cross': ['red cross', 'international red cross'],
            'United Nations Humanitarian Office': ['united nations humanitarian office'],
            'United Nations': ['united nations'],
            'NDMA': ['ndma', 'national disaster management authority'],
            'WHO': ['who', 'world health organization'],
            'UNICEF': ['unicef', 'united nations children'],
            'UNHCR': ['unhcr', 'united nations refugee'],
            'WFP': ['wfp', 'world food programme'],
            'ICRC': ['icrc', 'international committee red cross'],
            'MSF': ['msf', 'medecins sans frontieres', 'doctors without borders'],
            'Save the Children': ['save the children'],
            'Oxfam': ['oxfam'],
            'Care International': ['care international', 'care'],
            'Edhi Foundation': ['edhi foundation', 'edhi'],
            'Aga Khan': ['aga khan'],
            'Shaukat Khanum': ['shaukat khanum', 'shaukat khanum hospital'],
            'HBL': ['hbl', 'habib bank'],
            'Dawn': ['dawn', 'dawn news'],
            'Geo TV': ['geo tv', 'geo'],
            'ARY': ['ary', 'ary news'],
            'PEMRA': ['pemra'],
            'FIA': ['fia'],
            'Rangers': ['rangers'],
            'Police': ['police', 'police department'],
            'Army': ['army', 'military'],
            'Navy': ['navy'],
            'Air Force': ['air force'],
        }
        
        text_lower = text.lower()
        # Track found organizations to avoid duplicates (e.g., "Red Crescent" as subset of "Pakistan Red Crescent")
        found_org_keys = set()
        
        # Process in order of priority - longer/more specific names first to avoid duplicates
        sorted_orgs = sorted(known_orgs.items(), key=lambda x: -len(x[0]))
        
        for org_name, patterns in sorted_orgs:
            for pattern in patterns:
                if pattern in text_lower:
                    # Check if any existing found org conflicts with this one
                    is_subset = False
                    for existing_key in found_org_keys:
                        existing_name = known_orgs[existing_key][0]  # Get the display name
                        # Don't add if current org is a substring of existing org
                        if org_name.lower() in existing_name.lower():
                            is_subset = True
                            break
                    
                    if not is_subset:
                        orgs.append(org_name)
                        found_org_keys.add(org_name)
                    break
        
        # Pattern: Organization pattern (Acronyms or Capitalized phrases with "organization", "foundation", "agency", etc.)
        pattern1 = r'\b([A-Z]{2,}(?:\s+[A-Z]{2,})*)\s+(?:organization|foundation|agency|committee|council|bureau|office|institute|association|group|team|department)'
        matches = re.finditer(pattern1, text, re.IGNORECASE)
        for match in matches:
            if match.lastindex and match.lastindex >= 1:
                org = match.group(1).strip()
                # Skip if it's articles or starts with articles (The, And, A, An)
                if org and len(org) > 1 and org not in ['The', 'And', 'A', 'An'] and not org.startswith('The '):
                    # Check if this is a substring of already found org
                    is_substring = False
                    for existing_org in orgs:
                        existing_lower = existing_org.lower()
                        org_lower = org.lower()
                        # Skip if current org is contained in existing org
                        if org_lower in existing_lower and len(existing_lower) > len(org_lower) + 2:
                            is_substring = True
                            break
                    if not is_substring:
                        orgs.append(org)
        
        # Pattern: "NGO" or "NGOs" with names
        pattern2 = r'\b(?:local\s+)?NGOs?\s+like\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)'
        matches = re.finditer(pattern2, text, re.IGNORECASE)
        for match in matches:
            if match.lastindex and match.lastindex >= 1:
                org = match.group(1).strip()
                if org and len(org) > 2:
                    orgs.append(org)
        
        # Pattern: Hospital/Medical centers
        pattern3 = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:Hospital|Medical Center|Clinic|Health Center|Trauma Center|Emergency Department)'
        matches = re.finditer(pattern3, text, re.IGNORECASE)
        for match in matches:
            if match.lastindex and match.lastindex >= 1:
                org = match.group(1).strip()
                if org and len(org) > 2:
                    orgs.append(org)
        
        # Deduplicate and filter out partial matches
        seen = set()
        cleaned_orgs = []
        for org in orgs:
            org_lower = org.lower()
            if org_lower not in seen:
                # Check if this org is contained in any existing org (partial match)
                is_partial = False
                for existing in cleaned_orgs:
                    existing_lower = existing.lower()
                    # If current org is a prefix of existing org, skip it
                    if existing_lower.startswith(org_lower.rstrip()) and len(existing_lower) > len(org_lower) + 2:
                        is_partial = True
                        break
                
                if not is_partial:
                    seen.add(org_lower)
                    cleaned_orgs.append(org)
        
        return cleaned_orgs
    
    def extract_all(self, text: str) -> Dict:
        """
        Extract all entity types
        
        Args:
            text: Input text
            
        Returns:
            dict: Dictionary with all extracted entities
        """
        entities = self.extract_entities(text)
        
        result = {
            'locations': [],
            'persons': [],
            'organizations': [],
            'phone_numbers': [],
            'resources': [],
            'all_entities': entities
        }
        
        # Categorize entities from transformer model
        for entity in entities:
            entity_group = entity.get('entity_group', '').upper()
            word = entity.get('word', '')
            
            if entity_group in ['LOC', 'LOCATION', 'GPE', 'MISC']:
                result['locations'].append(word)
            elif entity_group in ['PER', 'PERSON']:
                result['persons'].append(word)
            elif entity_group in ['ORG', 'ORGANIZATION']:
                result['organizations'].append(word)
        
        # Use improved location extraction
        regex_locations = self.extract_locations(text)
        # Merge with NER locations and deduplicate
        all_locations = list(set(result['locations'] + regex_locations))
        result['locations'] = all_locations
        
        # Use regex-based person extraction (since transformer model doesn't detect them well)
        result['persons'] = self.extract_persons(text)
        
        # Use regex-based organization extraction (since transformer model doesn't detect them well)
        result['organizations'] = self.extract_organizations(text)
        
        # Extract phone numbers
        result['phone_numbers'] = self.extract_phone_numbers(text)
        
        # Extract resources with improved pattern for quantities
        resources = []
        
        # Pattern 1: [Number] [Resource] e.g., "500 tents", "1000 blankets", "medical supplies"
        quantity_pattern = r'(\d+)\s+(tents?|blankets?|supplies?|kits?|medicines?|medical\s+supplies?|food\s+items?|water\s+containers?|medicines?|antibiotics?|vaccines?)'
        matched_quantities = set()
        
        for match in re.finditer(quantity_pattern, text, re.IGNORECASE):
            quantity = match.group(1)
            resource = match.group(2)
            resources.append(f"{quantity} {resource}")
            matched_quantities.add(resource.lower().strip())
        
        # Pattern 2: Compound resources (medical supplies, food items, etc.)
        compound_resources = ['medical supplies', 'food items', 'water containers']
        text_lower = text.lower()
        for compound in compound_resources:
            if compound in text_lower and compound.split()[0] not in matched_quantities:
                # Only add if it's not already captured as a quantity match
                if not any(compound.lower() in r.lower() for r in resources):
                    resources.append(compound)
        
        # Pattern 3: Single keywords (if not already captured)
        resource_keywords = ['food', 'water', 'shelter', 'medical', 'medicine', 'aid', 'help']
        for kw in resource_keywords:
            if kw in text_lower and not any(kw.lower() in r.lower() for r in resources):
                resources.append(kw)
        
        result['resources'] = resources
        
        return result

def process_dataset_ner(df: pd.DataFrame, text_column: str, 
                       ner_model: MultilingualNER) -> pd.DataFrame:
    """
    Process entire dataset with NER
    
    Args:
        df: Input dataframe
        text_column: Name of text column
        ner_model: NER model instance
        
    Returns:
        pd.DataFrame: Dataframe with extracted entities
    """
    results = []
    
    for idx, row in df.iterrows():
        text = row[text_column]
        entities = ner_model.extract_all(text)
        
        results.append({
            'text': text,
            'locations': ', '.join(entities['locations']),
            'phone_numbers': ', '.join(entities['phone_numbers']),
            'persons': ', '.join(entities['persons']),
            'organizations': ', '.join(entities['organizations']),
            'resources': ', '.join(entities['resources']),
            'all_entities': str(entities['all_entities'])
        })
        
        if (idx + 1) % 100 == 0:
            print(f"Processed {idx + 1}/{len(df)} texts...")
    
    return pd.DataFrame(results)

if __name__ == "__main__":
    # Example usage
    ner = MultilingualNER()
    
    sample_text = "Emergency in Lahore. Contact 03001234567. Need food and water. Dr. Ahmed is helping."
    entities = ner.extract_all(sample_text)
    
    print("Extracted Entities:")
    for key, value in entities.items():
        if value:
            print(f"  {key}: {value}")


