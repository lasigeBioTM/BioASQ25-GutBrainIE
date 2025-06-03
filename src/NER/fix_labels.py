"""Script made to see how many unique labels are there in the outputs of NER models"""

import json 
from json_parser import parse_json_sequence


file = "data/intermediate/Mistral-7B-Instruct-v0.3-NER-it-outputs-baseline.json"
with open(file, 'r', encoding='utf-8') as f:
    data = json.load(f)

ENTITIES_KEY = "llm_output"

### First overall pass to see the labels ###
pmids = list(data.keys())
TOTAL_ENTITIES = 0
all_initial_labels = set() # Renamed for clarity
for pmid in pmids:
    annot = data[pmid]
    ents = annot[ENTITIES_KEY]

    ## Parse the JSON sequence if its is not yet a list
    if isinstance(ents, str): 
        ents = parse_json_sequence(ents)
        print(ents)

    for ent in ents:
        TOTAL_ENTITIES += 1
        label = ent['label']
        all_initial_labels.add(label) # Corrected to use .add()

print("Unique labels in the output: ")
print(all_initial_labels) # Updated variable name
print("\n\n")



### Fix the labels ### 
# We go pmid by pmid and then entity by entity. We discard those that return None and keep legal labels. 
class LabelFixer: 

    LEGAL_ENTITY_LABELS = [
    "anatomical location",
    "animal",
    "bacteria",
    "biomedical technique",
    "chemical",
    "DDF",
    "dietary supplement",
    "drug",
    "food",
    "gene",
    "human",
    "microbiome",
    "statistical technique"
    ]

    def fix_label(self, label: str) -> str | None:
        """
        Fixes the labels of the entities, either by removing them or by changing them to the closest legal one. 
        """
        # First, lower the labels if not 'DDF'
        if label != 'DDF':
            label = label.lower()

        # Check if the label is in the legal entity labels
        if label in self.LEGAL_ENTITY_LABELS:
            return label

        if "anatomical" in label:
            return "anatomical location"
        
        if "disease" in label:
            return "DDF"
        elif "ddf" in label:
            return "DDF"

        # If the label is not legal, return None
        return None
    

### Correcting the labels ###
fixer = LabelFixer()

all_fixed_labels_collected = set() 
for pmid in pmids:
    annot = data[pmid]
    original_entities = annot[ENTITIES_KEY] # Get the original list of entities

    ## Parse the JSON sequence if its is not yet a list
    if isinstance(original_entities, str): 
        original_entities = parse_json_sequence(original_entities)

    updated_entities = []              
    for ent in original_entities:       
        label = ent['label']
        fixed_label = fixer.fix_label(label)
        if fixed_label is not None:         ### Fix the label or remove it 
            ent['label'] = fixed_label    
            updated_entities.append(ent)  
            all_fixed_labels_collected.add(fixed_label) 
    del annot[ENTITIES_KEY] ### Remove 'llm_output' key
    annot["entities"] = updated_entities ### Replace

ENTITIES_KEY = "entities" ### Add the new key with the fixed labels
print("Unique labels in the output after fixing (collected during processing):")
print(all_fixed_labels_collected) 
print("\n\n")



### Another Pass to See the labels ###
all_labels_from_data_after_fixing = set()
TOTAL_ENTITIES_FIXED = 0
for pmid in pmids:
    annot = data[pmid]
    ents = annot[ENTITIES_KEY]
    
    for ent in ents:
        TOTAL_ENTITIES_FIXED += 1
        label = ent['label']
        all_labels_from_data_after_fixing.add(label)
print("Unique labels in the output by re-iterating data after fixing:")
print(all_labels_from_data_after_fixing)
print("\n\n")

print(f"Total entities in the original data: {TOTAL_ENTITIES}")
print(f"Total entities in the fixed data: {TOTAL_ENTITIES_FIXED}")


output_file = "data/processed/Mistral-7B-Instruct-v0.3-NER-it-outputs-baseline_fixed.json"
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=4) 