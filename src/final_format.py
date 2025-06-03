
"""This script processes JSON data containing ternary relations into the challenge format."""
import re
import json
from json.decoder import JSONDecodeError
#==================================================================================================
def load_data(doc_path):
    """Load JSON data from the specified file path."""
    with open(doc_path, 'r') as file:
        data = json.load(file)
    return data

#==================================================================================================
def fix_null_in_json(json_string):
    """
    This function replaces occurrences of 'NULL' with 'null', handles invalid entries
    like 'NOT_AVAILABLE' by replacing them with 'no_string', and removes incomplete entries
    from the lists, ensuring the list is properly closed.
    """
    # Replace all 'NULL' with 'null'
    fixed_json_string = json_string.replace('NULL', 'null')
    
    # Replace 'NOT_AVAILABLE' with 'no_string'
    fixed_json_string = fixed_json_string.replace('NOT_AVAILABLE', '"no_string"')
    

    try:
        data = json.loads(fixed_json_string)
        
        # remove incomplete entries
        for key in data:
            if isinstance(data[key], list):
                new_list = []
                for item in data[key]:
                    if isinstance(item, dict):
                        # Check if essential fields are present and non-null
                        valid_entry = True
                        for essential_field in ['subject_label', 'predicate', 'object_label']:
                            if item.get(essential_field) in [None, "null", ""]:
                                valid_entry = False
                                break
                        
                        if valid_entry:
                            # Replace any remaining `null` or `NOT_AVAILABLE` with "no_string"
                            for sub_key in item:
                                if item[sub_key] is None or item[sub_key] == "null" or item[sub_key] == "NOT_AVAILABLE":
                                    item[sub_key] = "no_string"
                            new_list.append(item)
                
                data[key] = new_list
        
        return data
    
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        return None
#==================================================================================================
def recover_truncated_json(truncated_str):
    """
    Recovers truncated JSON by finding the last complete structure and properly closing all open brackets/braces.
    
    Args:
        truncated_str (str): The truncated JSON string
        
    Returns:
        dict: The recovered JSON data as a dictionary, or None if recovery fails
    """
   
    try:
        return json.loads(truncated_str)
    except json.JSONDecodeError:
        pass  
    
    # Initialize variables
    stack = []
    last_good_index = 0
    
    for i, char in enumerate(truncated_str):
        if char in '{[':
            stack.append(char)
        elif char in '}]':
            if stack:
                stack.pop()
            last_good_index = i
    

    recovered_str = truncated_str[:last_good_index + 1]
    while stack:
        open_char = stack.pop()
        close_char = '}' if open_char == '{' else ']'
        recovered_str += close_char
    

    try:
        return json.loads(recovered_str)
    except json.JSONDecodeError:
        # If still failing, try to remove the last incomplete element
        last_comma = recovered_str.rfind(',')
        if last_comma != -1:
            recovered_str = recovered_str[:last_comma]
            # Close all structures again
            stack = []
            for char in recovered_str:
                if char in '{[':
                    stack.append(char)
                elif char in '}]':
                    if stack:
                        stack.pop()
            while stack:
                open_char = stack.pop()
                close_char = '}' if open_char == '{' else ']'
                recovered_str += close_char
            try:
                return json.loads(recovered_str)
            except json.JSONDecodeError:
                pass
    #empty relations
    dummy_relations=[
    {
        "ternary_tag_based_relations": [
            {
                "subject_label": "none",
                "predicate": "none",
                "object_label": "none"
            }
        ]
    },
    {
        "ternary_mention_based_relations": [
            {
                "subject_text_span": "none",
                "subject_label": "none",
                "predicate": "none",
                "object_text_span": "none",
                "object_label": "none"
            }
        ]
    }
]
    return dummy_relations[0]


#==================================================================================================
def filter_labels(data):
    """ Filters the input data to keep only the desired keys for ternary relations."""
    # Define the keys we want to keep
    desired_keys = ["subject_text_span", "subject_label", "predicate", "object_text_span", "object_label"]
    
    # Process each dictionary and filter out the unnecessary keys
    filtered_data = []
    for entry in data:
        filtered_entry = {key: entry[key] for key in desired_keys if key in entry}
        filtered_data.append(filtered_entry)
    
    return filtered_data
#==================================================================================================
LEGAL_RELATION_LABELS = [
    "administered",
    "affect",
    "change abundance",
    "change effect",
    "change expression",
    "compared to",
    "impact",
    "influence",
    "interact",
    "is a",
    "is linked to",
    "located in",
    "part of",
    "produced by",
    "strike",
    "target",
    "used by"
]

# doc_path = "data/intermediate/Mistral-7B-Instruct-v0.3-baseline-labels-it-outputs.json"
# output_path = "data/processed/lasigeBioTM_Mistral-7B-Instruct-v0.3-baseline-labels-it-outputs_V2.json"


# doc_path ="data/intermediate/Mistral-7B-Instruct-v0.3-raw-baseline-it-outputs.json"
# output_path = "data/processed//Mistral-7B-Instruct-v0.3-raw-baseline-it-outputs.json"

# doc_path ="data/intermediate/Mistral-7B-Instruct-v0.3-spacy-semantics-it-outputs.json"
# output_path = "data/processed/Mistral-7B-Instruct-v0.3-spacy-semantics-it-outputs.json"

doc_path ="data/intermediate/Mistral-7B-Instruct-v0.3-constparsing-it-outputs.json"
output_path = "data/processed/Mistral-7B-Instruct-v0.3-constparsing-it-outputs.json"

data=load_data(doc_path)

for annot in data:
    jbase= data[annot]

    keys=jbase.keys() 
    
    if 'llm_output' in keys:
        output_raw=jbase['llm_output']
        #truncated
        output=recover_truncated_json(output_raw)

        try:
            answer =fix_null_in_json(output)
        except AttributeError:
            answer= output

        #NOTE Error incomplete JSON
        jbase['ternary_tag_based_relations']=answer['ternary_tag_based_relations']
        
        try: 
            jbase['ternary_mention_based_relations']=filter_labels(answer['ternary_mention_based_relations'])
        except KeyError:
            jbase['ternary_mention_based_relations']={
                "ternary_mention_based_relations": [
                    {
                        "subject_text_span": "none",
                        "subject_label": "none",
                        "predicate": "none",
                        "object_text_span": "none",
                        "object_label": "none"
                    }
                ]
            }
        jbase.pop('llm_output')


    if "ternary_tag_based_relations_predicted" in keys:
        jbase['ternary_tag_based_relations']=jbase['ternary_tag_based_relations_predicted']
        jbase['ternary_mention_based_relations']=filter_labels(jbase['ternary_mention_based_relations_predicted'])

        jbase.pop('ternary_tag_based_relations_predicted')
        jbase.pop('ternary_mention_based_relations_predicted')
    
    else:
         jbase['ternary_mention_based_relations']=filter_labels(jbase['ternary_mention_based_relations'])


    if "ternary_tag_based_relations_predicted" not in keys and "ternary_tag_based_relations" not in keys:
        jbase['ternary_tag_based_relations']=[]

    if 'ternary_mention_based_relations_predicted' not in keys and 'ternary_mention_based_relations' not in keys:
        jbase['ternary_mention_based_relations']=[]

with open(output_path, "w") as f: 
        json.dump(data, f, indent=2)