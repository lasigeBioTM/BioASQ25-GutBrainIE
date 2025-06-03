#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
Created on 2025/05/06 10:17:04
@author: SIRConceicao

Script to merge LLM outputs into Subtask 6.1 (NER) Submission Format
'''
import re
import json
import pandas as pd
from collections import defaultdict

#==================================================================================================
def load_data(doc_path):
    with open(doc_path, 'r') as file:
        data = json.load(file)
    return data
#==================================================================================================
# STATISTICS
#==================================================================================================
def analyze_entity_statistics(df, doc_id):
    """
    Generate statistics for a single document's entities.
    
    Args:
        df (pd.DataFrame): Dataframe with entity information
        doc_id (str): The document identifier
        
    Returns:
        dict: Statistics for this document
    """
    stats = {}
    
    # Basic counts
    stats['total_entities'] = int(len(df))
    stats['valid_entities'] = int(df['is_valid'].sum())
    stats['invalid_entities'] = int(len(df) - df['is_valid'].sum())
    stats['adjusted_entities'] = int((df['offset'] != 0).sum())
    
    # Label statistics
    stats['label_counts'] = df['label'].value_counts().to_dict()
    
    # Case-insensitive text span statistics
    df['text_span_lower'] = df['text_span'].str.lower()
    stats['text_span_counts'] = df['text_span_lower'].value_counts().to_dict()
    
    # Adjusted entities breakdown
    adjusted_df = df[df['offset'] != 0]
    stats['adjustments'] = {
        'by_label': adjusted_df['label'].value_counts().to_dict(),
        'by_offset': adjusted_df['offset'].value_counts().to_dict()
    }
    
    return stats

def save_entity_statistics(all_stats, output_file="entity_statistics.json"):
    """
    Save accumulated statistics to a JSON file.
    
    Args:
        all_stats (dict): Dictionary containing all documents' statistics
        output_file (str): Path to save the JSON file
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_stats, f, indent=4, ensure_ascii=False)


#==================================================================================================
def fix_truncated_json(json_str):
    """
    Fix truncated JSON by finding the last complete entity and properly closing the structure.
    """
    # First try to load normally
    try:
        json.loads(json_str)
        return json_str
    except json.JSONDecodeError:
        pass

    # Pattern to find complete entities
    entity_pattern = r'\{[^{}]*\}'
    
    # Find all complete entities
    matches = list(re.finditer(entity_pattern, json_str, re.DOTALL))
    
    if not matches:
        return '{"entities": []}'
    
    # Find the last valid entity end position
    last_good_end = matches[-1].end()
    
    # Rebuild the JSON string
    fixed = json_str[:last_good_end]
    
    # Properly close the JSON structure
    fixed = fixed.rstrip().rstrip(',')  # Remove trailing comma if exists
    fixed += '\n        ]\n    }'  # Close array and root object
    
    # Add opening if missing
    if not fixed.startswith('{'):
        fixed = '{"entities": [' + fixed + '}'
    
    # Final validation
    try:
        json.loads(fixed)
        return fixed
    except json.JSONDecodeError as e:
        # Fallback: return valid JSON with whatever entities we could recover
        recovered_entities = []
        for match in matches:
            try:
                recovered_entities.append(json.loads(match.group(0)))
            except json.JSONDecodeError:
                continue
                
        return json.dumps({"entities": recovered_entities}, indent=4)
#==================================================================================================

def validate_and_adjust_text_spans(text, df, max_offset=2):
    """
    Validate text spans and attempt to adjust indices when there's a small offset.
    
    Args:
        text (str): The full text
        df (pd.DataFrame): Dataframe with start_idx, end_idx, text_span
        max_offset (int): Maximum characters to search in each direction
        
    Returns:
        pd.DataFrame: Original dataframe with added columns:
            - is_valid: whether the span matches (after possible adjustment)
            - adjusted_start: adjusted start index (if adjustment was made)
            - adjusted_end: adjusted end index (if adjustment was made)
            - offset: how many characters the adjustment was (+/-)
    """
    result_df = df.copy()
    result_df['is_valid'] = True
    result_df['adjusted_start'] = result_df['start_idx']
    result_df['adjusted_end'] = result_df['end_idx']
    result_df['offset'] = 0
    
    for idx, row in result_df.iterrows():
        start = row['start_idx']
        end = row['end_idx']
        expected_span = row['text_span']
        actual_span = text[start:end]
        
        if actual_span == expected_span:
            continue
            
        # Try to find the span in nearby positions
        found = False
        for offset in range(1, max_offset + 1):
            # Try shifting left
            new_start = start - offset
            new_end = end - offset
            if new_start >= 0:
                shifted_span = text[new_start:new_end]
                if shifted_span == expected_span:
                    result_df.at[idx, 'is_valid'] = True
                    result_df.at[idx, 'adjusted_start'] = new_start
                    result_df.at[idx, 'adjusted_end'] = new_end
                    result_df.at[idx, 'offset'] = -offset
                    found = True
                    break
                    
            # Try shifting right
            new_start = start + offset
            new_end = end + offset
            if new_end <= len(text):
                shifted_span = text[new_start:new_end]
                if shifted_span == expected_span:
                    result_df.at[idx, 'is_valid'] = True
                    result_df.at[idx, 'adjusted_start'] = new_start
                    result_df.at[idx, 'adjusted_end'] = new_end
                    result_df.at[idx, 'offset'] = offset
                    found = True
                    break
        
        if not found:
            result_df.at[idx, 'is_valid'] = False
            print(f"Validation failed for row {idx}:")
            print(f"Expected: '{expected_span}'")
            print(f"Actual:   '{actual_span}'")
            print(f"Indices: {start}-{end}\n")
    
    return result_df

#==================================================================================================
test_set_path = "data/GutBrainIE_Full_Collection_2025/Test_Data/Test_Data/articles_test.json"
output_llm_path = "data/intermediate/Mistral-7B-Instruct-v0.3-NER-it-outputs_V1.json"

test_data=load_data(test_set_path)
llm_data=load_data(output_llm_path)

all_statistics = {}
for test, llm in zip(test_data, llm_data):
    assert test==llm

    title=test_data[test]['title']
    abstract=test_data[test]['abstract']
    
    entities=llm_data[llm]['llm_output']
    
    if len(entities)>1:
        print("something fishy")
    

    try:
        df=pd.DataFrame(entities[0]['entities'])
    
    except KeyError:
        df=pd.DataFrame(entities)

    except TypeError: 
        fixed_json = fix_truncated_json(entities)
        fixed_data = json.loads(fixed_json) 
        entities = fixed_data['entities']  
        df=pd.DataFrame(entities)

    title_df = df[df['location']=='title']
    abstract_df = df[df['location']=='abstract']

    #Validade if entitie spans are correct in text and ajust
    val_title = validate_and_adjust_text_spans(title, title_df)
    val_abstract = validate_and_adjust_text_spans(abstract, abstract_df)
    
    
    #Join all entities from title and abstract
    entities_df = pd.concat([val_title,val_abstract],ignore_index=True)

    #-----------------------------------------------------------------------
    #Statistics
    doc_stats = analyze_entity_statistics(entities_df, test)
    all_statistics[test] = doc_stats
    #-----------------------------------------------------------------------

    invalid_entries = entities_df[entities_df['is_valid'] == False]
    if not invalid_entries.empty:
        a=0
    #---------------------------------------------------------------    
    #Clean entities
    #1. Remove invalid spans
    entities_df=entities_df[entities_df['is_valid'] == True]
    #2. Keep final columns
    trimmed_entities_df = entities_df[['location','text_span','label','adjusted_start','adjusted_end']]
    #Rename keys
    trimmed_entities_df.columns =['location', 'text_span', 'label', 'start_idx', 'end_idx'] 
    #---------------------------------------------------------------
    #Save entities
    test_data[test]['entities']=trimmed_entities_df.to_dict(orient="records")

#==================================================================================================

output_path="data/processed/lasigeBioTM_subtask6_1_NER_Mistral-7B-Instruct-v0.3.json"
with open(output_path, "w") as f: 
        json.dump(test_data, f, indent=2)


save_entity_statistics(all_statistics, "data/processed/lasigeBioTM_subtask6_1_NER_Mistral-7B-Instruct-v0.3_stats.json")