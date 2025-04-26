#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
Created on 2025/04/15 20:50:59
@author: SIRConceicao
'''
#TODO formatar dataset para correr na LLM

import json
import pandas as pd
#===================================================================================

def load_data(file_path: str):
    with open(file_path, "r") as f:
        data = json.load(f)
    return data
#===================================================================================

def kret_tags(sentence, df):
    """Inserts entity tags while adjusting offsets dynamically."""
        
    shift = 0 
    tagged_sentence = sentence  

    for i, row in df.iterrows():
      
        start =row.start_idx + shift
        end=row.end_idx + shift +1

        tag_num = row.label  # <eN> 

        entity_text = tagged_sentence[start:end]

        tagged_entity = f"<{tag_num}>{entity_text}</{tag_num}>"
        

        tagged_sentence = tagged_sentence[:start] + tagged_entity + tagged_sentence[end:]

        # Update shift for all future offsets
        shift += len(tagged_entity) - len(entity_text)  

    return tagged_sentence
#===================================================================================


#===================================================================================


doc_path= "data/GutBrainIE_Full_Collection_2025/Annotations/Dev/json_format/dev.json"
annotations_data= load_data(doc_path)

for annot in annotations_data:
    jbase= annotations_data[annot]
    metadata = jbase['metadata']
    title = metadata['title']
    abstract = metadata['abstract']
    #-------------------------------------------------------------------------------
    entities=jbase['entities']
    ent_df = pd.DataFrame(entities)

    ent_titles_df = ent_df[ent_df['location']=='title']
    ent_abstract_df = ent_df[ent_df['location']=='abstract']

    tagged_title = kret_tags(title, ent_titles_df)
    tagged_abstract = kret_tags(abstract, ent_abstract_df)
    #-------------------------------------------------------------------------------
    relations=jbase['relations']
    binary_tag=jbase['binary_tag_based_relations']
    ternary_tag=jbase['ternary_tag_based_relations']
    ternary_mention=jbase['ternary_mention_based_relations']
    