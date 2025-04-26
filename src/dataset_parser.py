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
# def build_hypergraph_old(df):
#     hypergraph = {
#     'nodes': set(),     # all unique nodes
#     'hyperedges': []    # list of edges: each edge connects subject, object (could be more in future)
#     }

#     # Fill nodes and hyperedges
#     for idx, row in df.iterrows():
#         subj_node = (row['subject_text_span'], row['subject_label'])
#         obj_node = (row['object_text_span'], row['object_label'])

#         # Add nodes
#         hypergraph['nodes'].add(subj_node)
#         hypergraph['nodes'].add(obj_node)

#         # Add hyperedge (connection between subject and object)
#         hypergraph['hyperedges'].append([subj_node, obj_node])

       
#         # Now hypergraph has nodes and hyperedges!
#         # Display nicely
#         print("Nodes:")
#         for node in hypergraph['nodes']:
#             print(f"  {node}")

#         print("\nHyperedges:")
#         for edge in hypergraph['hyperedges']:
#             print(f"  {edge}")

#     # Convert set back to list for better usability
#     hypergraph['nodes'] = list(hypergraph['nodes'])
#     return hypergraph

#===================================================================================
def build_hypergraph(df):
    #Set notation for Hypergraph representation
    #{Ent_A,Ent_B,Ent_C} ⟶ Ent_F

    hyperedges = {}

    # Group subjects that share (predicate, object_text_span)
    for idx, row in df.iterrows():
        key = (row['predicate'], row['object_text_span'])  
        subj = row['subject_text_span'] 

        if key not in hyperedges:
            hyperedges[key] = set()
        hyperedges[key].add(subj)

   
    print("Hypergraph Set Notation:\n")
    for (predicate, obj_text), subjects in hyperedges.items():
        subjects_text = ', '.join(subjects)
        print(f"{{{subjects_text}}} ⟶[{predicate}] {obj_text}")
    
    return hyperedges
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
    relations_df=pd.DataFrame(relations)

    relation_hypergraph= build_hypergraph(relations_df)
    #-------------------------------------------------------------------------------
    binary_tag=jbase['binary_tag_based_relations']
    #binary_tag_df=pd.DataFrame(binary_tag)
    #-------------------------------------------------------------------------------
    ternary_tag=jbase['ternary_tag_based_relations']
    ternary_tag_df=pd.DataFrame(ternary_tag)
    #-------------------------------------------------------------------------------
    ternary_mention=jbase['ternary_mention_based_relations']
    ternary_mention_df=pd.DataFrame(ternary_mention)
    ternary_mention_hypergraph= build_hypergraph(ternary_mention_df)
    #-------------------------------------------------------------------------------
    