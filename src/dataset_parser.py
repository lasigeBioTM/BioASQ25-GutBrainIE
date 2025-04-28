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
    entity_number=[]
    for i, row in df.iterrows():
      
        start =row.start_idx + shift
        end=row.end_idx + shift +1

        tag_num = i + 1  # <eN> 

        entity_text = tagged_sentence[start:end]

        tagged_entity = f"<e{tag_num}>@{row.label}$ {entity_text} @/{row.label}$</e{tag_num}>"
        

        tagged_sentence = tagged_sentence[:start] + tagged_entity + tagged_sentence[end:]

        # Update shift for all future offsets
        shift += len(tagged_entity) - len(entity_text)
        entity_number.append(f"e{tag_num}")  
    
    df['entity_num']=entity_number

    return tagged_sentence,df 
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
# def entity_number_match(relations_df,entitynum_df):
#     """Matches the entity number tag to the relations"""

#     entitynum_df['match_key'] = list(zip(entitynum_df['text_span'], entitynum_df['start_idx'], entitynum_df['end_idx']))


#     relations_df['subject_match_key'] = list(zip(relations_df['subject_text_span'], relations_df['subject_start_idx'], relations_df['subject_end_idx']))
#     relations_df['object_match_key']  = list(zip(relations_df['object_text_span'], relations_df['object_start_idx'], relations_df['object_end_idx']))


#     entity_lookup = dict(zip(entitynum_df['match_key'], entitynum_df['entity_num']))

#     #map
#     relations_df['subject_entity_num'] = relations_df['subject_match_key'].map(entity_lookup)
#     relations_df['object_entity_num']  = relations_df['object_match_key'].map(entity_lookup)

#     relations_df = relations_df.drop(columns=['subject_match_key', 'object_match_key'])

#     return relations_df



def entity_number_match(relations_df, entitynum_title_df, entitynum_abs_df):
    """Matches the entity number tag to the relations, considering title vs abstract"""

    # Add match keys including location
    entitynum_title_df['match_key'] = list(zip(entitynum_title_df['text_span'], entitynum_title_df['start_idx'], entitynum_title_df['end_idx'], entitynum_title_df['location']))
    entitynum_abs_df['match_key'] = list(zip(entitynum_abs_df['text_span'], entitynum_abs_df['start_idx'], entitynum_abs_df['end_idx'], entitynum_abs_df['location']))

    # Create lookups
    entity_lookup_title = dict(zip(entitynum_title_df['match_key'], entitynum_title_df['entity_num']))
    entity_lookup_abs = dict(zip(entitynum_abs_df['match_key'], entitynum_abs_df['entity_num']))

    # Create subject and object match keys, now including location
    relations_df['subject_match_key'] = list(zip(relations_df['subject_text_span'], relations_df['subject_start_idx'], relations_df['subject_end_idx'], relations_df['subject_location']))
    relations_df['object_match_key'] = list(zip(relations_df['object_text_span'], relations_df['object_start_idx'], relations_df['object_end_idx'], relations_df['object_location']))

    # Define a helper to choose the right lookup
    def match_subject(row):
        if row['subject_location'] == 'title':
            return entity_lookup_title.get(row['subject_match_key'])
        else:  # assume abstract
            return entity_lookup_abs.get(row['subject_match_key'])

    def match_object(row):
        if row['object_location'] == 'title':
            return entity_lookup_title.get(row['object_match_key'])
        else:
            return entity_lookup_abs.get(row['object_match_key'])

    # Apply the matching
    relations_df['subject_entity_num'] = relations_df.apply(match_subject, axis=1)
    relations_df['object_entity_num'] = relations_df.apply(match_object, axis=1)

    # Clean up
    relations_df = relations_df.drop(columns=['subject_match_key', 'object_match_key'])

    return relations_df

#===================================================================================

def build_tagged_hypergraph(df):
    # Set notation for Hypergraph representation
    # {<e5> Ent_A </e5>, <e6> Ent_B </e6>} ⟶[predicate] <e7> Ent_F </e7>

    hyperedges = {}

    # Group subjects that share (predicate, object_entity_num, object_text_span)
    for idx, row in df.iterrows():
        key = (row['predicate'], row['object_entity_num'], row['object_text_span'])
        subj = f"<{row['subject_entity_num']}> {row['subject_text_span']} </{row['subject_entity_num']}>"

        if key not in hyperedges:
            hyperedges[key] = set()
        hyperedges[key].add(subj)

    print("Tagged Hypergraph Set Notation:\n")
    for (predicate, obj_entity_num, obj_text), subjects in hyperedges.items():
        subjects_text = ', '.join(subjects)
        object_text = f"<{obj_entity_num}> {obj_text} </{obj_entity_num}>"
        print(f"{{{subjects_text}}} ⟶[{predicate}] {object_text}")

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

    tagged_title,title_ent_tagnum = kret_tags(title, ent_titles_df)
    tagged_abstract,abs_ent_tagnum = kret_tags(abstract, ent_abstract_df)
    #-------------------------------------------------------------------------------
    relations=jbase['relations']
    relations_df=pd.DataFrame(relations)

    filtered_df = relations_df[relations_df['subject_location'] != relations_df['object_location']]
    if not filtered_df.empty:
        a=0
    
    # relations_title =  relations_df[relations_df['subject_location']=='title']
    # relations_abstract =  relations_df[relations_df['subject_location']=='abstract']
    

    #tag_abs_relations= entity_number_match(relations_abstract,abs_ent_tagnum)
    tag_abs_relations = entity_number_match(relations_df,title_ent_tagnum,abs_ent_tagnum)

    tagged_hypergraph = build_tagged_hypergraph(tag_abs_relations)
    
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
    