#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
Created on 2025/04/15 20:50:59
@author: SIRConceicao

This script processes JSON files containing annotations for the GutBrainIE dataset.
It tags entities in sentences with the double tag format <eN>@entityType$ EntitySpanText @/entityType$</eN>.
'''

import json
import pandas as pd
import warnings
from pandas.errors import SettingWithCopyWarning
warnings.simplefilter(action='ignore', category=SettingWithCopyWarning)
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
def kret_tags_testset(sentence, df):
    """Inserts entity tags while adjusting offsets dynamically."""
        
    shift = 0 
    tagged_sentence = sentence  
    entity_number = []
    
    for i, row in df.iterrows():
      
        start = row.start_idx + shift
        end = row.end_idx + shift + 1

        tag_num = i + 1  # <eN> 

        entity_text = tagged_sentence[start:end]

        # Adjusted format: Properly formatted without extra spaces
        tagged_entity = f"<e{tag_num}>@{row.label}$ {entity_text} @/{row.label}$</e{tag_num}> "

        # Insert the tagged entity into the sentence
        tagged_sentence = tagged_sentence[:start] + tagged_entity + tagged_sentence[end:]

        # Update shift for all future offsets
        shift += len(tagged_entity) - len(entity_text)
        entity_number.append(f"e{tag_num}")  
    
    # Fix extra spaces around the tag structure
    tagged_sentence = tagged_sentence.replace(" @/", "@/").replace("</e ", "</e>")

    df['entity_num'] = entity_number

    return tagged_sentence, df

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

    hypergraphs=[]
    #print("Hypergraph Set Notation:\n")
    for (predicate, obj_text), subjects in hyperedges.items():
        subjects_text = ', '.join(subjects)
        #print(f"{{{subjects_text}}} ⟶[{predicate}] {obj_text}")
        hypergraphs.append(f"[{subjects_text}] ⟶[{predicate}] ⟶[{obj_text}]")
    
    return hypergraphs
#===================================================================================

def entity_number_match(relations_df, entitynum_title_df, entitynum_abs_df):
    """Matches the entity number tag to the relations, considering title vs abstract"""

    # match keys 
    entitynum_title_df['match_key'] = list(zip(entitynum_title_df['text_span'], entitynum_title_df['start_idx'], entitynum_title_df['end_idx'], entitynum_title_df['location']))
    entitynum_abs_df['match_key'] = list(zip(entitynum_abs_df['text_span'], entitynum_abs_df['start_idx'], entitynum_abs_df['end_idx'], entitynum_abs_df['location']))

    # Create lookups
    entity_lookup_title = dict(zip(entitynum_title_df['match_key'], entitynum_title_df['entity_num']))
    entity_lookup_abs = dict(zip(entitynum_abs_df['match_key'], entitynum_abs_df['entity_num']))

    # Create subject and object match keys, now including location
    relations_df['subject_match_key'] = list(zip(relations_df['subject_text_span'], relations_df['subject_start_idx'], relations_df['subject_end_idx'], relations_df['subject_location']))
    relations_df['object_match_key'] = list(zip(relations_df['object_text_span'], relations_df['object_start_idx'], relations_df['object_end_idx'], relations_df['object_location']))

    # helper to choose the right lookup
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

    #  match
    relations_df['subject_entity_num'] = relations_df.apply(match_subject, axis=1)
    relations_df['object_entity_num'] = relations_df.apply(match_object, axis=1)

    relations_df = relations_df.drop(columns=['subject_match_key', 'object_match_key'])

    return relations_df

#===================================================================================

def build_tagged_hypergraph(df):
    # Set notation for Hypergraph representation
    # {<e5> Ent_A </e5>, <e6> Ent_B </e6>} ⟶[predicate] <e7> Ent_F </e7>
    #returns [[subject]⟶[predicate]⟶[object]]

    hyperedges = {}

    # Group subjects that share (predicate, object_entity_num, object_text_span)
    for idx, row in df.iterrows():
        key = (row['predicate'], row['object_entity_num'], row['object_text_span'])
        subj = f"<{row['subject_entity_num']}> {row['subject_text_span']} </{row['subject_entity_num']}>"

        if key not in hyperedges:
            hyperedges[key] = set()
        hyperedges[key].add(subj)


    hypergraphs=[]
    #print("Tagged Hypergraph Set Notation:\n")
    for (predicate, obj_entity_num, obj_text), subjects in hyperedges.items():
        subjects_text = ', '.join(subjects)
        object_text = f"<{obj_entity_num}> {obj_text} </{obj_entity_num}>"
        #print(f"{{{subjects_text}}} ⟶[{predicate}] {object_text}")
        hypergraphs.append(f"[{subjects_text}] ⟶[{predicate}] ⟶[{object_text}]")

    return hypergraphs

#===================================================================================
def tag_dev_json(doc_path):

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

        tag_relations = entity_number_match(relations_df,title_ent_tagnum,abs_ent_tagnum)

        relations_tagged_hypergraph = build_tagged_hypergraph(tag_relations)
             
        #relation_hypergraph= build_hypergraph(relations_df)
        #-------------------------------------------------------------------------------
        #binary_tag=jbase['binary_tag_based_relations']
        #binary_tag_df=pd.DataFrame(binary_tag)
        #-------------------------------------------------------------------------------
        # ternary_tag=jbase['ternary_tag_based_relations']
        # ternary_tag_df=pd.DataFrame(ternary_tag)
        #-------------------------------------------------------------------------------
        ternary_mention=jbase['ternary_mention_based_relations']
        ternary_mention_df=pd.DataFrame(ternary_mention)
        ternary_mention_hypergraph= build_hypergraph(ternary_mention_df)
        #-------------------------------------------------------------------------------

        metadata['title_tagged']= tagged_title
        metadata['abstract_tagged'] = tagged_abstract
        jbase["relations_tagged_hypergraph"]= relations_tagged_hypergraph
        jbase['ternary_mention_tagged_relations'] = ternary_mention_hypergraph

        jbase.pop('entities',None)
        jbase.pop('ternary_mention_based_relations',None)
        metadata.pop('title',None)
        metadata.pop('abstract',None)
      
  
    return annotations_data

#===================================================================================

def tag_train_json(doc_path):

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

        if not relations_df.empty:

            tag_relations = entity_number_match(relations_df,title_ent_tagnum,abs_ent_tagnum)

            relations_tagged_hypergraph = build_tagged_hypergraph(tag_relations)
        else: 
             relations_tagged_hypergraph =[]
        
        #relation_hypergraph= build_hypergraph(relations_df)
        #-------------------------------------------------------------------------------
        #binary_tag=jbase['binary_tag_based_relations']
        #binary_tag_df=pd.DataFrame(binary_tag)
        #-------------------------------------------------------------------------------
        # ternary_tag=jbase['ternary_tag_based_relations']
        # ternary_tag_df=pd.DataFrame(ternary_tag)
        #-------------------------------------------------------------------------------
        # ternary_mention=jbase['ternary_mention_based_relations']
        # ternary_mention_df=pd.DataFrame(ternary_mention)
        # ternary_mention_hypergraph= build_hypergraph(ternary_mention_df)
        #-------------------------------------------------------------------------------

        metadata['title_tagged']= tagged_title
        metadata['abstract_tagged'] = tagged_abstract
        jbase["relations_tagged_hypergraph"]= relations_tagged_hypergraph
        # jbase['ternary_mention_tagged_relations'] = ternary_mention_hypergraph

        jbase.pop('entities',None)
        jbase.pop('ternary_mention_based_relations',None)
        metadata.pop('title',None)
        metadata.pop('abstract',None)
      
  
    return annotations_data

#===================================================================================
def tag_test_json(doc_path):

    annotations_data= load_data(doc_path)

    for annot in annotations_data:
        jbase= annotations_data[annot]
        #metadata = jbase['metadata']
        title = jbase['title']
        abstract = jbase['abstract']
        #-------------------------------------------------------------------------------
        entities=jbase['entities']
        ent_df = pd.DataFrame(entities)

        ent_titles_df = ent_df[ent_df['location']=='title']
        ent_abstract_df = ent_df[ent_df['location']=='abstract']

        tagged_title,title_ent_tagnum = kret_tags_testset(title, ent_titles_df)
        tagged_abstract,abs_ent_tagnum = kret_tags_testset(abstract, ent_abstract_df)
        #-------------------------------------------------------------------------------

        jbase['title_tagged']= tagged_title
        jbase['abstract_tagged'] = tagged_abstract      
  
    return annotations_data

#===================================================================================

def main():

    #doc_path= "data/GutBrainIE_Full_Collection_2025/Annotations/Dev/json_format/dev.json"

    #annotations_data = tag_dev_json(doc_path)


    # doc_path= "data/GutBrainIE_Full_Collection_2025/Annotations/Train/bronze_quality/json_format/train_bronze.json"
    # doc_path= "data/GutBrainIE_Full_Collection_2025/Annotations/Train/silver_quality/json_format/train_silver.json"
    # doc_path= "data/GutBrainIE_Full_Collection_2025/Annotations/Train/gold_quality/json_format/train_gold.json"
    # doc_path= "data/GutBrainIE_Full_Collection_2025/Annotations/Train/platinum_quality/json_format/train_platinum.json"

    # annotations_data = tag_train_json(doc_path)

    #---------------------------------------------------------------------------------
    doc_path="data/processed/lasigeBioTM_subtask6_1_NER_Mistral-7B-Instruct-v0.3_fixed.json"
    out_path="data/GutBrainIE_tagged/Annotations/Test/"
    annotations_data = tag_test_json(doc_path)
    #---------------------------------------------------------------------------------
    
    #out_path = "data/GutBrainIE_tagged/Annotations/Dev/dev_tagged.json"
    #out_path = "data/GutBrainIE_tagged/Annotations/Train/"
    name = doc_path.split("/")[-1].replace(".json","")
    
    with open(f"{out_path}{name}_tagged.json", "w") as f: 
        json.dump(annotations_data, f, indent=2)

if __name__ == "__main__":
    main()