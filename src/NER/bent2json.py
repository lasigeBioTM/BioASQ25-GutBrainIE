#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
Created on 2025/05/05 09:42:18
@author: SIRConceicao
'''

import os
import json
import pandas as pd

directory="GUTBRAIN/BENT_output/testset/"

def extract_number(file_path):
    file_name = file_path.split('/')[-1]
    number_part = file_name.split('_')[1].split('.')[0]
    return int(number_part)

# GET ALL ANNOTATED FILES
for root, _, files in os.walk(directory):  
    files=[os.path.join(root, filename) for filename in files] 
    sorted_file_paths = sorted(files, key=extract_number)

#===================================================================================
def brat2dataframe(annotation_path):
    with open(annotation_path, 'r') as file:
        text = file.read()

    lines = text.strip().split('\n')

    labels = []
    start_idxs = []
    end_idxs = []
    text_spans = []
    arg1_ids = []
    arg2_ids = []

    if lines != ['']: #no annotations
        last_line_type=""
        # Iterate through the lines and extract the data
        for idx, line in enumerate(lines):
            if line.startswith('T'):#ENTITIES
                if last_line_type=='T' and idx!=0:
                    #no link only entities
                    arg1_ids.append(None)
                    arg2_ids.append(None)
                
                last_line_type='T'
                parts = line.split('\t')
                entity_id = parts[0] # T1
                #parts[1] -> disease 4 21
                part_1=parts[1].split(" ")
                label = part_1[0] #disease 4 21
                start_idx = int(part_1[1])
                end_idx = int(part_1[2])
                text_span = parts[2]
                labels.append(label)
                start_idxs.append(start_idx)
                end_idxs.append(end_idx)
                text_spans.append(text_span)
            
        if idx==0 or idx>0 and line.startswith('T'): #only 1 ner line or none
            arg1_ids.append(None)
            arg2_ids.append(None)
    else: #no annotations
        pass

    df = pd.DataFrame({
        'label': labels,
        'start_idx': start_idxs,
        'end_idx': end_idxs,
        'text_span': text_spans,
    })

    return df
#===================================================================================
def remove_overlapping_entities(df):
    """Efficiently removes shorter overlapping entities while maintaining ordering."""
    df = df.sort_values(by=['start_idx', 'end_idx']).reset_index(drop=True)
    filtered_entities = []
    last_stop = -1

    for _, row in df.iterrows():
        if row['start_idx'] > last_stop:  
            filtered_entities.append(row)
            last_stop = row['end_idx']  

    return pd.DataFrame(filtered_entities).reset_index(drop=True)


#===================================================================================

def annotated_dicts(sorted_file_paths,cutoff_list,type):
   
    ner_nel_out=[]
    cutoff=len(cutoff_list)

    if type=="title":
        path= sorted_file_paths[:cutoff] # bent processed titles first
    elif type =="abstract":
        path= sorted_file_paths[cutoff:]

    for idx,file in enumerate(path):
        df= brat2dataframe(file)

        df=remove_overlapping_entities(df)
        
        if len(df)!=0:
            
            if type=="title":
                df['location']="title"
            
            elif type =="abstract":
                df['location']="abstract"

            ner_nel=df.to_dict(orient="records")
            ner_nel_out.append(ner_nel)

        else:#no annotations
            ner_nel=[]
            ner_nel_out.append(ner_nel)
           
    
    return ner_nel_out

#===================================================================================
doc_file="GUTBRAIN/articles_test.json"
with open(doc_file, 'r') as file:
    data = json.load(file)

titles=[data[doc]['title'] for doc in data]

abstracts= [data[doc]['abstract'] for doc in data]

#------------------------------------------------------------

ner_nel_title= annotated_dicts(sorted_file_paths,titles,"title")
ner_nel_abstract= annotated_dicts(sorted_file_paths,abstracts,"abstract")

full_ner = [ner_nel_title[i]  + ner_nel_abstract[i] for i in range(len(titles))]

for num,doc in enumerate(data):
    data[doc]['entities']=full_ner[num] 


with open("GUTBRAIN/articles_test_NER.json", "a") as file:
    json.dump(data, file,indent=2)
