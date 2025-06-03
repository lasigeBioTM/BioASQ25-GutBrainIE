#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
Created on 2025/05/01 15:27:52
@author: SIRConceicao

BASELINE FOR RE WITH NER

BENTMISTRAL System
'''
import os
import sys
from pathlib import Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import re
import json
import torch
from prompts.qwen_prompts import system_prompts
from prompts.llama_prompts import system_prompts as llama_sysprompt
from misc.utils import defined_relations,format_defined_relations
import llms_class

#==================================================================================================
def load_data(doc_path):
    with open(doc_path, 'r') as file:
        data = json.load(file)
    return data
#==================================================================================================
def dataset_info (doc_path):
    with open(doc_path, 'r') as file:
        data = json.load(file)
    titles=[data[doc]['metadata']['title_tagged'] \
        for doc in data]

    abstracts= [data[doc]['metadata']['abstract_tagged'] \
            for doc in data]

    hypergraphs= [data[doc]["relations_tagged_hypergraph"] \
            for doc in data]
    
    return titles, abstracts,hypergraphs
#==================================================================================================
def messages_format(system_prompt,defined_relations,tagged_text):
    """mistral and qwen format"""
    prompt = system_prompt.format(defined_relations=defined_relations,sentence=tagged_text)
    
    messages = [
        {"role": "user", "content": prompt}
    ]

    return messages
#==================================================================================================
def messages_format_llama(system_prompt,defined_relations,tagged_text):
    """llama format"""
    sys_prompt = system_prompt['sys_prompt'].format(defined_relations=defined_relations)
    content_prompt= system_prompt['content_prompt'].format(sentence=tagged_text)
    
    messages = [
        {"role": "user", "system": sys_prompt},
        {"role": "user", "content": content_prompt}
    ]

    return messages
#==================================================================================================
def main():
    #doc_path = "data/GutBrainIE_tagged/Annotations/Dev/dev_tagged.json"
    doc_path = "data/GutBrainIE_tagged/Annotations/Test/lasigeBioTM_subtask6_1_NER_Mistral-7B-Instruct-v0.3_fixed_tagged.json"
    data=load_data(doc_path)

    
    #Subject->predicate->object rules
    defined_relations_dict=defined_relations()
    formated_rel_dict=format_defined_relations(defined_relations_dict)

    #---------------------------------------------------------------------
    #LLM CONFIGS
    configs={
        "temperature":0.2,
        "max_new_tokens":4096,
        "quantization":True
        }
    #---------------------------------------------------------------------
    #Mistral
    model_name = "mistralai/Mistral-7B-Instruct-v0.3"
    mistral=llms_class.Mistral_Pipeline(model_name,configs)
    system_prompt = system_prompts['prompt1']
    #---------------------------------------------------------------------
    output_path=f"data/intermediate/{model_name.split("/")[-1]}-baseline-labels-it-outputs.json"
    output_logs=f"data/intermediate/{model_name.split("/")[-1]}-baseline-labels-it-outputs.LOGS"

    with open(output_logs,'w') as logfile:
        logfile.write(f"Input File {doc_path}\nLLM Config:\n{configs}\n\n")
        for annot in data:
            jbase= data[annot]
            # metadata = jbase['metadata']
            # title = metadata['title_tagged']
            # abstract = metadata['abstract_tagged']

            title = jbase['title_tagged']
            abstract = jbase['abstract_tagged']

            tagged_text = title + "\n " + abstract

            messages=messages_format(system_prompt,formated_rel_dict,tagged_text)
            output= mistral.inference(messages)
            content = output.split("[/INST]")[-1].strip("]</s>").strip()

            print(content)
            logfile.write(f"{annot}\n{output}\n\n")

            try:
                answer = json.loads(f"[{content}]")
                jbase['ternary_tag_based_relations_predicted']=answer['ternary_tag_based_relations']
                jbase['ternary_mention_based_relations_predicted']=answer['ternary_mention_based_relations']

            except TypeError: #[{'ternary_tag_based_relations': [...], 'ternary_mention_based_relations': [...]}]
                answer = json.loads(f"[{content}]")[0]
                jbase['ternary_tag_based_relations_predicted']=answer['ternary_tag_based_relations']
                jbase['ternary_mention_based_relations_predicted']=answer['ternary_mention_based_relations']
            
            except json.decoder.JSONDecodeError:
                answer= content
                print(f"eeeeerrrorr\n")
                jbase['llm_output']=answer
            
            torch.cuda.empty_cache()
    with open(output_path, "w") as f: 
        json.dump(data, f, indent=2)

#==================================================================================================

if __name__ == "__main__":
    main()
