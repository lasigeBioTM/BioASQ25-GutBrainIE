#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
Created on 2025/05/01 15:27:52
@author: SIRConceicao
'''

import os
import re
import json

import sys
from pathlib import Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from prompts.qwen_prompts import system_prompts
from misc.utils import defined_relations,format_defined_relations
from llms_class import Qwen_Pipeline

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

    prompt = system_prompt.format(defined_relations=defined_relations,sentence=tagged_text)
    
    messages = [
        {"role": "user", "content": prompt}
    ]

    return messages
#==================================================================================================
def main():

    doc_path = "data/GutBrainIE_tagged/Annotations/Dev/dev_tagged.json"
    titles, abstracts,hypergraphs= dataset_info(doc_path)
    full_text = [titles[i] + "\n " + abstracts[i] for i in range(len(titles))]

    defined_relations_dict=defined_relations()
    formated_rel_dict=format_defined_relations(defined_relations_dict)

    #---------------------------------------------------------------------
    #Qwen
    model_name = "Qwen/Qwen3-8B"
    qwen=Qwen_Pipeline(model_name)
    #---------------------------------------------------------------------
    system_prompt = system_prompts['prompt1']
    
    for tagged_text in full_text[:5]:
        messages=messages_format(system_prompt,formated_rel_dict,tagged_text)
        thinking_content, content= qwen.inference(messages)
        print(f"Thinking{thinking_content}\n\ncontent{content}")

if __name__ == "__main__":
    main()
