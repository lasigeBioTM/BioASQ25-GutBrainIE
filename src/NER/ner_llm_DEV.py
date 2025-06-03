#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
Created on 2025/05/05 19:28:19
@author: SIRConceicao
'''
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import json
import torch
import llms_class
import re
#==================================================================================================
def load_data(doc_path):
    with open(doc_path, 'r') as file:
        data = json.load(file)
    return data
#==================================================================================================
def messages_format(title, abstract):
    """mistral and qwen format"""
    

    entities_description="""
        Anatomical Location → Body parts or regions
        Animal → Non-human organisms (e.g., mammals, insects)
        Biomedical Technique → Methods for medical research (e.g.,16S rDNA survey, metatranscriptomics analyses, ELISA kit)
        Bacteria → Single-celled microbes (e.g., E. coli, Collinsella aerofaciens)
        Chemical → Substances like drugs, metabolites, or toxins (e.g., simple sugars,metabolite acetate)
        Dietary Supplement → Pills/extracts for nutrition (e.g., vitamins, probiotic)
        DDF →  (Disease/Disorder/Finding) Medical conditions or symptoms
        Drug → Therapeutic or recreational substances (e.g AGPs, antibiotic)
        Food → Edible items for nutrition (e.g whole grain cereals)
        Gene → DNA units encoding traits/functions
        Human → Homo sapiens (e.g., patients, breast cancer survivors)
        Microbiome → Microbial communities in an environment (e.g., gut microbiota)
        Statistical Technique → Data analysis methods (e.g., Multiple regression analysis)
        """


    system_prompt = """You are an information extraction assistant working on biomedical named-entity recognition (NER).

    You will receive:
    1. Text to analyze:
        - title
        - abstract

    Task:
    1.Extract new entities that fit the guide. 
    
    Rules:
    - Mandatory Instructions:
        - NO rewording, summarization or paraphrasing.
        - NO explanations,thoughts, or caveats
        - Output format: Strictly adhere to the JSON example below.
    - Entity Selection:
        - Specificity: Extract the longest/most specific span (e.g., "human brain" > "brain").
        - Overlap resolution: Keep only the longest span among overlapping candidates.

    Output Format:
        {{
        "start_idx": 57,
        "end_idx": 83,
        "location": "title",
        "text_span": "CNS autoimmune inflammation",
        "label": "DDF"
        }}
    
    Labels and Definitions Guide:
    {entities_description}

    Text to analyze:
    Title: {title}
    Abstract: {abstract}

    Response Requirements:
    - ONLY OUTPUT VALID JSON. 
    - BEGIN RESPONSE WITH {{
    """


    prompt = system_prompt.format(entities_description=entities_description,
                                  title=title,
                                  abstract=abstract)
    
    messages = [
        {"role": "user", "content": prompt}
    ]

    return messages

#==================================================================================================
def remove_entity_type_tag (tagged_sentence):
    """Remove  @type$  @/type$ tags"""
    text = re.sub(r'@\w+\$', '', tagged_sentence)     
    text = re.sub(r'@/\w+\$', '', text)    
    return text
#==================================================================================================
def main():
    doc_path = "data/GutBrainIE_Full_Collection_2025/Annotations/Dev/json_format/dev.json"
    data=load_data(doc_path)

    #---------------------------------------------------------------------
    #LLM CONFIGS
    configs={
        "temperature":0.2,
        "max_new_tokens":1024, #4096
        "quantization":True
        }
    #---------------------------------------------------------------------
    #Mistral
    model_name = "mistralai/Mistral-7B-Instruct-v0.3"
    llm=llms_class.Mistral_Pipeline(model_name,configs)

    #Qwen
    # model_name = "Qwen/Qwen3-8B"
    # llm=llms_class.Qwen_Pipeline(model_name,configs)
    
    #---------------------------------------------------------------------

    output_path=f"data/intermediate/{model_name.split("/")[-1]}-NER-it-outputs-DEV.json"
    output_logs=f"data/intermediate/{model_name.split("/")[-1]}-NER-it-outputs-DEV.LOGS"

    with open(output_logs,'w') as logfile:
        logfile.write(f"LLM Config:\n{configs}\n\n")
        for annot in data:
            jbase= data[annot]['metadata']
          
            title = jbase['title']
            abstract = jbase['abstract']
            #candidate_ner = jbase['entities']


            messages=messages_format(title, abstract)
            a=0
            
            if "mistralai" in model_name:
                output= llm.inference(messages)
                print(output)
                
                content = output.split("[/INST]")[-1].strip("]</s>").strip()
            elif "Qwen" in model_name:
                #With thinking
                # thinking_content, content_llm= llm.inference(messages)
                # print(f"Thinking{thinking_content}\n\ncontent{content}\n")   

                content= llm.inference(messages)
                print(f"content{content}\n") 
                output = content

            logfile.write(f"{annot}\n{output}\n\n")

            try:
                answer = json.loads(f"[{content}]")
            except json.decoder.JSONDecodeError:
                answer= content
                print(f"eeeeerrrorr\n")
            print(f"{annot}\n{answer}\n")

            jbase['llm_output']=answer
            torch.cuda.empty_cache()


    with open(output_path, "w") as f: 
        json.dump(data, f, indent=2)

if __name__ == "__main__":
    main()
