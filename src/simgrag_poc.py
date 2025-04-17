#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
Created on 2025/04/15 20:27:20
@author: SIRConceicao

Proof of concept

https://github.com/YZ-Cai/SimGRAG/blob/main/pipeline/FactKG_query.py

'''

import os 
import re
import torch
import json

from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer,BitsAndBytesConfig
from huggingface_hub import login
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

import prompts.annot_dev_query
from retriever import Retriever
#==================================================================================================
load_dotenv()
HUGGINGFACE_TOKEN=os.getenv('HUGGINGFACE_TOKEN')
login(token=HUGGINGFACE_TOKEN)
#==================================================================================================

# load dataset
#Criar KG graph das ontologias
dataset = FactKG(configs)
KG = dataset.get_KG()
type_to_nodes = dataset.get_type_to_nodes()
all_queries = dataset.get_queries()
all_groundtruths = dataset.get_groundtruths()

#==================================================================================================
#Load LLM
model_id="mistralai/Mistral-7B-Instruct-v0.3"
def model_setup(model_id):
    #tem mesmo de ser com BNB senao demora muito muito tempo
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto",quantization_config=bnb_config)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    return model,tokenizer

def inference(message):
    input_text = tokenizer.apply_chat_template(message, tokenize=False)
    model_inputs = tokenizer(input_text, return_tensors="pt").to("cuda")

    attention_mask = model_inputs["attention_mask"]
    generated_ids = model.generate(model_inputs["input_ids"],
                                max_new_tokens=1024,
                                do_sample=True,
                                temperature=0.2,
                                attention_mask=attention_mask,
                                pad_token_id=tokenizer.eos_token_id
                                )
    
    output=tokenizer.batch_decode(generated_ids)[0]

    return output
#==================================================================================================
# load retriever
retriever = "retriever = Retriever(configs, KG, type_to_nodes)"

#==================================================================================================
def run(query, groundtruths):
	res = {
		'query': query,
		'groundtruths': groundtruths,
		'retriever_configs': configs['retriever'],
		'llm_configs': configs['llm'],
		'rewrite_shot': configs['rewrite_shot'],
		'answer_shot': configs['answer_shot'],
	}
	
	try:
		# rewrite
		start = time.time()
		res['rewrite_prompt'] = prompts.rewrite_FactKG.get(query, shot=res['rewrite_shot'])
		res['rewrite_llm_output'] = llm.chat(res['rewrite_prompt'])
		res['rewrite_time'] = time.time() - start
  
		# extract graph
		res['query_graph'] = extract_graph(res['rewrite_llm_output'])
		
		# subgraph matching
		start = time.time()
		res['retrieval_details'] = retriever.retrieve(res['query_graph'], mode='greedy')
		res['evidences'] = [each[1] for each in res['retrieval_details']['results']]
		res['retrieval_time'] = time.time() - start

		# answer
		start = time.time()
		res['answer_prompt'] = prompts.answer_FactKG.get(res['query'], res['evidences'], shot=res['answer_shot'])
		res['answer_llm_output'] = llm.chat(res['answer_prompt'])
		res['answer_time'] = time.time() - start
  
		# check answer
		res['correct'] = check_answer(res['answer_llm_output'], groundtruths)
  
	except Exception as e:
		res['error_message'] = str(e)
  
	return res

# run for all queries
result_file = configs["output_filename"]
for query, groundtruths in tqdm(zip(all_queries, all_groundtruths), total=len(all_queries)):
	res = run(query, groundtruths)
	with open(result_file, 'a', encoding='utf-8') as f:
		f.write(json.dumps(res, ensure_ascii=False) + '\n')