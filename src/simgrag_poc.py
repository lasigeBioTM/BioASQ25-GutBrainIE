#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
Created on 2025/04/15 20:27:20
@author: SIRConceicao

Proof of concept
'''

import os 
import re
import torch
import json

from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer,BitsAndBytesConfig
from huggingface_hub import login
from dotenv import load_dotenv

import prompts.annot_dev_query

#==================================================================================================
load_dotenv()
HUGGINGFACE_TOKEN=os.getenv('HUGGINGFACE_TOKEN')
login(token=HUGGINGFACE_TOKEN)
#==================================================================================================

