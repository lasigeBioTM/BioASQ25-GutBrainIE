#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
Created on 2025/04/27 17:28:43
@author: SIRConceicao
'''
import re

def remove_entity_type_tag (tagged_sentence):
    """Remove  @type$  @/type$ tags"""
    text = re.sub(r'@\w+\$', '', tagged_sentence)     
    text = re.sub(r'@/\w+\$', '', text)    
    return text
   

def remove_entity_number_tag (tagged_sentence):
    """ remove  <eN> and </eN> tags """
    return re.sub(r'</?e\d+>', '', tagged_sentence)

def remove_all_tags (tagged_sentence):
    text = remove_entity_number_tag(tagged_sentence)
    text = remove_entity_type_tag(text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\s+([.,;:!?])', r'\1', text)
    return text.strip()

