#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
Created on 2025/05/05 08:54:54
@author: SIRConceicao
'''
import json
import bent.annotate as bt


doc_path= "GUTBRAIN/articles_test.json"

with open(doc_path, 'r') as file:
        data = json.load(file)
titles=[data[doc]['title'] \
        for doc in data]

abstracts= [data[doc]['abstract'] \
        for doc in data]

txt_list=titles + abstracts

#===================================================================================
dataset= bt.annotate(
        link=False,
        recognize=True,
        types={
            'disease': 'do',
            'chemical': 'chebi', 
            'gene': 'ncbi_gene',
            'organism': 'ncbi_taxon',
            'bioprocess': 'go_bp',
            'anatomical': '‘ctd_anat',
            'cell_component' : 'go_cc',
            'cell_line':'cellosaurus'
               },
        input_text=txt_list,
        out_dir='GUTBRAIN/BENT_output/testset/'
)
#===================================================================================