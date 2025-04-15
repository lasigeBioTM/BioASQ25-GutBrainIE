#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
Created on 2025/04/15 20:43:41
@author: SIRConceicao
'''
#TODO
examples =[]

def prompt (sentence,shot=12):
    #Retirado daqui
    #https://github.com/YZ-Cai/SimGRAG/blob/main/prompts/rewrite_FactKG.py

    prompt = """You need to segment the given query then extract the potential knowledge graph structures.

    Notes)
    1). Use the original description in the query with enough context, NEVER use unspecific words like 'in', 'is', 'for', 'of', 'have', 'go to', etc.
    2). For nodes or relations that are unknown, you can use the keyword 'UNKNOWN' with a unique ID, e.g., 'UNKNOWN artist 1', 'UNKNOWN relation 1'.
    3). For statements with negations, such as 'not', 'wasn't', 'didn't', you should use the keyword 'UNKNOWN' for the negated nodes, e.g., 'A does not live in B' results in the triple ('A', 'live in', 'UNKNOWN location 1').
    4). For values without textual semantic meanings, such as numbers, heights and speeds, you should only preserve the value itself with double quotes without any units, e.g., '"156"'. For large numbers with lots of 0000, use the scientific notation, e.g., '"5.2E06"'.
    5). Return the segmented query and extracted graph structures strictly following the format:
        {
            "divided": [
                "segment 1",
                ...
            ],
            "graph": [
                ('head', 'relation', 'tail'),
                ...
            ]
        }
    6). NEVER provide extra descriptions or explanations, such as something like 'Here is the extracted knowledge graph structure'.

    Examples)
    """

    for i in range(min(shot, len(examples))):
            prompt += f"""
    {i+1}. query: '{examples[i]['query']}'
    {{
        "divided": {examples[i]['divided']},
        "graph": {examples[i]['graph']}
    }}
    """

        return prompt + f"""
    Your task)
    **Read and follow the instructions and examples step by step**
    sentence: '{sentence}'
    """