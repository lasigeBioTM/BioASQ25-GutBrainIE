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

def defined_relations():
    return [
    {"Head Entity": "Bacteria", "Tail Entities": ["Animal"], "Predicate": "located in"},
    {"Head Entity": "Bacteria", "Tail Entities": ["Microbiome"], "Predicate": "part of"},

    {"Head Entity": "Chemical", "Tail Entities": ["Anatomical Loc.", "Human", "Animal"], "Predicate": "located in"},
    {"Head Entity": "Chemical", "Tail Entities": ["Chemical"], "Predicate": "interact"},
    {"Head Entity": "Chemical", "Tail Entities": ["Chemical"], "Predicate": "part of"},
    {"Head Entity": "Chemical", "Tail Entities": ["Bacteria", "Microbiome"], "Predicate": "impact"},
    {"Head Entity": "Chemical", "Tail Entities": ["Microbiome"], "Predicate": "produced by"},
    {"Head Entity": "Chemical", "Tail Entities": ["Disease, Disorder, or Finding"], "Predicate": "influence"},
    {"Head Entity": "Chemical", "Tail Entities": ["Gene"], "Predicate": "change expression"},

    {"Head Entity": "Dietary Supplement", "Tail Entities": ["Bacteria", "Microbiome"], "Predicate": "impact"},
    {"Head Entity": "Dietary Supplement", "Tail Entities": ["Disease, Disorder, or Finding"], "Predicate": "influence"},
    {"Head Entity": "Dietary Supplement", "Tail Entities": ["Gene"], "Predicate": "change expression"},
    
    {"Head Entity": "Drug", "Tail Entities": ["Bacteria", "Microbiome"], "Predicate": "impact"},
    {"Head Entity": "Drug", "Tail Entities": ["Gene"], "Predicate": "change expression"},
    {"Head Entity": "Drug", "Tail Entities": ["Chemical", "Drug"], "Predicate": "interact"},
    {"Head Entity": "Drug", "Tail Entities": ["Disease, Disorder, or Finding"], "Predicate": "change effect"},

    {"Head Entity": "Food", "Tail Entities": ["Bacteria", "Microbiome"], "Predicate": "impact"},
    {"Head Entity": "Food", "Tail Entities": ["Disease, Disorder, or Finding"], "Predicate": "influence"},
    {"Head Entity": "Food", "Tail Entities": ["Gene"], "Predicate": "change expression"},

    {"Head Entity": "Disease, Disorder, or Finding", "Tail Entities": ["Anatomical Location"], "Predicate": "strike"},
    {"Head Entity": "Disease, Disorder, or Finding", "Tail Entities": ["Bacteria", "Microbiome"], "Predicate": "change abundance"},
    {"Head Entity": "Disease, Disorder, or Finding", "Tail Entities": ["Chemical"], "Predicate": "interact"},
    {"Head Entity": "Disease, Disorder, or Finding", "Tail Entities": ["Disease, Disorder, or Finding"], "Predicate": "affect"},
    {"Head Entity": "Disease, Disorder, or Finding", "Tail Entities": ["Disease, Disorder, or Finding"], "Predicate": "is a"},
    {"Head Entity": "Disease, Disorder, or Finding", "Tail Entities": ["Human", "Animal"], "Predicate": "target"},

    {"Head Entity": "Human", "Tail Entities": ["Biomedical Technique"], "Predicate": "used by"},
    {"Head Entity": "Animal", "Tail Entities": ["Biomedical Technique"], "Predicate": "used by"},

    {"Head Entity": "Microbiome", "Tail Entities": ["Anatomical Loc.", "Human", "Animal"], "Predicate": "located in"},
    {"Head Entity": "Microbiome", "Tail Entities": ["Gene"], "Predicate": "change expression"},
    {"Head Entity": "Microbiome", "Tail Entities": ["Disease, Disorder, or Finding"], "Predicate": "is linked to"},
    {"Head Entity": "Microbiome", "Tail Entities": ["Microbiome"], "Predicate": "compared to"}
]

def format_defined_relations(relationships):

    # Group relationships by head entity and predicate
    grouped = {}
    head_order = []  # Maintain original head entity order
    for rel in relationships:
        head = rel["Head Entity"]
        predicate = rel["Predicate"]
        tail = rel["Tail Entities"]
        
        # Convert tail to list if it isn't already
        tails = tail if isinstance(tail, list) else [tail]
        
        # Track order of head entities
        if head not in grouped:
            grouped[head] = {"predicates": {}, "predicate_order": []}
            head_order.append(head)
            
        # Track order of predicates per head entity
        if predicate not in grouped[head]["predicates"]:
            grouped[head]["predicates"][predicate] = []
            grouped[head]["predicate_order"].append(predicate)
            
        grouped[head]["predicates"][predicate].extend(tails)
    
    # Generate formatted lines
    lines = []
    for head in head_order:
        for predicate in grouped[head]["predicate_order"]:
            tails = grouped[head]["predicates"][predicate]
            # Remove duplicates while preserving order
            unique_tails = []
            [unique_tails.append(t) for t in tails if t not in unique_tails]
            # Format tail entities
            tail_str = unique_tails[0] if len(unique_tails) == 1 else f"[{', '.join(unique_tails)}]"
            lines.append(f"{head} -> {predicate} -> {tail_str}")
    
    # Add numbering and combine into string
    return "\n".join([f"{i+1}. {line}" for i, line in enumerate(lines)])

