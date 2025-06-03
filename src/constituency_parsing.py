#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
Created on 2025/04/29 10:11:08
@author: SIRConceicao
'''
import re
import json
import spacy
import benepar
#benepar.download('benepar_en3') to download only once
from benepar.spacy_plugin import BeneparComponent

from nltk import Tree
from collections import defaultdict
#===================================================================================
def spacy_setup():
    nlp = spacy.load("en_core_web_trf")
    nlp.add_pipe('sentencizer')
    nlp.add_pipe('benepar', config={'model': 'benepar_en3'})

    return nlp

#===================================================================================
#  ** NER FUNCTIONS
#===================================================================================

def extract_entities_with_untagged_positions(text):
    """Gets entities offsets for the sentences without tags"""
    # Pattern to match outer tags with inner type tags
    pattern = re.compile(r'<e(\d+)>\s*@([^\s$]+(?:\s[^\s$]+)*)\$\s*(.*?)\s*@/\2\$\s*</e\1>', re.DOTALL)
    
    # We'll build the untagged text as we process matches
    untagged_text = []
    last_pos = 0
    entities = []
    
    for match in pattern.finditer(text):
        # Text before the current match
        untagged_text.append(text[last_pos:match.start()])
        last_pos = match.end()
        
        # Extract entity information
        outer_tag_num = match.group(1)
        entity_type = match.group(2)
        entity_text = match.group(3).strip()
        
        # Calculate positions in the untagged text
        start_pos = len(''.join(untagged_text))
        end_pos = start_pos + len(entity_text)
        
        # Add the entity text to our untagged text
        untagged_text.append(entity_text)
        
        entities.append({
            'entity': entity_text,
            'start': start_pos,
            'end': end_pos,
            'type': entity_type,
            'outer_tag': f'e{outer_tag_num}'
        })
    
    # Add any remaining text after the last match
    untagged_text.append(text[last_pos:])
    
    return {
        'entities': entities,
        'untagged_text': ''.join(untagged_text)
    }

#===================================================================================
def merge_spans(spans):
    """Function to merge overlapping spans"""
    # Sort spans by their start position
    spans = sorted(spans, key=lambda span: span.start)
    merged_spans = []
    current_span = None

    for span in spans:
        if current_span is None:
            current_span = span
        else:
            # If spans overlap or are adjacent, merge them
            if span.start <= current_span.end:
                #current_span = Span(doc, min(current_span.start, span.start), max(current_span.end, span.end), label=current_span.label_)
                current_span = span(doc, min(current_span.start, span.start), max(current_span.end, span.end), label=current_span.label_)
            else:
                merged_spans.append(current_span)
                current_span = span

    if current_span is not None:
        merged_spans.append(current_span)

    return merged_spans
#===================================================================================
def create_ner_spans(doc, extraction_result):
    """Create and merge NER spans from the extracted entities.
    
    Args:
        doc: The spaCy Doc object
        extraction_result: Result dict from extract_entities_with_untagged_positions()
    
    Returns:
        The doc with updated entities
    """
    spans = []
    
    # Create spans for each entity
    for entity in extraction_result['entities']:
        span = doc.char_span(
            entity['start'], 
            entity['end'], 
            label=entity['type']  # entity type as the label
        )
        if span is not None:
            spans.append(span)
    
    # Merge overlapping spans
    merged_spans = merge_spans(spans)  
    
    # Update the doc.ents
    doc.ents = merged_spans
    
    return doc
#===================================================================================
def find_entity_relations_simple(doc, parse_tree):
    """Find grammatical relations between entities using parse tree structure"""
    relations = []
    
    # Find all VP (verb phrase) nodes that contain multiple entities
    for subtree in parse_tree.subtrees():
        if subtree.label().startswith('VP'):
            entities_in_vp = []
            for leaf in subtree.leaves():
                for ent in doc.ents:
                    if leaf in ent.text:
                        entities_in_vp.append(ent)
            
            # If we found multiple entities in the same VP, they're likely related
            if len(entities_in_vp) > 1:
                relations.append({
                    'type': 'VP_relation',
                    'entities': [(e.text, e.label_) for e in entities_in_vp],
                    'context': ' '.join(subtree.leaves())
                })
    
    return relations
#===================================================================================
def find_flexible_relations(doc, parse_tree):
    """Find subject-predicate-object relations with enhanced relation typing"""
    relations = []
    entity_map = {(ent.start, ent.end): ent for ent in doc.ents}
    
    # Process all verbs in the sentence
    for token in doc:
        if token.pos_ == 'VERB':
            # Find subjects and objects
            subj = next((t for t in token.lefts if t.dep_ in ('nsubj', 'nsubjpass')), None)
            obj = next((t for t in token.rights if t.dep_ in ('dobj', 'obj')), None)
            
            # Check for prepositional objects - fixed syntax here
            if not obj:
                for prep in token.rights:
                    if prep.dep_ == 'prep':
                        obj = next((t for t in prep.rights if t.dep_ == 'pobj'), None)
                        break
            
            # Get containing entities
            subj_ent = find_containing_entity(subj, entity_map) if subj else None
            obj_ent = find_containing_entity(obj, entity_map) if obj else None
            
            if subj_ent and obj_ent:
                relations.append({
                    'subject': (subj_ent.text, subj_ent.label_),
                    'predicate': token.lemma_,
                    'object': (obj_ent.text, obj_ent.label_),
                    'relation_type': determine_relation_type(token, subj_ent, obj_ent),
                    'context': token.sent.text,
                    'confidence': calculate_confidence(token, subj, obj)
                })
    
    return relations

def find_containing_entity(token, entity_map):
    """Find the entity that contains this token"""
    for (start, end), ent in entity_map.items():
        if start <= token.i < end:
            return ent
    return None

def calculate_confidence(token, subj, obj):
    """Calculate confidence score for the relation"""
    score = 0.5  # Base score
    
    # Boost for clear grammatical relations
    if subj and obj:
        score += 0.3
    elif subj or obj:
        score += 0.15
    
    # Boost for certain verb types
    if token.lemma_.lower() in ('affect', 'cause', 'increase', 'decrease'):
        score += 0.1
        
    return min(1.0, max(0.5, score))  # Keep between 0.5-1.0

def determine_relation_type(verb, subj_ent, obj_ent):
    """Classify the relation type using the provided label set"""
    verb_lemma = verb.lemma_.lower()
    subj_type = subj_ent.label_
    obj_type = obj_ent.label_
    
    # Mapping from verb patterns to relation labels
    relation_map = {
        ('locate', 'reside', 'found'): "Located In",
        ('affect', 'influence', 'alter', 'modify'): "Affect",
        ('target', 'attack', 'bind'): "Target",
        ('link', 'associate', 'correlate'): "Is Linked To",
        ('influence', 'regulate', 'control'): "Influence",
        ('impact', 'affect', 'change'): "Impact",
        ('increase', 'decrease', 'reduce'): "Change Effect",
        ('be', 'represent'): "Is A",
        ('use', 'utilize', 'employ'): "Used By",
        ('administer', 'apply', 'give'): "Administered",
        ('comprise', 'contain', 'include'): "Part Of",
        ('damage', 'harm', 'impair'): "Strike",
        ('produce', 'generate', 'create'): "Produced By",
        ('express', 'transcribe', 'translate'): "Change Expression",
        ('compare', 'contrast'): "Compared To",
        ('interact', 'bind', 'react'): "Interact"
    }
    
    # Special case for abundance changes
    if verb_lemma in ('increase', 'decrease') and obj_type in ('MICROBE', 'CHEMICAL'):
        return "Change Abundance"
    
    # Find the most specific match
    for verbs, label in relation_map.items():
        if verb_lemma in verbs:
            return label
    
    # Default cases based on entity types
    if subj_type == "MICROBIOME" and obj_type == "DISEASE":
        return "Is Linked To"
    if subj_type == "DRUG" and obj_type == "DISEASE":
        return "Target"
    
    return "Interact"

######################################################
######################################################
def extract_cross_sentence_relations(doc):
    """Find meaningful relations using both discourse markers and your relation labels"""
    relations = []
    entity_tracker = defaultdict(list)
    
    # Enhanced discourse mapping incorporating your relation labels
    discourse_relations = {
        # Causal relationships
        "therefore": "Influence", 
        "thus": "Influence",
        "consequently": "Impact",
        "because": "Change Effect",
        "since": "Change Effect",
        
        # Contrast relationships
        "however": "Compared To",
        "although": "Compared To",
        "nevertheless": "Compared To",
        
        # Composition relationships
        "containing": "Part Of",
        "including": "Part Of",
        
        # Spatial relationships
        "within": "Located In",
        "inside": "Located In"
    }
    
    # Additional verb-based relation mapping
    verb_relations = {
        "affect": "Affect",
        "influence": "Influence",
        "target": "Target",
        "link": "Is Linked To",
        "impact": "Impact",
        "change": "Change Effect",
        "increase": "Change Abundance",
        "decrease": "Change Abundance",
        "produce": "Produced By",
        "express": "Change Expression",
        "compare": "Compared To",
        "interact": "Interact",
        "strike": "Strike",
        "use": "Used By",
        "administer": "Administered"
    }

    # Track all entities
    for sent_idx, sent in enumerate(doc.sents):
        for ent in sent.ents:
            entity_tracker[ent.text].append((sent_idx, ent))
    
    sentences = list(doc.sents)
    for i in range(1, len(sentences)):
        current_sent = sentences[i]
        prev_sent = sentences[i-1]
        
        current_ents = {ent.text: ent for ent in current_sent.ents}
        prev_ents = {ent.text: ent for ent in prev_sent.ents}
        
        # 1. Check for discourse markers first
        relation_type = None
        for token in current_sent:
            lower_token = token.text.lower()
            if lower_token in discourse_relations:
                relation_type = discourse_relations[lower_token]
                break
        
        # 2. If no discourse marker, check for connecting verbs
        if not relation_type:
            for token in current_sent:
                if token.pos_ == "VERB":
                    lemma = token.lemma_.lower()
                    if lemma in verb_relations:
                        relation_type = verb_relations[lemma]
                        break
        
        # Create relations for entity pairs
        for prev_ent in prev_ents.values():
            for curr_ent in current_ents.values():
                if prev_ent.text != curr_ent.text:
                    relations.append({
                        'subject': (prev_ent.text, prev_ent.label_),
                        'predicate': relation_type if relation_type else "Is Linked To",  # Default fallback
                        'object': (curr_ent.text, curr_ent.label_),
                        'context': f"{prev_sent.text} ||| {current_sent.text}",
                        'evidence': "discourse" if relation_type in discourse_relations.values() else "verb"
                    })
    
    return relations
#######################################################
######################################################
def are_connected(sentence, ent1, ent2):
    try:
        tree = Tree.fromstring(sentence._.parse_string)
    except:
        return False  # In case parsing fails

    ent1_start = ent1.start - sentence.start
    ent1_end = ent1.end - sentence.start
    ent2_start = ent2.start - sentence.start
    ent2_end = ent2.end - sentence.start

    ent1_leaves = list(range(ent1_start, ent1_end))
    ent2_leaves = list(range(ent2_start, ent2_end))

    valid_labels = {'VP', 'PP', 'S', 'NP', 'CLAUSE'}

    for l1 in ent1_leaves:
        for l2 in ent2_leaves:
            try:
                path1 = tree.leaf_treeposition(l1)
                path2 = tree.leaf_treeposition(l2)
            except IndexError:
                continue  # If leaf index is out of bounds (due to parsing issues)

            lca_path = []
            for p1, p2 in zip(path1, path2):
                if p1 == p2:
                    lca_path.append(p1)
                else:
                    break

            if not lca_path:
                continue

            lca_node = tree
            for p in lca_path:
                if isinstance(lca_node, Tree) and p < len(lca_node):
                    lca_node = lca_node[p]
                else:
                    break

            if isinstance(lca_node, Tree) and lca_node.label() in valid_labels:
                return True

    return False

def extract_relations(doc, relations_dict):
    relations = []
    entities = list(doc.ents)
    
    for i, head_ent in enumerate(entities):
        for j, tail_ent in enumerate(entities):
            if i == j:
                continue  # Skip same entity
            
            for rel in relations_dict:
                if head_ent.label_ == rel['Head Entity'] and tail_ent.label_ in rel['Tail Entities']:
                    predicate = rel['Predicate']
                    
                    if head_ent.sent == tail_ent.sent:
                        if are_connected(head_ent.sent, head_ent, tail_ent):
                            relations.append({
                                'Head': head_ent.text,
                                'Head_Label': head_ent.label_,
                                'Tail': tail_ent.text,
                                'Tail_Label': tail_ent.label_,
                                'Predicate': predicate
                            })
                    else:
                        relations.append({
                            'Head': head_ent.text,
                            'Head_Label': head_ent.label_,
                            'Tail': tail_ent.text,
                            'Tail_Label': tail_ent.label_,
                            'Predicate': predicate
                        })
    
    # Remove duplicates
    unique_relations = []
    seen = set()
    for rel in relations:
        key = (rel['Head'], rel['Predicate'], rel['Tail'])
        if key not in seen:
            seen.add(key)
            unique_relations.append(rel)
    
    return unique_relations

#######################################################
######################################################
def get_constituency_trees(doc):
       
    results = []
    for sent in doc.sents:
        parse_tree = Tree.fromstring(sent._.parse_string)
        # results.append({
        #     "sentence": sent.text,
        #     "constituency_tree": str(parse_tree).strip()
        # })
        results.append(str(parse_tree).strip())
    return results
#===================================================================================
def analyze_sentence_with_entities_no_parsing(nlp,sentence,defined_relations_dict):
    # Step 1: Extract entities and get clean text
    extraction_result = extract_entities_with_untagged_positions(sentence)
    clean_text = extraction_result['untagged_text']
    
    #NOTE para o test
    clean_text=" ".join(clean_text.split())
    # Step 2: Process with spaCy
    doc = nlp(clean_text)
    
    # Step 3: Add NER spans
    doc = create_ner_spans(doc, extraction_result)
    # for ent in doc.ents:
    #     print(f"{ent.text} ({ent.label_}) from {ent.start_char} to {ent.end_char}")
    #NOTE no parsing
    relations = extract_cross_sentence_relations(doc)
    # for rel in relations:
    #     print(f"{rel['subject'][0]} ({rel['subject'][1]}) → {rel['predicate']} → {rel['object'][0]} ({rel['object'][1]})")
    #     print(f"Context: {rel['context']}\n")
 
    return relations

#===================================================================================
def analyze_sentence_with_entities_parsing(nlp,sentence,defined_relations_dict):
    # Step 1: Extract entities and get clean text
    extraction_result = extract_entities_with_untagged_positions(sentence)
    clean_text = extraction_result['untagged_text']
    #NOTE para o test
    clean_text=" ".join(clean_text.split())
    # Step 2: Process with spaCy
    doc = nlp(clean_text)
    
    # Step 3: Add NER spans
    doc = create_ner_spans(doc, extraction_result)
    # for ent in doc.ents:
    #     print(f"{ent.text} ({ent.label_}) from {ent.start_char} to {ent.end_char}")
   
    #NOTE com PARSING
    # relations_cp = extract_relations(doc, defined_relations_dict)

    # for rel in relations_cp:
    #     print(f"{rel['Head']} ({rel['Head_Label']}) -- {rel['Predicate']} --> {rel['Tail']} ({rel['Tail_Label']})")

    cp_trees=get_constituency_trees(doc)
 
    return cp_trees