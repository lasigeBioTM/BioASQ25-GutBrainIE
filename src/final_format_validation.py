import json

# File paths
# INPUT_FILE = 'data/processed/lasigeBioTM_Mistral-7B-Instruct-v0.3-baseline-labels-it-outputs.json'
# OUTPUT_FILE = 'data/processed/lasigeBioTM_Mistral-7B-Instruct-v0.3-baseline-labels-it-outputs_CLEANED.json'


# INPUT_FILE = "data/processed//Mistral-7B-Instruct-v0.3-raw-baseline-it-outputs.json"
# OUTPUT_FILE = "data/processed//Mistral-7B-Instruct-v0.3-raw-baseline-it-outputs_CLEANED.json"

# INPUT_FILE = "data/processed/Mistral-7B-Instruct-v0.3-spacy-semantics-it-outputs.json"
# OUTPUT_FILE = "data/processed/Mistral-7B-Instruct-v0.3-spacy-semantics-it-outputs_CLEANED.json"

INPUT_FILE = "data/processed/Mistral-7B-Instruct-v0.3-constparsing-it-outputs.json"
OUTPUT_FILE = "data/processed/Mistral-7B-Instruct-v0.3-constparsing-it-outputs_CLEANED.json"

# Validation configuration
TEST_6_1_NER = True
TEST_6_3_TERNARY_TAG_RE = True
TEST_6_4_TERNARY_MENTION_RE = True

# Debugging options
print_all_details = False
print_article_details = False

# Legal labels
LEGAL_ENTITY_LABELS = [
    "anatomical location", "animal", "bacteria", "biomedical technique",
    "chemical", "DDF", "dietary supplement", "drug", "food", "gene",
    "human", "microbiome", "statistical technique"
]

LEGAL_RELATION_LABELS = [
    "administered", "affect", "change abundance", "change effect",
    "change expression", "compared to", "impact", "influence", "interact",
    "is a", "is linked to", "located in", "part of", "produced by",
    "strike", "target", "used by"
]

def extract_tag_relation_from_mention(mention_relation):
    """Extract tag-based relation from a mention-based relation."""
    return {
        "subject_label": mention_relation["subject_label"],
        "predicate": mention_relation["predicate"],
        "object_label": mention_relation["object_label"]
    }

def is_valid_entity(entity):
    """Check if an entity dictionary is valid."""
    required_fields = ["start_idx", "end_idx", "location", "text_span", "label"]
    if not all(field in entity for field in required_fields):
        if print_all_details:
            print(f"Removing entity missing required fields: {entity}")
        return False
    
    if str(entity["label"]) not in LEGAL_ENTITY_LABELS:
        if print_all_details:
            print(f"Removing entity with illegal label '{entity['label']}': {entity}")
        return False
    
    return True

def is_valid_ternary_tag_relation(relation):
    """Check if a ternary tag relation dictionary is valid."""
    required_fields = ["subject_label", "predicate", "object_label"]
    if not all(field in relation for field in required_fields):
        if print_all_details:
            print(f"Removing tag relation missing required fields: {relation}")
        return False
    
    subj_label = str(relation["subject_label"])
    predicate = str(relation["predicate"])
    obj_label = str(relation["object_label"])
    
    if subj_label not in LEGAL_ENTITY_LABELS:
        if print_all_details:
            print(f"Removing tag relation with illegal subject label '{subj_label}': {relation}")
        return False
    
    if obj_label not in LEGAL_ENTITY_LABELS:
        if print_all_details:
            print(f"Removing tag relation with illegal object label '{obj_label}': {relation}")
        return False
    
    if predicate not in LEGAL_RELATION_LABELS:
        if print_all_details:
            print(f"Removing tag relation with illegal predicate '{predicate}': {relation}")
        return False
    
    return True

def is_valid_ternary_mention_relation(relation):
    """Check if a ternary mention relation dictionary is valid."""
    required_fields = [
        "subject_text_span", "subject_label", "predicate",
        "object_text_span", "object_label"
    ]
    if not all(field in relation for field in required_fields):
        if print_all_details:
            print(f"Removing mention relation missing required fields: {relation}")
        return False
    
    subj_label = str(relation["subject_label"])
    predicate = str(relation["predicate"])
    obj_label = str(relation["object_label"])
    
    if subj_label not in LEGAL_ENTITY_LABELS:
        if print_all_details:
            print(f"Removing mention relation with illegal subject label '{subj_label}': {relation}")
        return False
    
    if obj_label not in LEGAL_ENTITY_LABELS:
        if print_all_details:
            print(f"Removing mention relation with illegal object label '{obj_label}': {relation}")
        return False
    
    if predicate not in LEGAL_RELATION_LABELS:
        if print_all_details:
            print(f"Removing mention relation with illegal predicate '{predicate}': {relation}")
        return False
    
    return True

def filter_article_data(article_data):
    """Filter an article's data, keeping only valid entries and ensuring required fields exist."""
    cleaned_data = {}
    
    # Process entities
    if TEST_6_1_NER:
        cleaned_data['entities'] = (
            [e for e in article_data.get('entities', []) if is_valid_entity(e)]
        )
    
    # Initialize relations lists
    if TEST_6_4_TERNARY_MENTION_RE:
        cleaned_data['ternary_mention_based_relations'] = []
    
    if TEST_6_3_TERNARY_TAG_RE:
        cleaned_data['ternary_tag_based_relations'] = []
    
    # Process mention relations
    if TEST_6_4_TERNARY_MENTION_RE and 'ternary_mention_based_relations' in article_data:
        mention_relations = [r for r in article_data['ternary_mention_based_relations'] 
                           if is_valid_ternary_mention_relation(r)]
        cleaned_data['ternary_mention_based_relations'] = mention_relations
    
    # Process tag relations
    if TEST_6_3_TERNARY_TAG_RE:
        tag_relations = []
        
        # First try to use existing tag relations
        if 'ternary_tag_based_relations' in article_data:
            tag_relations = [r for r in article_data['ternary_tag_based_relations'] 
                           if is_valid_ternary_tag_relation(r)]
            
            if print_article_details and tag_relations:
                print(f"Kept {len(tag_relations)} valid tag relations from original")
        
        # If no valid tag relations, try to extract from mentions
        if not tag_relations and 'ternary_mention_based_relations' in cleaned_data:
            tag_relations = [extract_tag_relation_from_mention(r) 
                           for r in cleaned_data['ternary_mention_based_relations']]
            tag_relations = [r for r in tag_relations if is_valid_ternary_tag_relation(r)]
            
            if print_article_details and tag_relations:
                print(f"Extracted {len(tag_relations)} tag relations from mention relations")
        
        if tag_relations:
            cleaned_data['ternary_tag_based_relations'] = tag_relations
    
    return cleaned_data

def process_and_validate_predictions(input_path, output_path):
    """Main processing function that loads, validates and saves the data."""
    try:
        with open(input_path, 'r', encoding='utf-8') as file:
            predictions = json.load(file)
    except OSError:
        raise OSError(f'Error opening input file: {input_path}')
    
    # Process all articles
    cleaned_predictions = {
        pmid: filter_article_data(article_data)
        for pmid, article_data in predictions.items()
    }
    
    # Calculate statistics
    stats = {
        'total_articles': len(cleaned_predictions),
        'articles_with_entities': sum(1 for a in cleaned_predictions.values() if a.get('entities', [])),
        'total_entities': sum(len(a.get('entities', [])) for a in cleaned_predictions.values()),
        'articles_with_tag_relations': sum(1 for a in cleaned_predictions.values() if a.get('ternary_tag_based_relations', [])),
        'total_tag_relations': sum(len(a.get('ternary_tag_based_relations', [])) for a in cleaned_predictions.values()),
        'articles_with_mention_relations': sum(1 for a in cleaned_predictions.values() if a.get('ternary_mention_based_relations', [])),
        'total_mention_relations': sum(len(a.get('ternary_mention_based_relations', [])) for a in cleaned_predictions.values()),
        'articles_with_extracted_tag_relations': sum(
            1 for a in cleaned_predictions.values() 
            if a.get('ternary_tag_based_relations', []) and 
            not any('subject_text_span' in r for r in a.get('ternary_tag_based_relations', []))
        )
    }
    
    # Save cleaned data
    try:
        with open(output_path, 'w', encoding='utf-8') as file:
            json.dump(cleaned_predictions, file, indent=2, ensure_ascii=False)
    except OSError:
        raise OSError(f'Error saving to output file: {output_path}')
    
    # Print summary
    print("\n===== Cleaning Summary =====")
    print(f"Processed {stats['total_articles']} articles")
    
    if TEST_6_1_NER:
        print(f"\nEntities:")
        print(f"  Articles with entities: {stats['articles_with_entities']}")
        print(f"  Total entities kept: {stats['total_entities']}")
    
    if TEST_6_3_TERNARY_TAG_RE:
        print(f"\nTernary Tag Relations:")
        print(f"  Articles with relations: {stats['articles_with_tag_relations']}")
        print(f"  Total relations kept: {stats['total_tag_relations']}")
        print(f"  Articles with relations extracted from mentions: {stats['articles_with_extracted_tag_relations']}")
    
    if TEST_6_4_TERNARY_MENTION_RE:
        print(f"\nTernary Mention Relations:")
        print(f"  Articles with relations: {stats['articles_with_mention_relations']}")
        print(f"  Total relations kept: {stats['total_mention_relations']}")
    
    print("\nCleaned data saved to:", output_path)

if __name__ == '__main__':
    process_and_validate_predictions(INPUT_FILE, OUTPUT_FILE)

################################################################################################################
# import json

# # File paths
# INPUT_FILE = 'data/processed/lasigeBioTM_Mistral-7B-Instruct-v0.3-baseline-labels-it-outputs_V2.json'
# OUTPUT_FILE = 'data/processed/lasigeBioTM_Mistral-7B-Instruct-v0.3-baseline-labels-it-outputs_CLEANED.json'

# # Validation configuration
# TEST_6_1_NER = True
# TEST_6_3_TERNARY_TAG_RE = True
# TEST_6_4_TERNARY_MENTION_RE = True

# # Debugging options
# print_all_details = False
# print_article_details = False

# # Legal labels
# LEGAL_ENTITY_LABELS = [
#     "anatomical location", "animal", "bacteria", "biomedical technique",
#     "chemical", "DDF", "dietary supplement", "drug", "food", "gene",
#     "human", "microbiome", "statistical technique"
# ]

# LEGAL_RELATION_LABELS = [
#     "administered", "affect", "change abundance", "change effect",
#     "change expression", "compared to", "impact", "influence", "interact",
#     "is a", "is linked to", "located in", "part of", "produced by",
#     "strike", "target", "used by"
# ]

# def extract_tag_relation_from_mention(mention_relation):
#     """Extract tag-based relation from a mention-based relation."""
#     return {
#         "subject_label": mention_relation["subject_label"],
#         "predicate": mention_relation["predicate"],
#         "object_label": mention_relation["object_label"]
#     }

# def is_valid_entity(entity):
#     """Check if an entity dictionary is valid."""
#     required_fields = ["start_idx", "end_idx", "location", "text_span", "label"]
#     if not all(field in entity for field in required_fields):
#         if print_all_details:
#             print(f"Removing entity missing required fields: {entity}")
#         return False
    
#     if str(entity["label"]) not in LEGAL_ENTITY_LABELS:
#         if print_all_details:
#             print(f"Removing entity with illegal label '{entity['label']}': {entity}")
#         return False
    
#     return True

# def is_valid_ternary_tag_relation(relation):
#     """Check if a ternary tag relation dictionary is valid."""
#     required_fields = ["subject_label", "predicate", "object_label"]
#     if not all(field in relation for field in required_fields):
#         if print_all_details:
#             print(f"Removing tag relation missing required fields: {relation}")
#         return False
    
#     subj_label = str(relation["subject_label"])
#     predicate = str(relation["predicate"])
#     obj_label = str(relation["object_label"])
    
#     if subj_label not in LEGAL_ENTITY_LABELS:
#         if print_all_details:
#             print(f"Removing tag relation with illegal subject label '{subj_label}': {relation}")
#         return False
    
#     if obj_label not in LEGAL_ENTITY_LABELS:
#         if print_all_details:
#             print(f"Removing tag relation with illegal object label '{obj_label}': {relation}")
#         return False
    
#     if predicate not in LEGAL_RELATION_LABELS:
#         if print_all_details:
#             print(f"Removing tag relation with illegal predicate '{predicate}': {relation}")
#         return False
    
#     return True

# def is_valid_ternary_mention_relation(relation):
#     """Check if a ternary mention relation dictionary is valid."""
#     required_fields = [
#         "subject_text_span", "subject_label", "predicate",
#         "object_text_span", "object_label"
#     ]
#     if not all(field in relation for field in required_fields):
#         if print_all_details:
#             print(f"Removing mention relation missing required fields: {relation}")
#         return False
    
#     subj_label = str(relation["subject_label"])
#     predicate = str(relation["predicate"])
#     obj_label = str(relation["object_label"])
    
#     if subj_label not in LEGAL_ENTITY_LABELS:
#         if print_all_details:
#             print(f"Removing mention relation with illegal subject label '{subj_label}': {relation}")
#         return False
    
#     if obj_label not in LEGAL_ENTITY_LABELS:
#         if print_all_details:
#             print(f"Removing mention relation with illegal object label '{obj_label}': {relation}")
#         return False
    
#     if predicate not in LEGAL_RELATION_LABELS:
#         if print_all_details:
#             print(f"Removing mention relation with illegal predicate '{predicate}': {relation}")
#         return False
    
#     return True

# def filter_article_data(article_data):
#     """Filter an article's data, keeping only valid entries."""
#     cleaned_data = {}
    
#     # Process entities
#     if TEST_6_1_NER and 'entities' in article_data:
#         cleaned_data['entities'] = [e for e in article_data['entities'] if is_valid_entity(e)]
    
#     # Process mention relations
#     mention_relations = []
#     if TEST_6_4_TERNARY_MENTION_RE and 'ternary_mention_based_relations' in article_data:
#         mention_relations = [r for r in article_data['ternary_mention_based_relations'] 
#                            if is_valid_ternary_mention_relation(r)]
#         cleaned_data['ternary_mention_based_relations'] = mention_relations
    
#     # Process tag relations
#     if TEST_6_3_TERNARY_TAG_RE:
#         tag_relations = []
        
#         # First try to use existing tag relations
#         if 'ternary_tag_based_relations' in article_data:
#             tag_relations = [r for r in article_data['ternary_tag_based_relations'] 
#                            if is_valid_ternary_tag_relation(r)]
            
#             if print_article_details and tag_relations:
#                 print(f"Kept {len(tag_relations)} valid tag relations from original")
        
#         # If no valid tag relations, try to extract from mentions
#         if not tag_relations and mention_relations:
#             tag_relations = [extract_tag_relation_from_mention(r) for r in mention_relations]
#             tag_relations = [r for r in tag_relations if is_valid_ternary_tag_relation(r)]
            
#             if print_article_details and tag_relations:
#                 print(f"Extracted {len(tag_relations)} tag relations from mention relations")
        
#         if tag_relations:
#             cleaned_data['ternary_tag_based_relations'] = tag_relations
    
#     return cleaned_data

# def process_and_validate_predictions(input_path, output_path):
#     """Main processing function that loads, validates and saves the data."""
#     try:
#         with open(input_path, 'r', encoding='utf-8') as file:
#             predictions = json.load(file)
#     except OSError:
#         raise OSError(f'Error opening input file: {input_path}')
    
#     # Process all articles
#     cleaned_predictions = {
#         pmid: filter_article_data(article_data)
#         for pmid, article_data in predictions.items()
#     }
    
#     # Calculate statistics
#     stats = {
#         'total_articles': len(cleaned_predictions),
#         'articles_with_entities': sum(1 for a in cleaned_predictions.values() if 'entities' in a),
#         'total_entities': sum(len(a.get('entities', [])) for a in cleaned_predictions.values()),
#         'articles_with_tag_relations': sum(1 for a in cleaned_predictions.values() if 'ternary_tag_based_relations' in a),
#         'total_tag_relations': sum(len(a.get('ternary_tag_based_relations', [])) for a in cleaned_predictions.values()),
#         'articles_with_mention_relations': sum(1 for a in cleaned_predictions.values() if 'ternary_mention_based_relations' in a),
#         'total_mention_relations': sum(len(a.get('ternary_mention_based_relations', [])) for a in cleaned_predictions.values()),
#         'articles_with_extracted_tag_relations': sum(
#             1 for a in cleaned_predictions.values() 
#             if 'ternary_tag_based_relations' in a and 
#             not any(r.get('subject_text_span', None) for r in a.get('ternary_tag_based_relations', []))
#         )
#     }
    
#     # Save cleaned data
#     try:
#         with open(output_path, 'w', encoding='utf-8') as file:
#             json.dump(cleaned_predictions, file, indent=2, ensure_ascii=False)
#     except OSError:
#         raise OSError(f'Error saving to output file: {output_path}')
    
#     # Print summary
#     print("\n===== Cleaning Summary =====")
#     print(f"Processed {stats['total_articles']} articles")
    
#     if TEST_6_1_NER:
#         print(f"\nEntities:")
#         print(f"  Articles with entities: {stats['articles_with_entities']}")
#         print(f"  Total entities kept: {stats['total_entities']}")
    
#     if TEST_6_3_TERNARY_TAG_RE:
#         print(f"\nTernary Tag Relations:")
#         print(f"  Articles with relations: {stats['articles_with_tag_relations']}")
#         print(f"  Total relations kept: {stats['total_tag_relations']}")
#         print(f"  Articles with relations extracted from mentions: {stats['articles_with_extracted_tag_relations']}")
    
#     if TEST_6_4_TERNARY_MENTION_RE:
#         print(f"\nTernary Mention Relations:")
#         print(f"  Articles with relations: {stats['articles_with_mention_relations']}")
#         print(f"  Total relations kept: {stats['total_mention_relations']}")
    
#     print("\nCleaned data saved to:", output_path)

# if __name__ == '__main__':
#     process_and_validate_predictions(INPUT_FILE, OUTPUT_FILE)


