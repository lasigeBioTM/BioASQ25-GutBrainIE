""" At the file: data/intermediate/Mistral-7B-Instruct-v0.3-NER-it-outputs-baseline.json
There is 7 entries where the 'llm_output' is an unparsed json string. 
The current code serves a baseline to parse these. """


import json

def parse_json_sequence(json_string_sequence: str) -> list[dict]:
    """
    Parses a string containing a sequence of JSON objects.

    Args:
        json_string_sequence (str): The string containing one or more
                                     JSON objects, typically separated by commas
                                     and whitespace.

    Returns:
        list: A list of Python dictionaries, where each dictionary
              represents a parsed JSON object. Returns an empty list
              if parsing fails or no objects are found.
    """
    parsed_objects = []
    decoder = json.JSONDecoder()
    # Remove any leading/trailing whitespace that might interfere
    text_to_parse = json_string_sequence.strip()
    current_pos = 0

    while current_pos < len(text_to_parse):
        try:
            # Find the start of the next JSON object (the next '{')
            # This helps skip over separators like ",\n    "
            obj_start_index = text_to_parse.index('{', current_pos)

            # Use raw_decode to parse one JSON object from the current position
            # It returns the Python object and the index where parsing stopped
            obj, end_pos = decoder.raw_decode(text_to_parse, obj_start_index)
            parsed_objects.append(obj)
            current_pos = end_pos # Update position to continue after the parsed object

        except ValueError:
            # This occurs if text_to_parse.index('{', current_pos) doesn't find a '{'.
            # It means no more objects are found, or the remaining string isn't an object.
            # print(f"No more JSON objects found or invalid sequence starting at position {current_pos}.")
            break
        except json.JSONDecodeError as e:
            print(f"JSON decoding error at/after position {obj_start_index}: {e}")
 
            next_comma_idx = text_to_parse.find(',', current_pos)
            if next_comma_idx != -1:
                current_pos = next_comma_idx + 1
            else:
                # Cannot find a comma, likely end of string or severe malformation
                break
        except Exception as e:
            # Catch any other unexpected errors during parsing
            print(f"An unexpected error occurred: {e}")
            break
            
    return parsed_objects

def test(strings: list[str]):
    """
    Test the JSON parsing function with a list of strings.
    """
    for i, string in enumerate(strings):
        print(f"Test {i+1}:")
        parsed_objects = parse_json_sequence(string)
        if parsed_objects:
            print(f"Parsed {len(parsed_objects)} objects:")
            for obj in parsed_objects:
                print(obj)
        else:
            print("No valid JSON objects found.")
        print("\n")


if __name__ == "__main__":

    test_strings = ["{\n    \"start_idx\": 57,\n    \"end_idx\": 83,\n    \"location\": \"title\",\n    \"text_span\": \"CNS autoimmune inflammation\",\n    \"label\": \"DDF\"\n    },\n    {\n    \"start_idx\": 11,\n    \"end_idx\": 33,\n    \"location\": \"abstract\",\n    \"text_span\": \"Depression and anxiety disorders\",\n    \"label\": \"DDF\"\n    },\n    {\n    \"start_idx\": 3,\n    \"end_idx\": 10,\n    \"location\": \"abstract\",\n    \"text_span\": \"C57BL/6J mice\",\n    \"label\": \"Animal\"\n    },\n    {\n    \"start_idx\": 15,\n    \"end_idx\": 20,\n    \"location\": \"abstract\",\n    \"text_span\": \"VD insufficiency\",\n    \"label\": \"DDF\"\n    },\n    {\n    \"start_idx\": 23,\n    \"end_idx\": 30,\n    \"location\": \"abstract\",\n    \"text_span\": \"gut microbiota\",\n    \"label\": \"Microbiome\"\n    },\n    {\n    \"start_idx\": 49,\n    \"end_idx\": 56,\n    \"location\": \"abstract\",\n    \"text_span\": \"VD status\",\n    \"label\": \"DDF\"\n    },\n    {\n    \"start_idx\": 64,\n    \"end_idx\": 71,\n    \"location\": \"abstract\",\n    \"text_span\": \"anxiety-related behavior\",\n    \"label\": \"DDF\"\n    },\n    {\n    \"start_idx\": 74,\n    \"end_idx\": 83,\n    \"location\": \"abstract\",\n    \"text_span\": \"CNS autoimmune inflammation\",\n    \"label\": \"DDF\"\n    },\n    {\n    \"start_idx\": 102,\n    \"end_idx\": 109,\n    \"location\": \"abstract\",\n    \"text_span\": \"VD receptor\",\n    \"label\": \"Biomarker\"\n    },\n    {\n    \"start_idx\": 112,\n    \"end_idx\": 120,\n    \"location\": \"abstract\",\n    \"text_span\": \"TPH1\",\n    \"label\": \"Gene\"\n    }\n    }", 
                    "{\n    \"start_idx\": 57,\n    \"end_idx\": 83,\n    \"location\": \"title\",\n    \"text_span\": \"CNS autoimmune inflammation\",\n    \"label\": \"DDF\"\n    },\n    {\n    \"start_idx\": 116,\n    \"end_idx\": 132,\n    \"location\": \"abstract\",\n    \"text_span\": \"Parkinson's disease\",\n    \"label\": \"DDF\"\n    },\n    {\n    \"start_idx\": 136,\n    \"end_idx\": 145,\n    \"location\": \"abstract\",\n    \"text_span\": \"substantia nigra\",\n    \"label\": \"Anatomical Location\"\n    },\n    {\n    \"start_idx\": 152,\n    \"end_idx\": 160,\n    \"location\": \"abstract\",\n    \"text_span\": \"dopaminergic cell loss\",\n    \"label\": \"DDF\"\n    },\n    {\n    \"start_idx\": 167,\n    \"end_idx\": 179,\n    \"location\": \"abstract\",\n    \"text_span\": \"HFE protein\",\n    \"label\": \"Gene\"\n    },\n    {\n    \"start_idx\": 182,\n    \"end_idx\": 190,\n    \"location\": \"abstract\",\n    \"text_span\": \"H63D variant\",\n    \"label\": \"Gene\"\n    },\n    {\n    \"start_idx\": 200,\n    \"end_idx\": 208,\n    \"location\": \"abstract\",\n    \"text_span\": \"H67D gene variant\",\n    \"label\": \"Gene\"\n    },\n    {\n    \"start_idx\": 212,\n    \"end_idx\": 220,\n    \"location\": \"abstract\",\n    \"text_span\": \"paraquat toxicity\",\n    \"label\": \"Biomedical Technique\"\n    },\n    {\n    \"start_idx\": 224,\n    \"end_idx\": 232,\n    \"location\": \"abstract\",\n    \"text_span\": \"substantia nigra\",\n    \"label\": \"Anatomical Location\"\n    },\n    {\n    \"start_idx\": 239,\n    \"end_idx\": 247,\n    \"location\": \"abstract\",\n    \"text_span\": \"R<sub>2</sub> relaxation rate\",\n    \"label\": \"Statistical Technique\"\n    },\n    {\n    \"start_idx\": 253,\n    \"end_idx\": 261,\n    \"location\": \"abstract\",\n    \"text_span\": \"gut microbiome profile\",\n    \"label\": \"Microbiome\"\n    },\n    {\n    \"start_idx\": 265,\n    \"end_idx\": 273,\n    \"location\": \"abstract\",\n    \"text_span\": \"L-ferritin staining\",\n    \"label\": \"Chemical\"\n    },\n    {\n    \"start_idx\": 277,\n    \"end_idx\": 285,\n    \"location\": \"abstract\",\n    \"text_span\": \"tyrosine hydroxylase staining\",\n    \"label\": \"Biochemical\"\n    },\n    {\n    \"start_idx\": 294,\n    \"end_idx\": 302,\n    \"location\": \"abstract\",\n    \"text_span\": \"paraquat-treated mice\",\n    \"label\": \"Animal\"\n    },\n    {\n    \"start_idx\": 305,\n    \"end_idx\": 313,\n    \"location\": \"abstract\",\n    \"text_span\": \"saline-treated counterparts\",\n    \"label\": \"Animal\"\n    },\n    {\n    \"start_idx\": 316,\n    \"end_idx\": 324,\n    \"location\": \"abstract\",\n    \"text_span\": \"H67D HFE mice\",\n    \"label\": \"Animal\"\n    },\n    {\n    \"start_idx\": 327,\n    \"end_idx\": 335,\n    \"location\": \"abstract\",\n    \"text_span\": \"WT mice\",\n    \"label\": \"Animal\"\n    }\n    }",
                    "{\n    \"start_idx\": 57,\n    \"end_idx\": 83,\n    \"location\": \"title\",\n    \"text_span\": \"CNS autoimmune inflammation\",\n    \"label\": \"DDF\"\n    },\n    {\n    \"start_idx\": 106,\n    \"end_idx\": 122,\n    \"location\": \"abstract\",\n    \"text_span\": \"inflammatory bowel disease\",\n    \"label\": \"DDF\"\n    },\n    {\n    \"start_idx\": 123,\n    \"end_idx\": 132,\n    \"location\": \"abstract\",\n    \"text_span\": \"irritable bowel syndrome\",\n    \"label\": \"DDF\"\n    },\n    {\n    \"start_idx\": 143,\n    \"end_idx\": 150,\n    \"location\": \"abstract\",\n    \"text_span\": \"Parkinson's disease\",\n    \"label\": \"DDF\"\n    },\n    {\n    \"start_idx\": 151,\n    \"end_idx\": 159,\n    \"location\": \"abstract\",\n    \"text_span\": \"Alzheimer's disease\",\n    \"label\": \"DDF\"\n    },\n    {\n    \"start_idx\": 160,\n    \"end_idx\": 169,\n    \"location\": \"abstract\",\n    \"text_span\": \"autism spectrum disorder\",\n    \"label\": \"DDF\"\n    }\n    }",
                    "{\n    \"start_idx\": 57,\n    \"end_idx\": 83,\n    \"location\": \"title\",\n    \"text_span\": \"CNS autoimmune inflammation\",\n    \"label\": \"DDF\"\n    },\n    {\n    \"start_idx\": 11,\n    \"end_idx\": 27,\n    \"location\": \"title\",\n    \"text_span\": \"Parkinson's disease\",\n    \"label\": \"DDF\"\n    },\n    {\n    \"start_idx\": 61,\n    \"end_idx\": 69,\n    \"location\": \"abstract\",\n    \"text_span\": \"gut microbiome\",\n    \"label\": \"Microbiome\"\n    },\n    {\n    \"start_idx\": 71,\n    \"end_idx\": 77,\n    \"location\": \"abstract\",\n    \"text_span\": \"SNCA gene\",\n    \"label\": \"Gene\"\n    },\n    {\n    \"start_idx\": 79,\n    \"end_idx\": 83,\n    \"location\": \"abstract\",\n    \"text_span\": \"PD\",\n    \"label\": \"DDF\"\n    },\n    {\n    \"start_idx\": 90,\n    \"end_idx\": 96,\n    \"location\": \"abstract\",\n    \"text_span\": \"16S rRNA DNA sequencing\",\n    \"label\": \"Biomedical Technique\"\n    },\n    {\n    \"start_idx\": 100,\n    \"end_idx\": 109,\n    \"location\": \"abstract\",\n    \"text_span\": \"fecal inflammatory calprotectin\",\n    \"label\": \"DDF\"\n    },\n    {\n    \"start_idx\": 114,\n    \"end_idx\": 120,\n    \"location\": \"abstract\",\n    \"text_span\": \"Lactobacillus sp.\",\n    \"label\": \"Bacteria\"\n    },\n    {\n    \"start_idx\": 128,\n    \"end_idx\": 134,\n    \"location\": \"abstract\",\n    \"text_span\": \"enriched environment\",\n    \"label\": \"Environmental Condition\"\n    },\n    {\n    \"start_idx\": 142,\n    \"end_idx\": 148,\n    \"location\": \"abstract\",\n    \"text_span\": \"pro-inflammatory cytokines\",\n    \"label\": \"DDF\"\n    },\n    {\n    \"start_idx\": 152,\n    \"end_idx\": 158,\n    \"location\": \"abstract\",\n    \"text_span\": \"inflammation inducing genes\",\n    \"label\": \"DDF\"\n    }\n    }"]
    test(test_strings)