# lasigeBioTM at BioASQ25 Task GutBrainIE - Lean Large language models with syntactic features

This repository contains the code used for our participation at the BioASQ2025 Task6 - GutBrainIE: Gut-Brain interplay Information Extraction @CLEF 2025

### [BENT annotation tool](https://github.com/lasigeBioTM/BENT)

## Systems
- Baseline: `src/raw_baseline.py`
- BENTMistral: `src/main_baseline_labels.py`
- BENTMistralSemantic: `src/spacy_relations.py`
- ConstParsing: `src/constituency_relations.py`

## Dataset Tagging System
Double Tag:
- External tag provides unique ID:
	- format: `'<eN> EntitySpanText </eN>'` → `'<e1> BrainBiota </e1>'`
- Inner tag provides entity type info:
	- format: `'@entityType$ EntitySpanText @/entityType$'` → `'@microbiome$ BrainBiota @/microbiome$'`

Full tag format: `'<eN>@entityType$ EntitySpanText @/entityType$</eN>'` → `'<e1>@microbiome$ BrainBiota @/microbiome$</e1>'`

#### Example 
*data/GutBrainIE_Full_Collection_2025/Annotations/Dev/json_format/dev.json*

**tagged_title**
```
'The association between <e1>@microbiome$ oral and gut microbiota @/microbiome$</e1> in <e2>@human$ male patients @/human$</e2> with <e3>@DDF$ alcohol dependence @/DDF$</e3>.'
```
**tagged_abstract**
```
'The relationship between <e4>@microbiome$ oral and gut microbiota @/microbiome$</e4> in <e5>@DDF$ alcohol dependence @/DDF$</e5> (<e6>@DDF$ AD @/DDF$</e6>) is not well understood, particularly the effects of <e7>@microbiome$ oral microbiota @/microbiome$</e7> on the <e8>@microbiome$ intestinal microbiota @/microbiome$</e8>. The current study aimed to explore the association between <e9>@microbiome$ oral and gut microbiota @/microbiome$</e9> in <e10>@DDF$ AD @/DDF$</e10> to clarify whether <e11>@microbiome$ oral microbiota @/microbiome$</e11> could ectopically colonize into the <e12>@anatomical location$ gut @/anatomical location$</e12>. 16S rRNA sequence libraries were used to compare oral and gut microbial profiles in <e13>@human$ persons @/human$</e13> with <e14>@DDF$ AD @/DDF$</e14> and <e15>@human$ healthy controls @/human$</e15> (<e16>@human$ HC @/human$</e16>). Source Tracker and NetShift were used to identify bacteria responsible for ectopic colonization and indicate the driver function of <e17>@bacteria$ ectopic colonization bacteria @/bacteria$</e17>. The α-diversity of <e18>@microbiome$ oral microbiota @/microbiome$</e18> and <e19>@microbiome$ intestinal microbiota @/microbiome$</e19> was significantly decreased in <e20>@human$ persons @/human$</e20> with <e21>@DDF$ AD @/DDF$</e21> (all <i>p</i> <\u20090.05). <e22>@statistical technique$ Principal coordinate analysis @/statistical technique$</e22> indicated greater similarity between <e23>@microbiome$ oral and gut microbiota @/microbiome$</e23> in <e24>@human$ persons @/human$</e24> with <e25>@DDF$ AD @/DDF$</e25> than that in <e26>@human$ HC @/human$</e26>, and oral-gut overlaps in microbiota were found for 9 genera in <e27>@human$ persons @/human$</e27> with <e28>@DDF$ AD @/DDF$</e28> relative to only 3 genera in <e29>@human$ HC @/human$</e29>. The contribution ratio of <e30>@microbiome$ oral microbiota @/microbiome$</e30> to intestinal microbiota composition in <e31>@DDF$ AD @/DDF$</e31> is 5.26% based on Source Tracker，and the <e32>@DDF$ AD @/DDF$</e32> with ectopic colonization showed the daily maximum standard drinks, red blood cell counts, hemoglobin content, and PACS scores decreasing (all <i>p</i> <\u20090.05). Results highlight the connection between <e33>@microbiome$ oral-gut microbiota @/microbiome$</e33> in <e34>@DDF$ AD @/DDF$</e34> and suggest novel potential mechanistic possibilities.'
```

# Project Structure
```
BioASQ25-GutBrainIE/
├── data/                       
│   ├── GutBrainIE_Full_Collection_2025/
│   │   └── Annotations/
│   └── ...                    # Contains datasets and annotation files used 
├── src/                        
│   ├── raw_baseline.py         # Baseline system implementation
│   ├── main_baseline_labels.py # BENTMistral system code
│   ├── spacy_relations.py      # BENTMistralSemantic system code
│   └── ...                    
├── utils/                      # Utility scripts and helper functions
│   └── ...                    
├── README.md                   
├── gutbrain.yml                # Dependencies

```

