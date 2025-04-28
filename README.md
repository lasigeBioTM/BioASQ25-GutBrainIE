# BioASQ25-GutBrainIE
BioASQ2025 Task6 -GutBrainIE: Gut-Brain interplay Information Extraction @CLEF 2025



### Dataset tagging System:
Double Tag :
- External tag provides unique ID:
	- format - `'<eN> EntitySpanText </eN>'` -> `'<e1> BrainBiota </e1>'`
- Inner tag provides entity type info:
	- format - `'@entityType$ EntitySpanText @/entityType$'` -> `'@microbiome$ BrainBiota @/microbiome$'`

Full tag format - `'<eN>@entityType$ EntitySpanText @/entityType$</eN>'`  -> `'<e1>@microbiome$ BrainBiota @/microbiome$</e1>'`



#### Example 
*GutBrainIE_Full_Collection_2025/Annotations/Dev/json_format/dev.json*
**tagged_title**
```
'The association between <e1>@microbiome$ oral and gut microbiota @/microbiome$</e1> in <e2>@human$ male patients @/human$</e2> with <e3>@DDF$ alcohol dependence @/DDF$</e3>.'
```
 **tagged_abstract**

```
'The relationship between <e4>@microbiome$ oral and gut microbiota @/microbiome$</e4> in <e5>@DDF$ alcohol dependence @/DDF$</e5> (<e6>@DDF$ AD @/DDF$</e6>) is not well understood, particularly the effects of <e7>@microbiome$ oral microbiota @/microbiome$</e7> on the <e8>@microbiome$ intestinal microbiota @/microbiome$</e8>. The current study aimed to explore the association between <e9>@microbiome$ oral and gut microbiota @/microbiome$</e9> in <e10>@DDF$ AD @/DDF$</e10> to clarify whether <e11>@microbiome$ oral microbiota @/microbiome$</e11> could ectopically colonize into the <e12>@anatomical location$ gut @/anatomical location$</e12>. 16S rRNA sequence libraries were used to compare oral and gut microbial profiles in <e13>@human$ persons @/human$</e13> with <e14>@DDF$ AD @/DDF$</e14> and <e15>@human$ healthy controls @/human$</e15> (<e16>@human$ HC @/human$</e16>). Source Tracker and NetShift were used to identify bacteria responsible for ectopic colonization and indicate the driver function of <e17>@bacteria$ ectopic colonization bacteria @/bacteria$</e17>. The α-diversity of <e18>@microbiome$ oral microbiota @/microbiome$</e18> and <e19>@microbiome$ intestinal microbiota @/microbiome$</e19> was significantly decreased in <e20>@human$ persons @/human$</e20> with <e21>@DDF$ AD @/DDF$</e21> (all <i>p</i> <\u20090.05). <e22>@statistical technique$ Principal coordinate analysis @/statistical technique$</e22> indicated greater similarity between <e23>@microbiome$ oral and gut microbiota @/microbiome$</e23> in <e24>@human$ persons @/human$</e24> with <e25>@DDF$ AD @/DDF$</e25> than that in <e26>@human$ HC @/human$</e26>, and oral-gut overlaps in microbiota were found for 9 genera in <e27>@human$ persons @/human$</e27> with <e28>@DDF$ AD @/DDF$</e28> relative to only 3 genera in <e29>@human$ HC @/human$</e29>. The contribution ratio of <e30>@microbiome$ oral microbiota @/microbiome$</e30> to intestinal microbiota composition in <e31>@DDF$ AD @/DDF$</e31> is 5.26% based on Source Tracker，and the <e32>@DDF$ AD @/DDF$</e32> with ectopic colonization showed the daily maximum standard drinks, red blood cell counts, hemoglobin content, and PACS scores decreasing (all <i>p</i> <\u20090.05). Results highlight the connection between <e33>@microbiome$ oral-gut microbiota @/microbiome$</e33> in <e34>@DDF$ AD @/DDF$</e34> and suggest novel potential mechanistic possibilities.'
```

**Relations HyperGraph:**
```
{<e1> oral and gut microbiota </e1>} ⟶[located in] <e2> male patients </e2>
{<e6> AD </e6>, <e3> alcohol dependence </e3>} ⟶[target] <e2> male patients </e2>
{<e4> oral and gut microbiota </e4>} ⟶[is linked to] <e5> alcohol dependence </e5>
{<e4> oral and gut microbiota </e4>} ⟶[is linked to] <e6> AD </e6>
{<e7> oral microbiota </e7>} ⟶[compared to] <e8> intestinal microbiota </e8>
{<e9> oral and gut microbiota </e9>} ⟶[is linked to] <e10> AD </e10>
{<e9> oral and gut microbiota </e9>} ⟶[located in] <e13> persons </e13>
{<e9> oral and gut microbiota </e9>} ⟶[located in] <e15> healthy controls </e15>
{<e9> oral and gut microbiota </e9>} ⟶[located in] <e16> HC </e16>
{<e14> AD </e14>} ⟶[target] <e13> persons </e13>
{<e18> oral microbiota </e18>, <e19> intestinal microbiota </e19>} ⟶[located in] <e20> persons </e20>
{<e21> AD </e21>} ⟶[target] <e20> persons </e20>
{<e23> oral and gut microbiota </e23>} ⟶[located in] <e24> persons </e24>
{<e23> oral and gut microbiota </e23>} ⟶[located in] <e26> HC </e26>
{<e25> AD </e25>} ⟶[target] <e24> persons </e24>
{<e31> AD </e31>, <e28> AD </e28>} ⟶[target] <e27> persons </e27>
{<e30> oral microbiota </e30>} ⟶[located in] <e27> persons </e27>
{<e30> oral microbiota </e30>} ⟶[is linked to] <e32> AD </e32>
{<e33> oral-gut microbiota </e33>} ⟶[is linked to] <e34> AD </e34>
```
