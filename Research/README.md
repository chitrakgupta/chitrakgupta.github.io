
# Molecular simulation of pH (Low) Insertion Peptide pHLIP

## The bigger picture

A major challenge in cancer treatment is off-target side effects, where the cancer drug attacks the healthy cells in addition to the cancerous ones. Distinguising cancerous tumors is made even harder by the heterogeneity of these tumors and rapid mutations that lead to drug resistance. A potential solution is to exploit the properties of tumor microenvironment that is universal of all cancer cells. One such property is the extracellular pH,  which is lower in tumor cells (~ 6.8) than healthy tissues (~ 7.2).

<img src="./Fig1a_cropped.png" align="left" height="200" width="200" style="display:inline;margin-right:20px; margin-bottom:20px;"/>

A peptide derived from the helix C of bacteriorhodopsin (see figure on left) has exactly this property, and has been named pH (Low) Insertion Peptide, pHLIP. This peptide is soluble and unstructured in solution (State I, see figure below) and binds to surface of lipid membranes under neutral and alkaline pH (State II, see figure below). However, once the pH drops to acidic level, pHLIP spontaneously forms a transmembrane helix. This property makes pHLIP a great candidate for cancer diagnostic and therapeutic. However, pHLIP insertion happens at a pH ( ~ 6.2) that is much lower than the pH of most cancerous tumors. Thus, modifications to the pHLIP sequence are necessary to develop pHLIP into a novel cancer-targeting agent. Such modifications require a detailed understanding of the properties of pHLIP at a molecular level. This is the focus of my PhD research.

<img src="./Fig1b.png" align="left" height="300" width="600" style="display:inline;margin-right:20px; margin-bottom:20px;"/>

## State I of pHLIP

While state I of pHLIP has mostly been ignored, I took a second look. This was due to two factors:

    (i) While conventional wisdom said pHLIP was unstructured in state II, some results showed otherwise
    (ii) Theories of peptide folding predict a equilibrium between completely coiled and partially folded states of a peptide that has a propensity to form a secondary structure.
    
    This rasied the question: Is there partial secondary structure in State I?
          
I am answering these questions by performing long timescale (~ 2 microsecond) simulation of pHLIP with selective acidic residues titrated, along with three unnatural mutants of pHLIP that were shown to have altered pH of membrane insertion. Such long timescales become feasible by using GPU-accelerated Amber simulation engine in combination with implicit solvation, which bypasses the need to explicitly define the solvent atoms thereby significantly reducing the computation time. This study has elucidated the influence of site-specific titration and mutation on the helix-forming propensity of pHLIP. Moreover, we have discovered transient sampling of the ramachandran space of pHLIP that was not known heretofore.

While this is a good starting point, I eventually want to understand the interactions of pHLIP with water molecule. I am achieving this by the use of gaussian accelerated molecular dynamics ([GaMD](http://gamd.ucsd.edu/)) 

## State II of pHLIP

Most of the magic really begins in state II. Unfortunately, very little is known about conformation of pHLIP in state II and its orientation with respect to the bilayer normal. I have performed long timescale ( ~ 800 ns) simulation of pHLIP placed on a model lipid membrane at various orientations. This preliminary study has shown desolvation of pHLIP, burial of one arginine and two tryptophan residues into the lipid bilayer, each of which is consistent with experimental results. Additionally, this study has revealed the favorable binding orientation of pHLIP.

### Effect of bilayer electrostatics

Two more factors have been mostly overlooked until now. One is the influence of anionic lipid headgroups on pHLIP-binding. Most cancerous tumors have higher levels of POPS lipid in the outer leaflet, which can be expected to repel the negatively charged pHLIP. Some recent experiments have began shedding some light on this issue. I have performed replica exchange with solute tempering (REST2) simulation of pHLIP-lipid system to gain a fundamental understanding of the interactions between pHLIP and the lipid bilayer. In my study, I systematically varied the POPS content of the lipid membrane, which helped explain some of the discrepancies of the experimental results. I have used [XSEDE](https://www.xsede.org/)'s resources for this study.

### Effect of salt concentration

While it is known that pHLIP aggregates under higher salt concentration, little has been known until now about the influence of the same on lipid-binding. Our [collaborators](https://barreralab.com/) are leading the way in discovering salt concentration effect. To gain a deeper, molecular-level understanding, I am performing gaussian accelerated molecular dynamics ([GaMD](http://gamd.ucsd.edu/)) simulation of pHLIP-lipid system with varying ionic strength.


```python

```