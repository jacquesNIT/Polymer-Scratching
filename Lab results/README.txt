Files Structure:

<material>_<maximum_load>_<load_rate>_<sample_number>_<try_number>

- sample_number (s1, s2, ...) is used to indicate another iteration of the same test.
- try_number (1, 2, ...) simply indicates that precedent tests failed, it does not matter.

Scratch Process:

All scratchs were performed using the Rockwell indenter with the "Scratch.Rx" model in MFT. 
All tests have at a length of 2mm and a progressive load.

The same results are also accessible on the scratch pc of the lab, in the "jacques/second_batch" folder.
The "Additionnal_Batch" folder contains unverified results that were used to test the machine, it is advised not to use those.

General Notes:

- The CAP depth values are very irregular and do not make sense most of the time. It is advised not to trust those.
- Some scratchs with higher forces of typically 15 or 20N are not fully visible in the images as widths are too large, this can sometimes lead to issues when calculating topographic values. 
- Some tests files are labeled with "25Nm" as a load rate, it indicates 2.5N/min, not 25.
- Some folders and .bcrf files are labelled with "t2", it indicates that the imaging was performed at least a day after the scratch instead of directly after. 
- Analysis files (.png) for PETG, PMMAGS and PP are not to be trusted, as smoothing / extraction is fitted to the PMMAXT and PC.

- Despite the transparency of some of these materials, finding the right focus for imaging is possible.
- Imaging was performed as soon as possible after the scratch, usually in a window of less than 30 minutes.
- As samples can be slightly inclined, non-negligible differences in values for right and left Pile-up may be observed.
- Scratchs are separated by at least 2 / 3 scratch width. It is possible that this value is not enough. 

Material references:

PC - Extruded polycarbonate / Makrolon, Exolon Group
PMMA-XT - PLEXIGLAS — extruded acrylic ISO 7823-2, Röhm / POLYVANTIS
PETG - VERALITE® 200 — i.P.B., IPB nv, Waregem (BE)
PMMA-GS - Plexiglas GS WH…D, Röhm / POLYVANTIS
PP - Unknown 

Material Cards: (Manufacturer's values)

Property                          PC               PMMA XT          PMMA GS          PETG 

Density (g/cm3)                   1.20             1.19             1.19             1.27
Tensile modulus E (MPa, ISO 527)  2350             3300             3300             approx. 2200
Tensile stress (MPa)              > 60 (yield)     72               80               51.5
Compressive stress (MPa, ISO 604) -                103              110              -
Poisson's ratio                   -                0.37             0.37             -
Dynamic shear modulus G (MPa)     -                1700             1700             -
Elongation at break (%)           > 50 (nominal)   4.5              5.5              > 100
Indentation hardness H961/30 (MPa)-                175              175              -
Vicat B50 (deg C)                 148              103              115              -
