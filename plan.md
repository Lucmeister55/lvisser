# Plan

## Predictive Model Methylation

Use segmentation to reduce CpG dimensionality, no need to account for hemi-methylation or haplotype phasing (average probs across reads for every position)

### Annotation-Based

Can use DGE and other annotation for filtering/construction of segments. Use all samples for which there is DGE information.

#### Promotor Regions

#### Gene Bodies

### Annotation-agnostic

No use of DGE or other annotation.

#### Statistics-Based

Use WGBS_tools on one cell line, run DGE, and test on other cell line.

#### Density-Based

Use all samples from all cell lines for segmentation (DBSCAN + supervised subsegmentation), then run DMR on only training set.

## Statistical Model Methylation

### CpG DM

Use modkit for CpG level DM and link back to genes/papers.

### Haplotypes DM

Use modkit to split haplotagged by haplotype, and run DM to find allele specific methylation. Link ASM regions/CpGs back to genes/papers.

### CpG Island DMR

Use modkit to run DMR between R and S.

### Sensitized vs non-sensitized DM

Use modkit for DM between sensitized vs non-sensitized.

## DGE

Report differential gene expression results.