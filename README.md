# Clustering Districts of India by Multidimensional Poverty Profile (Problem 41)

B.Sc. (H) Computer Science, Semester V, Ramanujan College, University of Delhi
Author: Shashikant (Roll No. 24020570019) | Supervisor: Mr Nikhil Rajput

Unsupervised learning (k-means, Ward, average/complete linkage, Gaussian mixture) on district-level
National MPI estimates (NITI Aayog, *National MPI: A Progress Review 2023*, NFHS-4 and NFHS-5).
Result: 631 districts fall into three stable tiers (High 109, Medium 266, Low 256).

## Files
| File | Purpose |
|---|---|
| `extract_data.py` | Extracts district tables from the NITI Aayog PDF into a CSV |
| `niti_mpi_districts_extracted.csv` | Extracted data (704 districts) |
| `analysis.py` | All clustering, validation and figures, writes `results/` |
| `build_paper.py` | Builds the journal-format PDF from `results/` |
| `Shashikant_MPI_Clustering_Paper.pdf` | Final paper |
| `results/` | Figures, cluster assignments, `results.json` |

## Run
```bash
pip install -r requirements.txt
# optional: re-extract from the PDF (needs poppler-utils for pdftotext)
python extract_data.py India-National-Multidimentional-Poverty-Index-2023.pdf
python analysis.py
python build_paper.py
```

## Data source
NITI Aayog, *India: National Multidimensional Poverty Index, A Progress Review 2023*:
https://www.niti.gov.in/sites/default/files/2023-08/India-National-Multidimentional-Poverty-Index-2023.pdf

## Limitation
The report publishes the 12 indicators only at State level, so clusters describe the level, depth and
change of poverty (headcount ratio and intensity), not indicator-specific deprivation.

## AI use
## AI use
Claude (Anthropic) was used to help write the data-extraction and analysis code and to draft the paper text. I ran the code, checked the extracted data against the NITI Aayog report, reviewed the results, and take full responsibility for the content.
