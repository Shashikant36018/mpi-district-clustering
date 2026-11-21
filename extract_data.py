"""Step 1: extract district MPI tables from the NITI Aayog PDF into niti_mpi_districts_extracted.csv.
Usage: python extract_data.py India-National-Multidimentional-Poverty-Index-2023.pdf   (needs poppler's pdftotext)"""
import re, subprocess, sys, pandas as pd
txt = subprocess.run(["pdftotext", "-layout", sys.argv[1], "-"], capture_output=True, text=True).stdout
P = r'(–|\d+\.\d+%)'; M = r'(–|\d\.\d{3})'
pat = re.compile(r'^\s*(.*?)\s{2,}' + r'\s+'.join([P, P, M, P, P, M]) + r'\s*$')
rows = []
for p in txt.split('\f'):
    m = re.search(r'^\s*([A-Za-z &,\.\-]+?):\s*Overview of District', p, re.M)
    if not m: continue
    st = m.group(1).strip(); prev = None
    for l in p.split('\n'):
        q = pat.match(l)
        if q:
            rows.append([st, q.group(1).strip() or prev or ''] + list(q.groups()[1:])); prev = None
        elif l.strip() and not re.search(r'\d|District$|Headcount|NFHS|MPI|Overview', l): prev = l.strip()
        elif l.strip() and not re.match(r'^\s*\(', l) and re.search(r'\d', l): prev = None
d = pd.DataFrame(rows, columns=['State', 'District', 'Headcount_NFHS4', 'Intensity_NFHS4', 'MPI_NFHS4', 'Headcount_NFHS5', 'Intensity_NFHS5', 'MPI_NFHS5'])
for c in d.columns[2:]: d[c] = pd.to_numeric(d[c].astype(str).str.replace('%', ''), errors='coerce')
d.loc[d.District == '', 'District'] = 'Kaimur (Bhabua)'   # name wrapped onto a separate line in the PDF (Bihar)
d.to_csv('niti_mpi_districts_extracted.csv', index=False); print(len(d), 'rows saved')
