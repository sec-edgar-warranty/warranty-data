# warranty-data
This repository contains warranty data from SEC EDGAR. There are two data sets in this repository.

## Data Set 1

This data set contains warranty tables extracted from various filings (mostly 10-K and 10-Q) from SEC's EDGAR system. The file `warranty_data_transposed.csv` has nine matadata and 33 warranty related columns. We will add more details here soon.

## Data Set 2

This data set is created by scraping Form 10-Qs from 2007 and 2008. Companies were not required to use XBRL before 2009 so this is the only way to extract the data. We share the Python code to achieve this task. The Python code is extensive and contains multiple functions. We strongly advise users to read the code carefully before using it.

This data set has eight warranty tables shared as Excel files. These tables contain accession number, which is the key used to match with the metadata files to get company names, CIK, SIC codes, etc.
