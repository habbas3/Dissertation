#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 14:40:39 2023

@author: habbas3
"""
import pdfplumber
import pandas as pd
import os

pdf_path = 'Manufacturer_Communications.pdf'
csv_temp_path = 'temp_data_manufacturerComm.csv'
last_page_processed_path = 'last_page_processed_MC.txt'
final_excel_path = 'Manufacturer_Communications_NHTSA.xlsx'

# Check if last_page_processed.txt exists to determine starting page
if os.path.exists(last_page_processed_path):
    with open(last_page_processed_path, 'r') as f:
        start_page = int(f.read().strip()) + 1
else:
    start_page = 1

    # Initialize an empty CSV to store the data if it doesn't exist
    if not os.path.exists(csv_temp_path):
        with open(csv_temp_path, 'w') as f:
            f.write(','.join(['NHTSA ID', 'DOCUMENT NAME', 'MAKE', 'MODEL', 'MODEL YEAR', 'SUMMARY']) + '\n')

with pdfplumber.open(pdf_path) as pdf:
    total_pages = len(pdf.pages)
    
    for page_num in range(start_page, total_pages):
        # Print progress
        print(f"Processing page {page_num} out of {total_pages - 1}")
        
        page = pdf.pages[page_num]
        tables = page.extract_tables()

        # Write data to the temporary CSV
        with open(csv_temp_path, 'a') as f:
            for table in tables:
                for row in table[1:]:
                    f.write(','.join(['"' + cell.replace('"', '""') + '"' for cell in row]) + '\n')

        # Update the last processed page
        with open(last_page_processed_path, 'w') as f:
            f.write(str(page_num))

# Once all data is extracted, load the CSV and save as Excel
data_df = pd.read_csv(csv_temp_path)
data_df.to_excel(final_excel_path, index=False, engine='openpyxl')

# Optionally, delete the temporary CSV and the last page processed file
os.remove(csv_temp_path)
os.remove(last_page_processed_path)
print("All Done!")


