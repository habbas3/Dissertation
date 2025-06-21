#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 14:07:22 2023

@author: habbas3
"""

# Load the data files
#Load the recalls file
import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer
# import spacy
# from gensim.models import Word2Vec
import torch
# from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset
# from tqdm.auto import tqdm
import time
from textblob import TextBlob
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
import re
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from transformers import DistilBertTokenizer
import numpy as np
import pickle
import gc 



#Load the saved safety ratings data
df_SafetyRatings = pd.read_excel("nhtsa_vehicle_SafetyRatingsData.xlsx", engine='openpyxl')
df_SafetyRatings.columns = df_SafetyRatings.columns.str.upper()
df_SafetyRatings = df_SafetyRatings.rename(columns={"MODELYEAR": "YEAR"})
df_SafetyRatings['YEAR'] = df_SafetyRatings['YEAR'].astype(str)




# Display the first few rows of the loaded dataframe
print(df_SafetyRatings.head())

# Define column names based on description txt file
column_names = [
    "RECORD_ID", "CAMPNO", "MAKE", "MODEL", "YEAR", "MFGCAMPNO",
    "COMPNAME", "MFGNAME", "BGMAN", "ENDMAN", "RCLTYPECD", "POTAFF",
    "ODATE", "INFLUENCED_BY", "MFGTXT", "RCDATE", "DATEA", "RPNO",
    "FMVSS", "DESC_DEFECT", "CONEQUENCE_DEFECT", "CORRECTIVE_ACTION",
    "NOTES", "RCL_CMPT_ID", "MFR_COMP_NAME", "MFR_COMP_DESC", "MFR_COMP_PTNO"
]

# Read the txt file using pandas
file_path = 'FLAT_RCL 2.txt'
df_flat_rcl = pd.read_csv(file_path, delimiter="\t", names=column_names, dtype=str, on_bad_lines='skip')

#Date columns to datetime format
date_columns = ['BGMAN','ENDMAN','ODATE', 'RCDATE', 'DATEA']
for col in date_columns:
    df_flat_rcl[col] = pd.to_datetime(df_flat_rcl[col], format='%Y%m%d', errors='coerce').dt.date

# Display the first few rows to check the data
print(df_flat_rcl.head())

#Load the quarterly recalls file
# Define the file path
file_path = 'FLAT_RCL_Qtrly_Rpts.txt'

# Define column names
columns = [
    'MFGTXT', 'CAMPNO', 'MFGCAMPNO', 'RCLSUBJ', 'ODATE', 
    'ODATEEND', 'RPTNO', 'RPTQTR', 'INVOLVED', 'TTLREMEDIED',
    'TTLUNREACH', 'TTLREMOVED', 'SUBMDATE'
]

# Read the data
df_qtrly_rpts = pd.read_csv(file_path, sep='\t', header=None, names=columns,on_bad_lines='skip')

# To display the first few rows of the dataframe
print(df_qtrly_rpts.head())

#Date columns to datetime format
date_columns = ['ODATE', 'ODATEEND', 'SUBMDATE']
for col in date_columns:
    df_qtrly_rpts[col] = pd.to_datetime(df_qtrly_rpts[col], format='%Y%m%d', errors='coerce').dt.date

#Load the annual recalls file
# Define column names based on the provided description
column_names = [
    "MFGTXT", "CAMPNO", "MFGCAMPNO", "RCLSUBJ", "ODATE", "ODATEEND", 
    "RPTNO", "REPORT_YEAR", "INVOLVED", "TTLREMEDIED", 
    "TTLUNREACH", "TTLREMOVED", "SUBMDATE"
]

# Read the tab-delimited file
df_annual_rpts = pd.read_csv('FLAT_RCL_Annual_Rpts.txt', delimiter='\t', names=column_names)

# Parse dates
date_columns = ["ODATE", "ODATEEND", "SUBMDATE"]
for date_col in date_columns:
    df_annual_rpts[date_col] = pd.to_datetime(df_annual_rpts[date_col], format='%Y%m%d', errors='coerce').dt.date

# Display the first few rows of the dataframe for verification
print(df_annual_rpts.head())


#Load in the Recall_Communications file
df_RecallCommunication = pd.read_excel("Recalls_Communication_NHTSA.xlsx", engine='openpyxl')

# Renaming 'MODEL YEAR' to 'YEAR' in df_RecallCommunication
df_RecallCommunication = df_RecallCommunication.rename(columns={"MODEL YEAR": "YEAR"})
print(df_RecallCommunication.head())



#Load the investigation data
# Define the column names
cols = ['NHTSA ACTION NUMBER', 'MAKE', 'MODEL', 'YEAR', 'COMPNAME', 
        'MFR_NAME', 'ODATE', 'CDATE', 'CAMPNO', 'SUBJECT', 'SUMMARY']

# Read the file
df_investigation = pd.read_csv('FLAT_INV.txt', delimiter='\t', header=None, names=cols, encoding='utf-8', engine='python')

df_investigation['ODATE'] = pd.to_datetime(df_investigation['ODATE'], format='%Y%m%d', errors='coerce').dt.date
df_investigation['CDATE'] = pd.to_datetime(df_investigation['CDATE'], format='%Y%m%d', errors='coerce').dt.date
df_investigation['YEAR'] = df_investigation['YEAR'].astype(str)

# Display the data
print(df_investigation.head())

#Load the complaints data
file_path = 'FLAT_CMPL.txt'

# Define column names
columns = [
    "CMPLID", "ODINO", "MFR_NAME", "MAKE", "MODEL", "YEAR", "CRASH",
    "FAILDATE", "FIRE", "INJURED", "DEATHS", "COMPDESC", "CITY", "STATE",
    "VIN", "DATEA", "LDATE", "MILES", "OCCURENCES", "CDESCR", "CMPL_TYPE",
    "POLICE_RPT_YN", "PURCH_DT", "ORIG_OWNER_YN", "ANTI_BRAKES_YN", "CRUISE_CONT_YN",
    "NUM_CYLS", "DRIVE_TRAIN", "FUEL_SYS", "FUEL_TYPE", "TRANS_TYPE", "VEH_SPEED",
    "DOT", "TIRE_SIZE", "LOC_OF_TIRE", "TIRE_FAIL_TYPE", "ORIG_EQUIP_YN", "MANUF_DT",
    "SEAT_TYPE", "RESTRAINT_TYPE", "DEALER_NAME", "DEALER_TEL", "DEALER_CITY",
    "DEALER_STATE", "DEALER_ZIP", "PROD_TYPE", "REPAIRED_YN", "MEDICAL_ATTN", "VEHICLES_TOWED_YN"
]

# Read the data
df_complaints = pd.read_csv(file_path, delimiter='\t', header=None, names=columns, dtype=str,on_bad_lines='skip')


#Load Manufacturer Communications
columns = [
    "BULNO",
    "BULREP",
    "ID",
    "BULDTE",
    "COMPNAME",
    "MAKE",
    "MODEL",
    "YEAR",
    "DATEA",
    "SUMMARY"
]

# Read the file
df_ManufacturerCommunications = pd.read_csv('flat_tsbs.txt', delimiter='\t', header=None, names=columns,on_bad_lines='skip')
date_columns = ["BULDTE","DATEA"]
for date_col in date_columns:
    df_ManufacturerCommunications[date_col] = pd.to_datetime(df_ManufacturerCommunications[date_col], format='%Y%m%d', errors='coerce').dt.date

# Display the first few rows of the dataframe
print(df_ManufacturerCommunications.head())

#Load Manufacturer Communications PDF(originally) File
df_Manufacturer_Communications_pdf = pd.read_excel("Manufacturer_Communications_NHTSA.xlsx", engine='openpyxl')
df_Manufacturer_Communications_pdf = df_Manufacturer_Communications_pdf.rename(columns={"MODEL YEAR": "YEAR"})
# Display the first few rows of the loaded dataframe
print(df_Manufacturer_Communications_pdf.head())


#Merge the dataframes that do not have make, model,year but do have campno with the ones that have all 4 in order to map it correctly and have make,model,year in all dataframes
def merge_based_on_campno(df_target, df_source1, df_source2, key_column, columns_to_add):
    # Merging with the first source dataframe and dropping duplicates
    merged_df1 = pd.merge(df_target, df_source1[[key_column] + columns_to_add], on=key_column, how='left').drop_duplicates(subset=[key_column])

    # Merging with the second source dataframe and dropping duplicates
    merged_df2 = pd.merge(df_target, df_source2[[key_column] + columns_to_add], on=key_column, how='left').drop_duplicates(subset=[key_column])

    # Combine the results and keep only the first match for each CAMPNO
    combined_df = pd.concat([merged_df1, merged_df2]).drop_duplicates(subset=[key_column], keep='first')

    # Drop rows where MAKE, MODEL, and YEAR are NaN (i.e., no match found)
    combined_df.dropna(subset=columns_to_add, inplace=True)

    return combined_df

columns_to_add = ['MAKE', 'MODEL', 'YEAR']
df_qtrly_rpts = merge_based_on_campno(df_qtrly_rpts, df_flat_rcl, df_investigation, 'CAMPNO', columns_to_add)
df_annual_rpts = merge_based_on_campno(df_annual_rpts, df_flat_rcl, df_investigation, 'CAMPNO', columns_to_add)

#Clean the Data
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower().strip()
    return text

# Applying the function to each text column in each DataFrame
text_columns = ['DESC_DEFECT', 'CONEQUENCE_DEFECT','CORRECTIVE_ACTION','NOTES', 'MFGTXT', 'RCLSUBJ', 'COMPDESC', 'SUMMARY', 'CDESCR']  
dataframes = [df_flat_rcl, df_qtrly_rpts, df_annual_rpts, df_ManufacturerCommunications, df_Manufacturer_Communications_pdf, df_RecallCommunication, df_complaints, df_investigation]
for df in dataframes:
    df['YEAR'] = df['YEAR'].astype(str)
    
for df in dataframes:
    for col in text_columns:
        if col in df.columns:
            df[col] = df[col].apply(clean_text)
            
#Feature Extraction
def add_sentiment_analysis(df, text_column):
    # Check if the text column exists in the DataFrame
    if text_column in df.columns:
        # Sentiment polarity score
        df[text_column + '_sentiment'] = df[text_column].apply(lambda x: TextBlob(x).sentiment.polarity if isinstance(x, str) else None)
    return df

# List of DataFrames and their respective text columns for sentiment analysis
df_text_columns = [
    (df_flat_rcl, ['DESC_DEFECT', 'CONEQUENCE_DEFECT']),
    (df_qtrly_rpts, ['MFGTXT', 'RCLSUBJ']),
    (df_annual_rpts, ['MFGTXT','RCLSUBJ', 'NOTES']),
    (df_ManufacturerCommunications, ['SUMMARY']),
    (df_Manufacturer_Communications_pdf, ['SUMMARY']),
    (df_RecallCommunication, ['SUMMARY']),
    (df_complaints, ['COMPDESC', 'CDESCR']),
    (df_investigation, ['SUMMARY'])
]

# Applying sentiment analysis
for df, text_cols in df_text_columns:
    for col in text_cols:
        df = add_sentiment_analysis(df, col)
            
#Topic Modeling: LDA (Latent Dirichlet Allocation)
some_minimum_length = 5
def perform_topic_modeling(df, text_column, n_topics=5, n_words=10):
    if text_column in df.columns:
        text_data = df[text_column].dropna()
        
        # Check if text_data is empty or has very short documents
        if text_data.empty or text_data.str.len().mean() < some_minimum_length:
            print(f"Skipping {text_column} due to insufficient text data.")
            return
        # Vectorize the text
        vectorizer = CountVectorizer(max_df=0.90, min_df=1, stop_words=None)
        dtm = vectorizer.fit_transform(text_data)

        # LDA model
        lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
        lda.fit(dtm)

        # Print topics and words
        print(f"Topics for {text_column}:")
        for index, topic in enumerate(lda.components_):
            print(f'Top {n_words} words for topic #{index}')
            print([vectorizer.get_feature_names_out()[i] for i in topic.argsort()[-n_words:]])
            print('\n')
            
            # Assigning dominant topic to each document
        topic_results = lda.transform(dtm)
        df[text_column + '_dominant_topic'] = topic_results.argmax(axis=1)
            
# Applying LDA topic modeling
for df, text_cols in df_text_columns:
    for col in text_cols:
        perform_topic_modeling(df, col, n_topics=5, n_words=10)
  
#aggregating sentiment scores
def aggregate_sentiment(df, sentiment_col):
    return df.groupby(['MAKE', 'MODEL', 'YEAR'])[sentiment_col].mean().reset_index()

aggregated_sentiments = []

for df, text_cols in df_text_columns:
    for col in text_cols:
        sentiment_col = col + '_sentiment'
        if sentiment_col in df.columns:
            agg_df = aggregate_sentiment(df, sentiment_col)
            agg_df.rename(columns={sentiment_col: f'avg_{sentiment_col}'}, inplace=True)
            aggregated_sentiments.append(agg_df)

# Concatenate all aggregated sentiment scores into a single dataframe
final_sentiment_df = pd.concat(aggregated_sentiments, ignore_index=True)

df_SafetyRatings = df_SafetyRatings.merge(final_sentiment_df, on=['MAKE', 'MODEL', 'YEAR'], how='left')
print(df_SafetyRatings.head())

########################
#Feature Engineering
#Recalls
df_flat_rcl['recall_count'] = 1
recall_counts = df_flat_rcl.groupby(['MAKE', 'MODEL', 'YEAR'])['recall_count'].sum().reset_index()

# Merging recall counts with the Safety Ratings data
df_SafetyRatings = df_SafetyRatings.merge(recall_counts, on=['MAKE', 'MODEL', 'YEAR'], how='left')

#Severity Score
def assign_severity_score(text):
    if pd.isna(text):
        return 0  # Assigning a default score for NaN values

    # Define keywords and their corresponding severity scores
    severity_keywords = {
        'fire': 10, 'explosion': 10, 'crash': 9, 'collision': 9, 'fatal': 9,
        'death': 9, 'injury': 8, 'burn': 8, 'electrocution': 8, 'poisoning': 8,
        'toxic': 7, 'leak': 7, 'failure': 6, 'malfunction': 6, 'defect': 5,
        'error': 5, 'fault': 5, 'recall': 4, 'problem': 4, 'issue': 3,
        'complaint': 3, 'negative': 2, 'minor': 1, 'cosmetic': 1
    }

    # Assign the highest severity score based on the keywords present
    severity_score = 0
    for keyword, score in severity_keywords.items():
        if keyword in text:
            severity_score = max(severity_score, score)

    return severity_score

for df in dataframes:
    for col in text_columns:
        if col in df.columns:
            df[col + '_severity'] = df[col].apply(assign_severity_score)
            
        

            
for df in dataframes:
    for col in text_columns:
        severity_col = col + '_severity'
        if severity_col in df.columns:
            agg_df = df.groupby(['MAKE', 'MODEL', 'YEAR'])[severity_col].mean().reset_index()
            df_SafetyRatings = df_SafetyRatings.merge(agg_df, on=['MAKE', 'MODEL', 'YEAR'], how='left', suffixes=('', '_y'))
            
# Dropping duplicated columns that have '_y' suffix if they exist
for col in [col for col in df_SafetyRatings.columns if col.endswith('_y')]:
    if col in df_SafetyRatings:
        df_SafetyRatings.drop(col, axis=1, inplace=True)
        
for col in [col for col in df_SafetyRatings.columns if col.endswith('_x')]:
    if col in df_SafetyRatings:
        df_SafetyRatings.drop(col, axis=1, inplace=True)

#Recall Impact and weightage
# Apply the severity score calculation to df_flat_rcl
df_flat_rcl['severity_score'] = df_flat_rcl['DESC_DEFECT'].apply(assign_severity_score)
df_flat_rcl['recall_impact'] = df_flat_rcl['severity_score'] * df_flat_rcl['POTAFF']  # Assuming 'POTAFF' is the potential number of affected vehicles
recall_impact = df_flat_rcl.groupby(['MAKE', 'MODEL', 'YEAR'])['recall_impact'].sum().reset_index()
df_SafetyRatings = df_SafetyRatings.merge(recall_impact, on=['MAKE', 'MODEL', 'YEAR'], how='left')

#Investigations
def categorize_outcome(summary):
    if pd.isna(summary):
        return 'unknown'
    summary = summary.lower()
    if 'recall' in summary:
        return 'led_to_recall'
    elif 'no longer in business' in summary or 'terminated its business' in summary:
        return 'closed_no_action'
    else:
        return 'inconclusive'

df_investigation['investigation_outcome'] = df_investigation['SUMMARY'].apply(categorize_outcome)

def investigation_weight(row):
    if row['investigation_outcome'] == 'led_to_recall':
        return 5
    elif row['investigation_outcome'] == 'inconclusive':
        return 2
    elif row['investigation_outcome'] == 'closed_no_action':
        return 1
    else:
        return 0

df_investigation['investigation_weight'] = df_investigation.apply(investigation_weight, axis=1)

investigation_weights = df_investigation.groupby(['MAKE', 'MODEL', 'YEAR'])['investigation_weight'].sum().reset_index()
df_SafetyRatings = df_SafetyRatings.merge(investigation_weights, on=['MAKE', 'MODEL', 'YEAR'], how='left').fillna(0)

#Temporal Recall Data
df_flat_rcl['RCDATE'] = pd.to_datetime(df_flat_rcl['RCDATE'], errors='coerce')
df_qtrly_rpts['ODATE'] = pd.to_datetime(df_qtrly_rpts['ODATE'], errors='coerce')
df_annual_rpts['ODATE'] = pd.to_datetime(df_annual_rpts['ODATE'], errors='coerce')

df_flat_rcl['RCDATE_year'] = df_flat_rcl['RCDATE'].dt.year
df_flat_rcl['RCDATE_month'] = df_flat_rcl['RCDATE'].dt.month
df_flat_rcl['RCDATE_quarter'] = df_flat_rcl['RCDATE'].dt.quarter

df_qtrly_rpts['ODATE_year'] = df_qtrly_rpts['ODATE'].dt.year
df_qtrly_rpts['ODATE_month'] = df_qtrly_rpts['ODATE'].dt.month

df_annual_rpts['ODATE_year'] = df_annual_rpts['ODATE'].dt.year

# Calculating days since the first recall
df_flat_rcl['days_since_first_recall'] = (df_flat_rcl['RCDATE'] - df_flat_rcl['RCDATE'].min()).dt.days

df_flat_rcl['RCDATE_year'] = df_flat_rcl['RCDATE'].dt.year.astype(str)
df_SafetyRatings['YEAR'] = df_SafetyRatings['YEAR'].astype(str)

# Counting recalls per year for each MAKE-MODEL combination
yearly_recall_counts = df_flat_rcl.groupby(['MAKE', 'MODEL', 'RCDATE_year']).size().reset_index(name='annual_recall_count')

#Merging
df_SafetyRatings = df_SafetyRatings.merge(yearly_recall_counts, left_on=['MAKE', 'MODEL', 'YEAR'], right_on=['MAKE', 'MODEL', 'RCDATE_year'], how='left')
df_SafetyRatings.fillna({'annual_recall_count': 0}, inplace=True)


# Selecting numerical columns for normalization
# structured_features = ['severity_score', 'recall_impact', 'investigation_weight', 'outcome_counts','annual_recall_count']  # Add other numerical columns as needed
structured_features = ['recall_impact', 'investigation_weight', 'annual_recall_count']  # Add other numerical columns as needed

# Replace empty strings with NaN and then handle missing values
df_SafetyRatings[structured_features] = df_SafetyRatings[structured_features].replace('', np.nan)

#fill missing values with a specific value : 0
df_SafetyRatings[structured_features] = df_SafetyRatings[structured_features].fillna(0)

# Convert columns to numeric
df_SafetyRatings[structured_features] = df_SafetyRatings[structured_features].apply(pd.to_numeric, errors='coerce')

scaler = StandardScaler()
df_SafetyRatings[structured_features] = scaler.fit_transform(df_SafetyRatings[structured_features])

#############################3
#Prepare to use transformers
#concatenate the text data
def aggregate_text(df, text_columns, groupby_cols=['MAKE', 'MODEL', 'YEAR']):
    df['combined_text'] = df[text_columns].fillna('').agg(' '.join, axis=1)
    return df.groupby(groupby_cols)['combined_text'].agg(' '.join).reset_index()

# Specify the text columns for each dataframe
df_flat_rcl_text_cols = ['DESC_DEFECT', 'CONEQUENCE_DEFECT', 'CORRECTIVE_ACTION', 'NOTES']
df_qtrly_rpts_text_cols = ['MFGTXT', 'RCLSUBJ']
df_annual_rpts_text_cols = ['MFGTXT','RCLSUBJ']
df_ManufacturerCommunications_text_cols = ['SUMMARY']
df_Manufacturer_Communications_pdf_text_cols = ['SUMMARY']
df_RecallCommunication_text_cols =  ['SUMMARY']
df_complaints_text_cols = ['COMPDESC', 'CDESCR']
df_investigation_text_cols =  ['SUMMARY']


# Aggregate text for each dataframe
df_flat_rcl_agg = aggregate_text(df_flat_rcl, df_flat_rcl_text_cols)
df_qtrly_rpts_agg = aggregate_text(df_qtrly_rpts, df_qtrly_rpts_text_cols)
df_annual_rpts_agg = aggregate_text(df_annual_rpts, df_annual_rpts_text_cols)
df_ManufacturerCommunications_agg = aggregate_text(df_ManufacturerCommunications, df_ManufacturerCommunications_text_cols)
df_Manufacturer_Communications_pdf_agg = aggregate_text(df_Manufacturer_Communications_pdf, df_Manufacturer_Communications_pdf_text_cols)
df_RecallCommunication_agg = aggregate_text(df_RecallCommunication, df_RecallCommunication_text_cols)
df_complaints_agg = aggregate_text(df_complaints, df_complaints_text_cols)
df_investigation_agg = aggregate_text(df_investigation, df_investigation_text_cols)

# Initialize combined_text in df_SafetyRatings
df_SafetyRatings['combined_text'] = ''

# Define a list of all aggregated text dataframes
aggregated_text_dfs = [df_flat_rcl_agg, df_qtrly_rpts_agg,df_annual_rpts_agg, df_ManufacturerCommunications_agg, df_Manufacturer_Communications_pdf_agg, df_RecallCommunication_agg, df_complaints_agg, df_investigation_agg]

# Merge each aggregated text dataframe with df_SafetyRatings
for agg_df in aggregated_text_dfs:
    df_SafetyRatings = df_SafetyRatings.merge(agg_df, on=['MAKE', 'MODEL', 'YEAR'], how='left', suffixes=('', '_extra'))
    df_SafetyRatings['combined_text'] = df_SafetyRatings['combined_text'] + ' ' + df_SafetyRatings['combined_text_extra'].fillna('')
    df_SafetyRatings.drop('combined_text_extra', axis=1, inplace=True)


#Saving the df_SafetyRatings dataframe
import os
import pickle

directory = './datasets/NHTSA_Data'  # 'data' folder in the current working directory
file_name = 'df_SafetyRatings.pkl'
file_path = os.path.join(directory, file_name)
if not os.path.exists(directory):
    os.makedirs(directory)
    
with open(file_path, 'wb') as file:
    pickle.dump(df_SafetyRatings, file)
    
# Loading the df_SafetyRatings dataframe
with open(file_path, 'rb') as file:
    df_SafetyRatings = pickle.load(file)

    
#Tokenizing Text Data for DistilBERT

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')


def batch_tokenize(tokenizer, texts, batch_size=100):
    batched_input_ids = []
    batched_attention_masks = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = tokenizer.batch_encode_plus(
            batch,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        batched_input_ids.append(inputs['input_ids'])
        batched_attention_masks.append(inputs['attention_mask'])

    input_ids = torch.cat(batched_input_ids, dim=0)
    attention_masks = torch.cat(batched_attention_masks, dim=0)
    return input_ids, attention_masks

# Apply batch tokenization
input_ids, attention_masks = batch_tokenize(tokenizer, df_SafetyRatings['combined_text'].tolist(), batch_size=100)

# Convert structured features to PyTorch tensor
structured_data = torch.tensor(df_SafetyRatings[structured_features].values.astype('float32'))

# Create the final dataset
final_dataset = torch.utils.data.TensorDataset(input_ids, attention_masks, structured_data)

# Run garbage collection
gc.collect()

# Tokenizing the text data
# tokenized_data = tokenizer.batch_encode_plus(
#     df_SafetyRatings['combined_text'].tolist(),
#     max_length = 512,
#     padding='max_length',
#     truncation=True,
#     return_tensors='pt'
# )

# # Convert structured features to a PyTorch tensor
# structured_data = torch.tensor(df_SafetyRatings[structured_features].values)

# # Your final dataset for training will include 'input_ids', 'attention_mask' from tokenized_data and 'structured_data'
# final_dataset = torch.utils.data.TensorDataset(
#     tokenized_data['input_ids'], 
#     tokenized_data['attention_mask'], 
#     structured_data
# )

###############
#Modeling & Training
from transformers import DistilBertModel
import torch.nn as nn

class CustomHybridModel(nn.Module):
    def __init__(self, num_structured_features, num_labels):
        super(CustomHybridModel, self).__init__()
        self.distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.classifier = nn.Linear(self.distilbert.config.hidden_size + num_structured_features, num_labels)

    def forward(self, input_ids, attention_mask, structured_data):
        distilbert_output = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = distilbert_output[0][:, 0]
        combined_input = torch.cat((pooled_output, structured_data), 1)
        output = self.classifier(combined_input)
        return output

# Initialize the model
num_structured_features = len(structured_features)
num_labels = 3
model = CustomHybridModel(num_structured_features, num_labels)

from torch.utils.data import random_split
epochs_num = 1
# Splitting the data into training and validation sets
train_size = int(0.8 * len(final_dataset))
val_size = len(final_dataset) - train_size
train_dataset, val_dataset = random_split(final_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

# Define an optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
loss_fn = nn.BCEWithLogitsLoss()

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def calculate_metrics(preds, labels):
    preds = preds.round()  # Adjust based on your specific needs
    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds)
    recall = recall_score(labels, preds)
    f1 = f1_score(labels, preds)
    return accuracy, precision, recall, f1


# Training loop
for epoch in range(epochs_num):  # Number of epochs
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids, attention_mask, structured_data = batch
        labels = structured_data[:, :num_labels]
        outputs = model(input_ids, attention_mask, structured_data)
        loss = loss_fn(outputs, labels)  # Update as per your label structure
        loss.backward()
        optimizer.step()
        
    model.eval()
    val_preds, val_labels = [], []
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            input_ids, attention_mask, structured_data = batch
            labels = structured_data[:, :num_labels]  # Extract labels
            outputs = model(input_ids, attention_mask, structured_data)
            val_preds.append(outputs.sigmoid().cpu().numpy())
            val_labels.append(labels.cpu().numpy())

    # Convert predictions and labels to flat lists
    val_preds = np.concatenate(val_preds).flatten()
    val_labels = np.concatenate(val_labels).flatten()

    # Calculate and print metrics
    accuracy, precision, recall, f1 = calculate_metrics(val_preds, val_labels)
    print(f"Epoch {epoch+1}: Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}")    
    
    
from sklearn.metrics import mean_squared_error, roc_auc_score

# Example for calculating MSE and RMSE for a regression task
mse = mean_squared_error(val_labels, val_preds)
rmse = np.sqrt(mse)
print(f"MSE: {mse}, RMSE: {rmse}")

# Example for calculating ROC-AUC for a classification task
roc_auc = roc_auc_score(val_labels, val_preds)
print(f"ROC-AUC: {roc_auc}")


from sklearn.metrics import mean_squared_error, mean_absolute_error

# Splitting data into training and validation sets
train_data, val_data, train_indices, val_indices = train_test_split(
    df_SafetyRatings,  # your main DataFrame
    df_SafetyRatings.index,  # indices of your DataFrame
    test_size=0.2,  # 20% of the data for validation
    random_state=42
)
# Assuming your model predicts a continuous score
val_preds_continuous = model(input_ids, attention_mask, structured_data).squeeze().detach().numpy()
val_labels_continuous = df_SafetyRatings.loc[val_indices, 'OVERALLRATING'].values  # Adjust based on your dataset

# Calculate regression metrics
mse = mean_squared_error(val_labels_continuous, val_preds_continuous)
mae = mean_absolute_error(val_labels_continuous, val_preds_continuous)
print(f"Mean Squared Error: {mse}, Mean Absolute Error: {mae}")


       
#############################################################