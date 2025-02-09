"""
Here's a summary of my debugging journey to arrive at this code - it is my data story!

1. Initial Data Understanding
- First, I discovered two key files:
  * GSE218542_Matrix_processed.txt (large file with methylation data)
  * GSE218542_series_MetaData.txt (metadata file with labels)
- Found that methylation data had samples as columns, which is unusual for machine learning tasks

2. Key Challenge Identified
- I found a mismatch between identifiers:
  * Methylation data used array IDs (like "206439840009_R04C01")
  * Metadata used GSM IDs (like "GSM6751587")
- This explained why my first attempt at merging failed

3. Metadata Structure Discovery
- By printing metadata content, I found three crucial lines:
  * Sample_description line → array IDs
  * Sample_geo_accession line → GSM IDs 
  * methylation class line → labels
- This allowed me to create the mapping between different IDs

4. Data Integration Solution
- Created two dictionaries to map between ID types:
  * gsm_to_array: maps GSM IDs to array IDs
  * array_to_gsm: maps array IDs to GSM IDs
  * labels: maps array IDs to their class labels

5. Data Transformation
- Cleaned methylation data by removing P-VALUE columns
- Transposed data to get samples in rows (ML standard format)
- Added labels using the mapping I created

6. Validation
- Final shape: (86, 128526)
- Confirmed 5 classes with expected distribution:
  * GG, PTPN11: 65 samples
  * LGG, GG: 9 samples
  * LGG, DNT: 5 samples
  * LGG, PXA: 4 samples
  * LGG, MYB: 3 samples

This debugging journey demonstrates:
1. The importance of understanding data structure
2. How to handle ID mapping challenges
3. The value of proper data validation at each step
4. The process of preparing genomic data for machine learning

"""


# 1. Initial Data Understanding
# First attempt to read the data
import pandas as pd

# Read methylation data
methylation_data = pd.read_csv('Data/GSE218542_Matrix_processed.txt', sep='\t')
print("Methylation data shape:", methylation_data.shape)
print("\nFirst few columns:", methylation_data.columns[:5])

# 2. Exploring Metadata Structure
# Look at metadata content
with open('Data/GSE218542_series_MetaData.txt', 'r') as file:
    content = file.readlines()
    # Print first few lines to understand structure
    for line in content[:20]:
        if 'Sample' in line:  # Only look at Sample-related lines
            print(line.strip())

# 3. Create ID Mappings
# Create mapping dictionaries
gsm_to_array = {}
array_to_gsm = {}
labels = {}

with open('Data/GSE218542_series_MetaData.txt', 'r') as file:
    content = file.readlines()
    
    for line in content:
        if 'Sample_description' in line and '_R' in line:  # Array IDs
            array_ids = line.split('"')[1::2]
        elif 'Sample_geo_accession' in line:  # GSM IDs
            gsm_ids = line.split('"')[1::2]
        elif 'methylation class:' in line:  # Labels
            class_labels = line.split('"')[1::2]
            class_labels = [label.split(': ')[1] for label in class_labels]

    # Create mappings
    for gsm_id, array_id, label in zip(gsm_ids, array_ids, class_labels):
        gsm_to_array[gsm_id] = array_id
        array_to_gsm[array_id] = gsm_id
        labels[array_id] = label

# Verify mappings
print("Number of GSM IDs:", len(gsm_ids))
print("Number of Array IDs:", len(array_ids))
print("Number of Labels:", len(class_labels))
```

# 4. Clean and Transform Methylation Data
# Read and clean methylation data
methylation_data = pd.read_csv('Data/GSE218542_Matrix_processed.txt', sep='\t')

# Remove P-VALUE columns
value_cols = [col for col in methylation_data.columns if 'P-VALUE' not in col]
cleaned_methylation = methylation_data[value_cols]

# Create final dataset
methylation_matrix = cleaned_methylation.set_index('ID_REF').transpose()

# Add labels
methylation_matrix['label'] = methylation_matrix.index.map(lambda x: labels.get(x))

# 5. Validate Final Dataset
# Check final dataset
print("Final dataset shape:", methylation_matrix.shape)
print("\nNumber of samples per class:")
print(methylation_matrix['label'].value_counts())
print("\nFirst few rows and columns:")
print(methylation_matrix.iloc[:5, :5])

# Save processed data
methylation_matrix.to_csv('Data/methylation_with_labels.csv')

# 6. Final Verification of Saved Data
# Read saved data and verify
final_data = pd.read_csv('Data/methylation_with_labels.csv')
print("Loaded data shape:", final_data.shape)
print("\nColumns (first 5 and last 5):")
print("First 5:", final_data.columns[:5].tolist())
print("Last 5:", final_data.columns[-5:].tolist())
print("\nClass distribution:")
print(final_data['label'].value_counts())

'''
Each of these code blocks represents a key step in my data preparation process, 
and I can run them sequentially to recreate my data processing pipeline. 
* Note: Input files (GSE218542_Matrix_processed.txt and GSE218542_series_MetaData.txt) 

'''

############################################ All codes in one shot #######################################
# import pandas as pd
# import numpy as np

# # Step 1: Create mapping between GSM IDs and array IDs from metadata
# gsm_to_array = {}
# array_to_gsm = {}
# labels = {}

# with open('Data/GSE218542_series_MetaData.txt', 'r') as file:
#     content = file.readlines()
    
#     for line in content:
#         if 'Sample_description' in line and '_R' in line:  # This line contains array IDs
#             array_ids = line.split('"')[1::2]
#         elif 'Sample_geo_accession' in line:  # This line contains GSM IDs
#             gsm_ids = line.split('"')[1::2]
#         elif 'methylation class:' in line:  # This line contains labels
#             class_labels = line.split('"')[1::2]
#             class_labels = [label.split(': ')[1] for label in class_labels]

#     # Create the mappings
#     for gsm_id, array_id, label in zip(gsm_ids, array_ids, class_labels):
#         gsm_to_array[gsm_id] = array_id
#         array_to_gsm[array_id] = gsm_id
#         labels[array_id] = label

# # Step 2: Read and clean methylation data
# methylation_data = pd.read_csv('Data/GSE218542_Matrix_processed.txt', sep='\t')

# # Remove P-VALUE columns
# value_cols = [col for col in methylation_data.columns if 'P-VALUE' not in col]
# cleaned_methylation = methylation_data[value_cols]

# # Step 3: Create final dataset
# methylation_matrix = cleaned_methylation.set_index('ID_REF').transpose()

# # Add labels
# methylation_matrix['label'] = methylation_matrix.index.map(lambda x: labels.get(x))

# print("Final dataset shape:", methylation_matrix.shape)
# print("\nNumber of samples per class:")
# print(methylation_matrix['label'].value_counts())
# print("\nFirst few rows and columns:")
# print(methylation_matrix.iloc[:5, :5])

# # Save data
# methylation_matrix.to_csv('Data/methylation_with_labels.csv')