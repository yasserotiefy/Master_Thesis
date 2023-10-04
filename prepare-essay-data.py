import os
import re
import pandas as pd

def process_file(file_path, output_dir):
    with open(file_path, 'r') as f:
        data = f.read()
    
    # Initialize lists to store table data
    premise_list = []
    claim_list = []
    relation_list = []
    
    # Regular expressions for parsing
    claim_re = re.compile(r'(T\d+)\s+Claim.*\t(.*)')
    premise_re = re.compile(r'(T\d+)\s+Premise.*\t(.*)')
    relation_re = re.compile(r'R\d+\s+(supports|attacks) Arg1:(T\d+) Arg2:(T\d+)')
    
    # Parsing the data
    for line in data.split('\n'):
        claim_match = claim_re.match(line)
        premise_match = premise_re.match(line)
        relation_match = relation_re.match(line)
        
        if claim_match:
            claim_id, claim_text = claim_match.groups()
            claim_list.append({"ID": claim_id, "Text": claim_text})
        elif premise_match:
            premise_id, premise_text = premise_match.groups()
            premise_list.append({"ID": premise_id, "Text": premise_text})
        elif relation_match:
            relation_type, arg1, arg2 = relation_match.groups()
            relation_list.append({"Type": relation_type, "Arg1": arg1, "Arg2": arg2})

    # Initialize list to store the filtered CSV rows
    filtered_csv_rows = []

    # Populate the filtered CSV rows considering only Claim-Premise relationships
    for relation in relation_list:
        relation_type = relation['Type']
        arg1_id = relation['Arg1']
        arg2_id = relation['Arg2']

        arg1_text = next((item['Text'] for item in premise_list if item['ID'] == arg1_id), None)
        arg2_text = next((item['Text'] for item in claim_list if item['ID'] == arg2_id), None)

        if arg1_text and arg2_text:
            filtered_csv_rows.append({"Premise": arg1_text, "Claim": arg2_text, "Relation Type": relation_type})

    # Convert to a DataFrame and then save as CSV
    filtered_csv_df = pd.DataFrame(filtered_csv_rows)
    return filtered_csv_df

# Specify the directory containing the files and the output directory for CSVs
input_dir = "ArgumentAnnotatedEssays-1.0/brat-project"
output_dir = "ArgumentAnnotatedEssays-1.0/output"

# Create output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Iterate over each file in the directory
list_of_dataframes = []
for filename in os.listdir(input_dir):
    if filename.endswith('.ann'):  # Assuming the files are text files
        file_path = os.path.join(input_dir, filename)
        df = process_file(file_path, output_dir)
        list_of_dataframes.append(df)

# Concatenate all the dataframes into a single dataframe
final_df = pd.concat(list_of_dataframes)


# concate claim and premise str with [SEP] token
final_df["text"] = final_df["Claim"] + " [SEP] " + final_df["Premise"]

# Create label
final_df["label"] = "Related"

# Drop unnecessary columns

# create negative examples
# Extract unique claims given text is a concatenated claim-premise by [SEP] token
unique_claims = final_df["Claim"].unique()



# Initialize lists to store negative examples
negative_texts = []
negative_labels = []

# Iterate through unique claims for negative examples

for claim in unique_claims:
    # Get all the premises for the given claim
    premises = final_df[final_df["Claim"] == claim]["Premise"]
    
    # Iterate through premises
    for premise in premises:
        # Create negative example by concatenating claim and premise with [SEP] token
        negative_text = str(claim) + " [SEP] " + str(premise)
        
        # Append negative example to list of negative examples
        negative_texts.append(negative_text)
        
        # Append label to list of labels
        negative_labels.append("Not_Related")



# Create DataFrame for negative examples with number of negative examples equal to number of positive examples
negative_df = pd.DataFrame({"text": negative_texts[:len(final_df)], "label": negative_labels[:len(final_df)]})

# Concatenate positive and negative examples and shuffle

final_df = pd.concat([final_df, negative_df])
final_df = final_df.sample(frac=1).reset_index(drop=True)

# Drop unnecessary columns
final_df = final_df.drop(["Claim", "Premise", "Relation Type"], axis=1)

# convert label to 0 and 1
final_df["label"] = final_df["label"].apply(lambda x: 1 if x == "Related" else 0)

# Save the final dataframe as a CSV
final_df.to_csv(os.path.join(output_dir, "essay-data.csv"), index=False)
