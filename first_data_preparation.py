import pandas as pd
import numpy as np
import ast

# Assuming df is your DataFrame
df = pd.read_csv('data/earningsCall_argQ_final.csv')

# Convert string representations of lists to actual lists
df['premise_texts'] = df['premise_texts'].apply(ast.literal_eval)
df['relation type'] = df['relation_types'].apply(ast.literal_eval)

# Initialize lists to store new data
texts = []
targets = []

# Iterate over DataFrame
for i, row in df.iterrows():
    claim_text = row['claim_text']
    premises = row['premise_texts']
    relation_types = row['relation type']
    
    for premise, relation_type in zip(premises, relation_types):
        # Concatenate claim_text and premise
        text = claim_text + " [SEP] " + premise
        texts.append(text)
        
        # Convert relation type to binary
        target = 1 if relation_type in ['SUPPORT', 'ATTACK'] else 0
        targets.append(target)

# Create new DataFrame
new_df = pd.DataFrame({'text': texts, 'label': targets})

# Negative sampling: randomly pair claims with premises from other claims
claim_indices = np.random.permutation(new_df.shape[0])
premise_indices = np.random.permutation(new_df.shape[0])

half = len(claim_indices)//2 -1
new_df.loc[claim_indices[:half], 'text'] = new_df.loc[claim_indices[:half], 'text'] + " [SEP] " + new_df.loc[premise_indices[half:], 'text']
new_df.loc[claim_indices[:half], 'label'] = 0

new_df = new_df.dropna()

# Save new DataFrame to csv
new_df.to_csv('data/argument_relation_class.csv', index=False)
