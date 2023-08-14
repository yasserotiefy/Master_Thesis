import pandas as pd
import numpy as np
import ast
import matplotlib.pyplot as plt

# Read CSV File
df = pd.read_csv("data/earningsCall_argQ_final.csv")

# Convert string representations of lists to actual lists
df["premise_texts"] = df["premise_texts"].apply(ast.literal_eval)
df["relation_types"] = df["relation_types"].apply(ast.literal_eval)

# Initialize lists to store positive examples
texts = []
targets = []

# Iterate over DataFrame for positive examples
for i, row in df.iterrows():
    claim_text = row["claim_text"]
    premises = row["premise_texts"]
    relation_types = row["relation_types"]

    for premise, relation_type in zip(premises, relation_types):
        text = claim_text + " [SEP] " + premise
        texts.append(text)
        target = 1 if relation_type in ["SUPPORT", "ATTACK"] else 0
        targets.append(target)

# Create DataFrame for positive examples
positive_df = pd.DataFrame({"text": texts, "label": targets})

# Extract unique claims
unique_claims = df["claim_text"].unique()

# Initialize lists to store negative examples
negative_texts = []
negative_labels = []

# Iterate through unique claims for negative examples
# Iterate through unique claims for negative examples
for claim in unique_claims:
    related_premises = df[df["claim_text"] == claim]["premise_texts"].explode().values
    unrelated_premises = df[df["claim_text"] != claim]["premise_texts"].explode().values

    # Convert the claim to a string
    claim_str = str(claim)

    for unrelated_premise in unrelated_premises:
        # Convert the unrelated_premise to a string
        unrelated_premise_str = str(unrelated_premise)

        if unrelated_premise_str not in related_premises:
            text = claim_str + " [SEP] " + unrelated_premise_str
            negative_texts.append(text)
            negative_labels.append(0)


# Create DataFrame for negative examples
negative_df = pd.DataFrame({"text": negative_texts, "label": negative_labels})

# Concatenate positive and negative examples
final_df = pd.concat([positive_df, negative_df], ignore_index=True)

# Separate positive and negative classes
positive_class = final_df[final_df['label'] == 1]
negative_class = final_df[final_df['label'] == 0]

# Randomly sample negative examples to match the number of positive examples
negative_class_sampled = negative_class.sample(len(positive_class))

# Concatenate positive and sampled negative examples
balanced_df = pd.concat([positive_class, negative_class_sampled], ignore_index=True)

# Shuffle the final DataFrame
balanced_df = balanced_df.sample(frac=1).reset_index(drop=True)

# Save new DataFrame to csv
balanced_df.to_csv("data/argument_relation_class.csv", index=False)

# Plot the distribution of classes
class_counts = balanced_df['label'].value_counts()
plt.bar(class_counts.index, class_counts.values, tick_label=['Negative', 'Positive'])
plt.xlabel('Class')
plt.ylabel('Frequency')
plt.title('Distribution of Classes')
plt.savefig('data/class_distribution.png')
plt.show()
