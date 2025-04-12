# Which prescription drugs have seen the highest price increases from 2018 to 2022?

# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the dataset (assuming it's already been generated in File 1)
drug_df = pd.read_csv('drug_data.csv')

# Create column to store price change data
drug_df['drug_price_change'] = drug_df['avg_spend_2022'] - drug_df['avg_spend_2018']

# Select top 10 drugs with the highest price increases
top_price_increases = drug_df.nlargest(10, 'drug_price_change')

# Filter out rows where either avg_spend_2018 or avg_spend_2022 are None or NaN
top_price_increases = top_price_increases.dropna(subset=['avg_spend_2018', 'avg_spend_2022'])

# Melt the data to convert the spending columns into a single column
top_price_melted = top_price_increases.melt(id_vars=['drug_name'],
                                            value_vars=['avg_spend_2018', 'avg_spend_2022'],
                                            var_name='Year',
                                            value_name='Average Spending')

# Plot paired bar graph
plt.figure(figsize=(12, 8))
sns.barplot(data=top_price_melted,
            x='drug_name',
            y='Average Spending',
            hue='Year',
            palette='GnBu',
            errorbar=None)

# Rename the legend labels
plt.legend(title='Year', labels=['2018 Medicaid Spending', '2022 Medicaid Spending'])

# Add axis labels and title
plt.xlabel('Drug (Brand Name)', fontsize=14)
plt.ylabel('Average Spending Per Weighted Dosage ($)', fontsize=14)
plt.xticks(fontsize=12, rotation=45)
plt.yticks(fontsize=12)
plt.title("Top 10 Drugs by Average Spending Per Weighted Dosage Increase (2018-2022)",
          fontsize=16)

plt.tight_layout()
plt.show()
