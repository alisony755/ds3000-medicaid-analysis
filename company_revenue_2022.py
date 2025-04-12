# What are the top 5 pharmaceutical companies that received the most revenue from Medicaid drug spending in 2022?

# Import libraries
import pandas as pd
import plotly.express as px

# Load the dataset
drug_df = pd.read_csv('drug_data.csv')

# Aggregate total Medicaid spending by company
medicaid_company_spending = drug_df.groupby('company', as_index=False)['medicaid_spending_2022'].sum()

# Ensure medicaid_spending_2022 column is numeric and nonzero
medicaid_company_spending = medicaid_company_spending[medicaid_company_spending['medicaid_spending_2022'] > 0]

# Create treemap
fig = px.treemap(medicaid_company_spending,
                 path=['company'],
                 values='medicaid_spending_2022',
                 title="Medicaid Drug Spending by Pharmaceutical Company (2022)",
                 color='medicaid_spending_2022',
                 color_continuous_scale='BuPu')

# Change legend title
fig.update_layout(coloraxis_colorbar=dict(title="Medicaid Spending ($)"))

# Adjust layout for better visualization
fig.update_layout(margin=dict(t=50, l=25, r=25, b=25))

fig.show()
