# Can we predict which pharmaceutical companies will see the highest total Medicaid spending in the future (2025-2026) based on past spending data?

# Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from matplotlib.ticker import FuncFormatter

# Load the dataset
drug_df = pd.read_csv('drug_data.csv')

# Define years used for training
years = np.array([2018, 2019, 2020, 2021, 2022])

# Add new columns for predictions (2023-2026)
future_years = [2023, 2024, 2025, 2026]
for year in future_years:
    drug_df[f'medicaid_spending_{year}'] = np.nan

def predict_future_spending(row):
    """ Predicts Medicaid spending for a row (2023-2026) via linear regression.

    Arguments:
      row (pd.Series): A row from the DataFrame containing Medicaid spending
      data for a particular drug from 2018-2022.

    Returns:
      row (pd.Series): The same row with added predictions for Medicaid spending
      in 2023, 2024, 2025, and 2026.

    """

    # Extract the spending data
    y = np.array([
        row['medicaid_spending_2018'],
        row['medicaid_spending_2019'],
        row['medicaid_spending_2020'],
        row['medicaid_spending_2021'],
        row['medicaid_spending_2022']
    ])

    # Ensure there are no NaNs in the data
    if np.all(np.isfinite(y)):
        # Reshape data for linear regression
        X = years.reshape(-1, 1)
        X = np.hstack((np.ones_like(X), X))  # Add intercept term

        # Fit the linear regression model using the normal equation
        theta = np.linalg.inv(X.T @ X) @ X.T @ y

        # Predict future spending
        for year in future_years:
            x_future = np.array([1, year])  # Add intercept term
            predicted_spending = np.dot(x_future, theta)
            row[f'medicaid_spending_{year}'] = max(predicted_spending, 0)  # Ensure no negative predictions

    return row

# Apply the prediction function to each row
drug_df_with_predictions = drug_df.apply(predict_future_spending, axis=1)

def aggregate_medicaid_spending_by_company(df):
    """ Aggregates Medicaid spending by pharmaceutical company (2018-2026).

    Args:
        df (pd.DataFrame): DataFrame containing Medicaid spending data with columns:
                           - 'company' (str): Company name.
                           - 'medicaid_spending_{year}' (float): Mediciaid spending
                              for company by year.

    Returns:
        company_spending (pd.DataFrame): New DataFrame with each company as a
                                         row and total Medicaid spending from
                                         2018 to 2026 as columns.
    """
    # Create a new DataFrame with unique companies and yearly spending columns
    company_spending = pd.DataFrame({
        'company': df['company'].unique()
    })

    # Add profit columns for each year (2018-2026), initialized to 0.0 (float)
    for year in range(2018, 2027):
        company_spending[f'profit_{year}'] = 0.0

    # Aggregate spending by company
    for _, row in df.iterrows():
        company = row['company']

        for year in range(2018, 2027):
            # Ensure the spending value is treated as a float (no list handling)
            spending_value = float(row[f'medicaid_spending_{year}'])

            # Add the spending to the corresponding company and year
            company_spending.loc[company_spending['company'] == company,
                                 f'profit_{year}'] += spending_value

    return company_spending

# Get the aggregated spending by company
company_spending_df = aggregate_medicaid_spending_by_company(drug_df_with_predictions)

# Sort by total Medicaid spending in 2026 and display the top 5 companies
top_companies = company_spending_df.sort_values('profit_2026', ascending=False).head()

def top_companies_2026(df):
    """ Identifies the top 5 drugs by Medicaid spending in 2026.

    Arguments:
        df (pd.DataFrame): DataFrame containing Medicaid spending data for
                           2018-2026.

    Returns:
        top_5_df (pd.DataFrame): A new DataFrame containing the top 5 drugs and
                                 their spending data across the years 2018-2026.

    """

    # Get the top 5 most expensive drugs by predicted Medicaid spending (2026)
    top_5 = df.nlargest(5, 'profit_2026')

    # Return the top 5 drugs
    return top_5

# Get the top 5 companies by predicted total Medicaid spending (2026)
top_5_companies = top_companies_2026(top_companies)

def plot_top_5_companies_by_spending(company_spending_df):
    """
    Plots Medicaid spending trends (2018-2026) for the top 5 companies by total spending.

    Args:
        company_spending_df (pd.DataFrame): DataFrame containing company spending data
                                            across the years 2018-2026.

    Returns:
        None, outputs a graph showing the Medicaid spending for the 5 companies.

    """

    # Sort by total Medicaid spending across all years
    company_spending_df['total_spending'] = company_spending_df[[f'profit_{year}'
                                            for year in range(2018, 2027)]].sum(axis=1)

    # Get the top 5 companies by total spending
    top_5_companies = company_spending_df.sort_values('total_spending',
                                                      ascending=False).head(5)

    # Set up the plot
    fig, ax = plt.subplots(figsize=(12, 6))

    # Define the columns to plot
    years = [f'profit_{year}' for year in range(2018, 2027)]

    # Use the 'plasma' colormap
    cmap = plt.get_cmap('plasma')

    # Plot spending for each of the top 5 companies with the plasma color map
    for i, (_, row) in enumerate(top_5_companies.iterrows()):
        ax.plot(range(2018, 2027), row[years], label=row['company'], marker='o',
                color=cmap(i / len(top_5_companies)))

    # Adding title, labels, and legend
    ax.set_title('Top 5 Companies by Total Medicaid Spending Recieved (2018-2026)', fontsize=16)
    ax.set_xlabel('Year', fontsize=14)
    ax.set_ylabel('Total Medicaid Spending Recieved (Billions)', fontsize=14)
    ax.legend(title='Company', title_fontsize=14, fontsize=12)
    ax.grid(True)

    # Format the x-axis with years (2018-2026)
    ax.set_xticks(range(2018, 2027))
    ax.set_xticklabels(range(2018, 2027), rotation=45, fontsize=12)

    # Format y-axis to display in billions
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x * 1e-9:.2f}B'))
    ax.tick_params(axis='y', labelsize=12)  # Increase fontsize for y-tick labels

    # Display the plot
    plt.tight_layout()
    plt.show()

# Display the plot of the top 5 companies by total Medicaid spending (2018-2026)
plot_top_5_companies_by_spending(company_spending_df)
