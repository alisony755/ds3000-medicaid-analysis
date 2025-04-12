# Read in the Medicaid Spending by Drug data from Data.CMS.gov

# Import libraries
import requests
from bs4 import BeautifulSoup
from zipfile import ZipFile
from io import BytesIO
import pandas as pd
import os

# Centers for Medicare & Medicaid Services (CMS) dataset URL
base_url = "https://data.cms.gov/data-api/v1/dataset/be64fce3-e835-4589-b46b-024198e524a6/data"

# Parameters for pagination
size = 100  # Number of records per page
offset = 0  # Starting offset

# Dictionary to store scraped data
data_dict = {
    "drug_name": [],      # Name of drug
    "company": [],        # Company manufacturing the drug
    "avg_spend_2018": [], # Average amount spent on dosage (weighted) in 2018
    "avg_spend_2022": [], # Average amount spent on dosage (weighted) in 2022
    "medicaid_spending_2018": [],   # Total amount Medicaid spent on drug in 2018
    "medicaid_spending_2019": [],   # Total amount Medicaid spent on drug in 2019
    "medicaid_spending_2020": [],   # Total amount Medicaid spent on drug in 2020
    "medicaid_spending_2021": [],   # Total amount Medicaid spent on drug in 2021
    "medicaid_spending_2022": []    # Total amount Medicaid spent on drug in 2022
}

# Loop through the pages and fetch data
while True:
    # Construct the request URL with size and offset
    url = f"{base_url}?size={size}&offset={offset}"

    # Fetch data from CMS API
    response = requests.get(url)

    # Check if request is successful
    if response.status_code == 200:
        try:
            data = response.json()  # Parse JSON response

            # If no data is returned, exit loop
            if not data:
                break

            for record in data:
                # Get company name
                company = record.get("Mftr_Name")

                # Skip "Overall" company
                if company != "Overall":
                    # Get the drug name
                    drug_name = record.get("Brnd_Name", None)

                    # Parse spending data
                    avg_spend_2018 = record.get("Avg_Spnd_Per_Dsg_Unt_Wghtd_2018")
                    avg_spend_2022 = record.get("Avg_Spnd_Per_Dsg_Unt_Wghtd_2022")
                    medicaid_spending_2018 = record.get("Tot_Spndng_2018")
                    medicaid_spending_2019 = record.get("Tot_Spndng_2019")
                    medicaid_spending_2020 = record.get("Tot_Spndng_2020")
                    medicaid_spending_2021 = record.get("Tot_Spndng_2021")
                    medicaid_spending_2022 = record.get("Tot_Spndng_2022")

                    # Append values to the dictionary
                    data_dict["drug_name"].append(drug_name if drug_name else 'Unknown')
                    data_dict["company"].append(company if company else 'Unknown')
                    data_dict["avg_spend_2018"].append(float(avg_spend_2018) if avg_spend_2018 else 0)
                    data_dict["avg_spend_2022"].append(float(avg_spend_2022) if avg_spend_2022 else 0)
                    data_dict["medicaid_spending_2018"].append(float(medicaid_spending_2018) if medicaid_spending_2018 else 0)
                    data_dict["medicaid_spending_2019"].append(float(medicaid_spending_2019) if medicaid_spending_2019 else 0)
                    data_dict["medicaid_spending_2020"].append(float(medicaid_spending_2020) if medicaid_spending_2020 else 0)
                    data_dict["medicaid_spending_2021"].append(float(medicaid_spending_2021) if medicaid_spending_2021 else 0)
                    data_dict["medicaid_spending_2022"].append(float(medicaid_spending_2022) if medicaid_spending_2022 else 0)

            # Update the offset for the next request
            offset += size

        except Exception as e:
            print(f"Error parsing JSON: {e}")

    else:
        # Handle request failure
        print(f"Failed to retrieve data. Status code: {response.status_code}")
        break

# Create DataFrame from dictionary
drug_df = pd.DataFrame(data_dict)

# Print preview of data
print(f"Total records fetched: {len(data_dict['drug_name'])}")
print("\nData head:")
print(drug_df.head(25))
print("\nData tail:")
print(drug_df.tail(25))

# Export the DataFrame for use in other files
drug_df.to_csv("drug_data.csv", index=False)
