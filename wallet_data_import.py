# John Sherrill
# May 19, 2025
"""
Dave Tannenbaum Wallet project.
https://github.com/davetannenbaum/lost-wallets-social-capital
This file was largely constructed as a Chatgpt conversion of the .do file in Dave
Tannenbaum's github repo for the wallet study.
"""

# %%
import os
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------
# Data Wrangling
# ---------------------------------------------------------------------

os.chdir("lost-wallets-social-capital/data/")

# Add Integrated Values Survey (WVS/EVS combined) data set
ivs = pd.read_stata("IVS.dta")

# List of columns that need coding to be reordered:
bad_order = [
    "civic1",
    "civic2",
    "civic3",
    "civic_1",
    "trust_others",
    "trust_justmet",
    "employment_status",
]

# Reverse code of trust_others measure. Higher should indicate more trust.
for col in bad_order:
    ivs[col] = ivs[col].cat.reorder_categories(ivs[col].cat.categories[::-1])

# Proper coding for generalized trust
ivs["general_trust"] = ivs["general_trust"].cat.rename_categories(
    {0: "Must be very careful"}
)

# Append Kenya data
kenya = pd.read_stata("Kenya.dta")
ivs = pd.concat([ivs, kenya], ignore_index=True)

# placeholder for UAE
ivs.loc[len(ivs), :] = {"Country": "UAE"}

# Add Falk et al. Global Preference data
gps = pd.read_stata("gps_honesty.dta")
ivs = ivs.merge(gps, on="Country", how="left")
ivs.rename(
    columns={
        "trust": "GPS_trust",
        "altruism": "GPS_altruism",
        "posrecip": "GPS_posrecip",
    },
    inplace=True,
)

# Add Enke MFQ data
mfq = pd.read_stata("moral_values_honesty.dta")
ivs = ivs.merge(mfq, on="Country", how="left")
ivs.rename(columns={"values_uniform": "MFQ_genmorality"}, inplace=True)
ivs["MFQ_genmorality"] = 11 - ivs["MFQ_genmorality"]  # reverse code

# Add government efficiency
gee = pd.read_stata("hall_jones_honesty.dta")
ivs = ivs.merge(gee, on="Country", how="left")

# Add World Bank Governance Indicators (WGI)
wgi = pd.read_stata("wgidataset2017.dta")
ivs = ivs.merge(wgi, on="Country", how="left")
# Fix isocode_x, isocode_y duplicates.
ivs = ivs.rename(columns={"isocode_x": "isocode", "isocode_y": "isocode_using"})

# Add Government Letter Grade data
grades = pd.read_stata("letter_grading.dta")
ivs = ivs.merge(grades, on="Country", how="left")

# Fix country codes
country_fix = {
    "Denmark": "DNK",
    "Malaysia": "MYS",
    "New Zealand": "NZL",
    "Norway": "NOR",
}
ivs["isocode"] = ivs.apply(
    lambda row: country_fix.get(row["Country"], row.get("isocode", np.nan)), axis=1
)

# Add country GDP data from 2017 Penn World Tables
pwt = pd.read_stata("pwt91 2017.dta")
ivs = ivs.merge(pwt, on="isocode", how="left")
ivs["log_gdp"] = np.log(ivs["rgdpna"] / ivs["pop"])
ivs["log_tfp"] = np.log(ivs["ctfp"])
ivs = ivs.rename(columns={"Country_x": "Country", "Country_y": "Country_using"})

# Add World Risk Poll data (survey measures of people returning a lost item)
wrp = pd.read_stata("world risk poll.dta")
ivs = ivs.merge(wrp, on="Country", how="left")

# Convert categorical columns into numeric for processing purposes.
ivs = ivs.apply(
    lambda col: (
        col.cat.codes.replace({-1: np.nan}) if col.dtype.name == "category" else col
    )
)

# %%

# ------------------------------------------------
# Create predictors frames
# ------------------------------------------------
predictors1 = ivs.copy()
predictors2 = ivs.copy()
predictors3 = ivs.copy()


# ------------------------------------------------
# Function to compute civic norms PCA and standardize
# ------------------------------------------------
def process_predictors(df):
    """Function to compute civic norms PCA and standardize."""
    cols_to_keep = [
        "general_trust",
        "general_morality",
        "MFQ_genmorality",
        "civic1",
        "civic2",
        "civic3",
        "GPS_trust",
        "GPS_posrecip",
        "GPS_altruism",
        "stranger1",
        "stranger2",
        "log_gdp",
        "log_tfp",
        "gee",
        "letter_grading",
        "trust_justmet",
        "trust_others",
        "general_fair",
        "general_fair2",
        "Country",
    ]
    df_grouped = df[cols_to_keep].groupby("Country").mean().reset_index()

    # PCA on civic norms
    civic_df = df_grouped[["Country", "civic1", "civic2", "civic3"]].dropna()
    pca = PCA(n_components=1)
    civic_df["civic_cooperation"] = pca.fit_transform(
        civic_df[["civic1", "civic2", "civic3"]]
    )

    df_grouped = df_grouped.drop(columns=["civic1", "civic2", "civic3"]).merge(
        civic_df[["Country", "civic_cooperation"]], how="left", on="Country"
    )

    # Standardize variables
    scale_cols = [
        "general_trust",
        "general_morality",
        "MFQ_genmorality",
        "civic_cooperation",
        "GPS_trust",
        "GPS_posrecip",
        "GPS_altruism",
        "stranger1",
        "stranger2",
        "trust_justmet",
        "trust_others",
        "general_fair",
        "general_fair2",
    ]
    scaler = StandardScaler()
    df_grouped[scale_cols] = scaler.fit_transform(df_grouped[scale_cols])

    return df_grouped


# ------------------------------------------------
# Filter for restricted sample (predictors2)
# ------------------------------------------------
predictors2 = predictors2[
    (
        predictors2["employment_status"].isin([1, 2, 3])
        | predictors2["kenya_employment"].isin([2, 3])
        | predictors2["Country"].isin(["UAE"])
    )
    & (
        predictors2["city_size1"].isin([7, 8])
        | predictors2["city_size2"].isin([4, 5])
        | predictors2["city_size3"].isin([4, 5])
        | predictors2["kenya_urban"].isin([1])
        | predictors2["Country"].isin(["Israel", "UAE"])
    )
]

# ------------------------------------------------
# Filter for most recent wave (predictors3)
# 'predictor3' is missing 'trust_others' and 'general_fair' measures.
# ------------------------------------------------
drop_conditions = (
    (predictors3["Country"] == "Argentina") & (predictors3["wvs_wave"] != 6)
    | (predictors3["Country"] == "Australia") & (predictors3["wvs_wave"] != 7)
    | (predictors3["Country"] == "Brazil") & (predictors3["wvs_wave"] != 6)
    | (predictors3["Country"] == "Canada") & (predictors3["wvs_wave"] != 5)
    | (predictors3["Country"] == "Chile") & (predictors3["wvs_wave"] != 7)
    | (predictors3["Country"] == "China") & (predictors3["wvs_wave"] != 6)
    | (predictors3["Country"] == "Croatia") & (predictors3["evs_wave"] != 5)
    | (predictors3["Country"] == "Czech Republic") & (predictors3["evs_wave"] != 5)
    | (predictors3["Country"] == "Denmark") & (predictors3["evs_wave"] != 5)
    | (predictors3["Country"] == "France") & (predictors3["evs_wave"] != 5)
    | (predictors3["Country"] == "Germany") & (predictors3["wvs_wave"] != 6)
    | (predictors3["Country"] == "Ghana") & (predictors3["wvs_wave"] != 6)
    | (predictors3["Country"] == "Greece") & (predictors3["wvs_wave"] != 7)
    | (predictors3["Country"] == "India") & (predictors3["wvs_wave"] != 6)
    | (predictors3["Country"] == "Indonesia") & (predictors3["wvs_wave"] != 7)
    | (predictors3["Country"] == "Israel") & (predictors3["wvs_wave"] != 4)
    | (predictors3["Country"] == "Italy") & (predictors3["evs_wave"] != 5)
    | (predictors3["Country"] == "Kazakhstan") & (predictors3["wvs_wave"] != 7)
    | (predictors3["Country"] == "Malaysia") & (predictors3["wvs_wave"] != 7)
    | (predictors3["Country"] == "Mexico") & (predictors3["wvs_wave"] != 7)
    | (predictors3["Country"] == "Morocco") & (predictors3["wvs_wave"] != 6)
    | (predictors3["Country"] == "Netherlands") & (predictors3["wvs_wave"] != 6)
    | (predictors3["Country"] == "New Zealand") & (predictors3["wvs_wave"] != 7)
    | (predictors3["Country"] == "Norway") & (predictors3["evs_wave"] != 5)
    | (predictors3["Country"] == "Peru") & (predictors3["wvs_wave"] != 7)
    | (predictors3["Country"] == "Poland") & (predictors3["wvs_wave"] != 6)
    | (predictors3["Country"] == "Portugal") & (predictors3["evs_wave"] != 5)
    | (predictors3["Country"] == "Romania") & (predictors3["wvs_wave"] != 6)
    | (predictors3["Country"] == "Russia") & (predictors3["wvs_wave"] != 7)
    | (predictors3["Country"] == "Serbia") & (predictors3["wvs_wave"] != 7)
    | (predictors3["Country"] == "South Africa") & (predictors3["wvs_wave"] != 6)
    | (predictors3["Country"] == "Spain") & (predictors3["evs_wave"] != 5)
    | (predictors3["Country"] == "Sweden") & (predictors3["evs_wave"] != 5)
    | (predictors3["Country"] == "Switzerland") & (predictors3["evs_wave"] != 5)
    | (predictors3["Country"] == "Thailand") & (predictors3["wvs_wave"] != 7)
    | (predictors3["Country"] == "Turkey") & (predictors3["wvs_wave"] != 6)
    | (predictors3["Country"] == "UK") & (predictors3["evs_wave"] != 5)
    | (predictors3["Country"] == "USA") & (predictors3["wvs_wave"] != 7)
)
predictors3 = predictors3[~drop_conditions]

# ------------------------------------------------
# Process predictors
# ------------------------------------------------
predictors1 = process_predictors(predictors1)
predictors2 = process_predictors(predictors2)
predictors3 = process_predictors(predictors3)

# ------------------------------------------------
# Merge predictors with behavioral data
# ------------------------------------------------
behavioral = pd.read_stata("behavioral data.dta")
data1 = behavioral.merge(predictors1, on="Country", how="left")
data2 = behavioral.merge(predictors2, on="Country", how="left")
data3 = behavioral.merge(predictors3, on="Country", how="left")

# %%

paper_predictors = [
    "general_trust",
    "GPS_trust",
    "general_morality",
    "MFQ_genmorality",
    "civic_cooperation",
    "GPS_posrecip",
    "GPS_altruism",
    "stranger1",
]


cat_variables = [
    "male",
    "above40",
    "computer",
    "coworkers",
    "other_bystanders",
    "institution",
    "cond",
    "security_cam",
    "security_guard",
    "local_recipient",
    "no_english",
    "understood_situation",
]

sc_measures = [
    "log_gdp",
    "log_tfp",
    "gee",
    "letter_grading",
]

data_for_ml = data1[
    ["country"] + ["response"] + paper_predictors + cat_variables + sc_measures
]
data_for_ml.to_csv("data_for_ml.csv", index=False)

# %%
