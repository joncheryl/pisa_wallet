# John Sherrill
# May 2025
"""
Dave Tannenbaum Wallet project.
https://github.com/davetannenbaum/lost-wallets-social-capital

Exploratory data analysis. Looking at variables/features used in paper:

Also looking at NAEP scores.
naep_wallet.loc[(naep_wallet['subject'] == 'math') & (naep_wallet['grade'] == 8), /
['city', 'naep_score', 'response']].set_index('city').corr()
"""

# %%
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import chi2_contingency
import numpy as np
from pandas.api.types import is_numeric_dtype

# %%

numeric_features = [
    "general_trust",
    "GPS_trust",
    "general_morality",
    "MFQ_genmorality",
    "civic_cooperation",
    "GPS_posrecip",
    "GPS_altruism",
    "stranger1",
]

cat_cols = [
    "country",
    "response",
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

df = pd.read_csv(
    "lost-wallets-social-capital/data/data_for_ml.csv",
    dtype={col: "category" for col in cat_cols},
)

# %%
# Vizualizations of each feature.
## 1. Histograms for continuous variables and bar plots (histograms) for categorical.


def subplot_subset(cols):
    """Make 2x4 grid of histograms of features."""

    n_cols = 4
    n_rows = 2
    fig = make_subplots(rows=n_rows, cols=n_cols)
    for sp_row, feature in enumerate(cols):
        fig.add_trace(
            go.Histogram(x=df[feature], name=feature),
            row=int(((sp_row) / n_cols) + 1),
            col=(sp_row % n_cols) + 1,
        )

    fig.update_layout(title="Histogram for Each Feature")
    fig.show()


subplot_subset(df.columns[1:9])
subplot_subset(df.columns[9:17])

# %%
## 2. Generalized correlation matrix.


def cramers_v(ex, y_cramer):
    """Cramer correlation between two categorical variables."""
    confusion_matrix = pd.crosstab(ex, y_cramer)
    chi2_cramers = chi2_contingency(confusion_matrix, correction=False)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2_cramers / n
    r, k = confusion_matrix.shape
    return np.sqrt(phi2 / min(k - 1, r - 1))


def correlation_ratio(values, categories):
    """Correlation ratio between a categorical and numeric variable."""

    na_mask = values.notna() & categories.notna()
    num_var = values[na_mask]
    cat_var = categories[na_mask]

    fcat, _ = pd.factorize(cat_var)
    cat_num = np.max(fcat) + 1
    y_avg = num_var.mean()

    numerator = sum(
        len(num_var[fcat == i]) * (num_var[fcat == i].mean() - y_avg) ** 2
        for i in range(cat_num)
    )
    denominator = np.sum((num_var - y_avg) ** 2)

    return np.sqrt(numerator / denominator) if denominator != 0 else 0


df_cols = df.columns.drop("country")
assoc = pd.DataFrame(index=df_cols, columns=df_cols)

for col1 in df_cols:
    for col2 in df_cols:
        if is_numeric_dtype(df[col1]) and is_numeric_dtype(df[col2]):
            # Both numeric:
            assoc.loc[col1, col2] = df[[col1, col2]].corr().iloc[0, 1]
        elif is_numeric_dtype(df[col1]) and not is_numeric_dtype(df[col2]):
            # First numeric, second categorical:
            assoc.loc[col1, col2] = correlation_ratio(df[col1], df[col2])
        elif is_numeric_dtype(df[col2]) and not is_numeric_dtype(df[col1]):
            # First categorical, second numeric:
            assoc.loc[col1, col2] = correlation_ratio(df[col2], df[col1])
        else:
            # Both categorical:
            assoc.loc[col1, col2] = cramers_v(df[col1], df[col2])

assoc = assoc.astype(float)

fig_assoc = px.imshow(
    assoc,
    labels=dict(x="Features", y="Features", color="Correlation"),
    color_continuous_scale="RdBu_r",
    zmin=-1,
    zmax=1,
    aspect="auto",
)
fig_assoc.update_layout(title_text="Association Matrix (generalized corr matrix)")
fig_assoc.show()

# %%
# Preprocessing

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder,
)
from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import ElasticNet, LogisticRegression, RidgeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier

# Numeric feature pipeline
numeric_features = [
    "general_trust",
    "GPS_trust",
    "general_morality",
    "MFQ_genmorality",
    "civic_cooperation",
    "GPS_posrecip",
    "GPS_altruism",
    "stranger1",
]
numeric_transformer = Pipeline(
    steps=[("imputer", SimpleImputer(strategy="mean")), ("scaler", StandardScaler())]
)

# Categorical feature pipeline
categorical_features = [
    "male",
    "above40",
    "computer",
    "coworkers",
    "other_bystanders",
    "institution",
    "cond",
]
categorical_transformer = Pipeline(
    steps=[
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ("selector", SelectPercentile(chi2, percentile=50)),
    ]
)

# target feature
target = df["response"].cat.codes

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

clf = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        # ("classifier", LinearSVC()),
        # ("classifier", SVC(random_state=123)),
        ("classifier", SVC(random_state=123, class_weight="balanced")),  # current best
        # ("classifier", KNeighborsClassifier(n_neighbors=31))
        # balanced weights might not matter because of distribution of wallet response
        # ("classifier", LogisticRegression(max_iter=10000, random_state=123))
        # ("classifier", RidgeClassifier(alpha=5))
        # ("classifier", GaussianNB())
        # ("classifier", tree.DecisionTreeClassifier())
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    df.drop(columns="response"), df["response"], test_size=0.2, random_state=23
)

# Curious if weighting the observations according to the inverse frequency of "cond"
# will help the model...
cond_counts = X_train["cond"].value_counts(normalize=True)
weights = X_train["cond"].map(lambda x: 1 / cond_counts[x])

clf.fit(X_train, y_train, classifier__sample_weight=weights)
print("model score: %.3f" % clf.score(X_test, y_test))
clf

# %%

import seaborn as sns

_ = sns.pairplot(df[numeric_features + ["response"]], hue="response")

# %%

# PISA education data: https://nces.ed.gov/surveys/pisa/idepisa/report.aspx
pd.set_option("future.no_silent_downcasting", True)

pisa_reading = (
    pd.read_excel(
        "data/PISA_reading_2022.xlsx",
        skiprows=list(range(0, 11)) + list(range(112, 117)),
        usecols=["Jurisdiction", "Average", "Standard Error"],
        na_values="†",
    )
    .dropna()
    .rename(columns={"Jurisdiction": "country"})
    .replace(
        {
            "Russian Federation": "Russia",
            "United Arab Emirates": "UAE",
            "United Kingdom": "UK",
            "United States": "USA",
        }
    )
    .astype({"Average": float})
)

pisa_reading_2018 = (
    pd.read_excel(
        "data/PISA_reading_2018.xlsx",
        skiprows=list(range(0, 11)) + list(range(112, 117)),
        usecols=["Jurisdiction", "Average", "Standard Error"],
        na_values="†",
    )
    .dropna()
    .rename(columns={"Jurisdiction": "country"})
    .replace(
        {
            "B-S-J-Z (China)": "China",
            "Russian Federation": "Russia",
            "United Arab Emirates": "UAE",
            "United Kingdom": "UK",
            "United States": "USA",
        }
    )
    .astype({"Average": float})
)

wallet_rates = (
    df.groupby("country", observed=True)["response"]
    .value_counts(normalize=True)
    .xs("100", level="response")
    .reset_index()
    .rename(columns={"proportion": "response"})
)

reading_wallets = pisa_reading.merge(wallet_rates, how="inner", on="country")
reading_wallets = reading_wallets.set_index("country")

print(reading_wallets[["Average", "response"]].corr())
px.scatter(
    reading_wallets.reset_index(), x="response", y="Average", hover_data="country"
)

# 1. PISA is more strongly correlated with response rate than survey measures.
# 2. PISA is correlated to survey measures similarly to response rate.
# 3. PISA is a little more correlated to other measures of social capital (log_gdp, etc)
# than response rate (makes a better predictor)

# Correlation matrix
df_corr = df.astype({"response": int}).merge(pisa_reading, how='left', on='country')
df_corr.groupby("country", observed=True)[
    ["response"] + ['Average'] + numeric_features + sc_measures
].mean().corr()

# %%

# cut_countries_east = ["China", "Indonesia", "Malaysia", "Thailand"]
# cut_countries_middle = ["Israel", "Turkey", "Kazakhstan", "Morocco", "UAE"]

# reading_west = reading_wallets.set_index("country").drop(
#     cut_countries_east + cut_countries_middle
# )

# Predictive value of wallet return rate for PISA scores.
# One thing that's interesting is that it appears that the survey data does not capture
# info about social capital that is important to education. But wallet reporting rates
# do.

import statsmodels.api as sm

survey_measures = [
    "general_trust",
    "GPS_trust",
    "general_morality",
    "MFQ_genmorality",
    "civic_cooperation",
    "GPS_posrecip",
    "GPS_altruism",
    "stranger1",
]

for measure in survey_measures:
    fdsa = reading_wallets.merge(
        df[["country", measure]].drop_duplicates(),
        how="left",
        on="country",
    ).dropna()

    # fdsa = df.astype({"response": float}).groupby("country")[
    #     ["response", measure, "log_gdp"]
    # ].mean().dropna()

    rewq = fdsa.set_index("country")

    y = rewq["Average"]
    # y = rewq["log_gdp"]
    X = rewq[[measure, "response"]]
    X = (X - X.mean()) / X.std()
    X = sm.add_constant(X)
    results = sm.OLS(y, X).fit(cov_type="HC1")
    print(results.summary())

# %%
# ML techniques with average PISA per country. Kinda pointless because trying to
# predict a category (response) with survey data.

df_with_pisa = df.merge(pisa_reading, how="left", on="country")

numeric_features = [
    "general_trust",
    "GPS_trust",
    "general_morality",
    "MFQ_genmorality",
    "civic_cooperation",
    "GPS_posrecip",
    "GPS_altruism",
    "stranger1",
    "Average",
    "log_gdp",
    "log_tfp",
    "gee",
    "letter_grading",
]
numeric_transformer = Pipeline(
    steps=[("imputer", SimpleImputer(strategy="mean")), ("scaler", StandardScaler())]
)

# Categorical feature pipeline
categorical_features = [
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
categorical_transformer = Pipeline(
    steps=[
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ("selector", SelectPercentile(chi2, percentile=50)),
    ]
)

# target feature
target = df_with_pisa["response"].cat.codes

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

clf = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        # ("classifier", LinearSVC()),
        # ("classifier", SVC(random_state=123)),
        # ("classifier", SVC(C=1.5, random_state=123, class_weight="balanced")),
        # ("classifier", KNeighborsClassifier(n_neighbors=80))
        # ("classifier", LogisticRegression(max_iter=10000, random_state=123))
        # ("classifier", RidgeClassifier(alpha=5))
        # ("classifier", GaussianNB())
        # ("classifier", tree.DecisionTreeClassifier())
        # ("classifier", RandomForestClassifier(max_depth=8, random_state=0)),
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    df_with_pisa.drop(columns="response"),
    df_with_pisa["response"],
    test_size=0.2,
    random_state=23,
)

# Curious if weighting the observations according to the inverse frequency of "cond"
# will help the model...
cond_counts = X_train["cond"].value_counts(normalize=True)
weights = X_train["cond"].map(lambda x: 1 / cond_counts[x])

clf.fit(X_train, y_train)#, classifier__sample_weight=weights)
print("model score: %.3f" % clf.score(X_test, y_test))
clf

# %%
