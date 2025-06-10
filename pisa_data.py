# John Sherrill
# June 2025
"""
Reading mean PISA scores in pdf from
https://www.oecd.org/en/publications/pisa-2022-results-volume-i_53f23881-en.html
"""

import re
import pandas as pd
from tabula.io import read_pdf

def read_pisa(year, subject, page_list):
    """
    Read the PISA pdf for a general year and subject.
    Of course I do this and then realize the format for the tables in the pdfs changed
    between 2015 and 2018 so this function only applies to 2018 and 2022. :(((
    """
    pages = [
        read_pdf("data/PISA_" + str(year) + "_report.pdf", pages=[page])
        for page in page_list
    ]

    # Get each page into a dataframe.
    pages = [pd.concat(list(page)) for page in pages]

    # Drop useless rows and columns.
    pages = [df.loc[:, df.isna().mean() <= 0.5] for df in pages]
    pages = [df.loc[df.isna().mean(axis=1) <= 0.5] for df in pages]

    # Assuming at this point that the first two rows are what we're looking for.
    pages = [df.iloc[:, [0, 1]] for df in pages]
    for page in pages:
        page.columns = ["pisa_score", "country"]
    df = pd.concat(pages)

    df["year"] = year
    df["subject"] = subject
    df["pisa_score"] = pd.to_numeric(df["pisa_score"], errors="coerce")

    return df.dropna().astype({"pisa_score": int})


subject_pages = [
    (2022, "math", [54, 55]),
    (2022, "reading", [56, 57]),
    (2022, "science", [58, 59]),
    (2018, "math", [61, 62]),
    (2018, "reading", [59, 60]),
    (2018, "science", [63, 64]),
    (2015, "math", [179]),
    (2015, "reading", [151]),
    (2015, "science", [69]),
]

pisa = pd.concat(
    [read_pisa(year, subject, pages) for (year, subject, pages) in subject_pages]
)

# Remove trailing asterisks and .
pisa["country"] = pisa["country"].map(
    lambda x: re.sub(r"[*1]+$", "", x) if isinstance(x, str) else x
)

# Conform naming to be consistent with other dataframes in project.
pisa = pisa.replace(
    {
        "Russian Federation": "Russia",
        "United Arab Emirates": "UAE",
        "United Kingdom": "UK",
        "United States": "USA",
        "FYROM": "North Macedonia",
        "Viet Nam": "Vietnam",
        "Ukrainian regions (18 of 27)": "Ukraine",
        "Türkiye": "Turkey",
        "Korea": "South Korea",
        "Slovak Republic": "Slovakia",
        "Brunei Darussalam": "Brunei",
        "Chinese Taipei": "Taiwan"
    }
)

########################################################################
# Alternatively downloading with pandas from wikipedia which as of June 9, 2025 is
# more comprehensive somehow.
########################################################################

pisa_tables = pd.read_html(
    "https://en.wikipedia.org/wiki/Programme_for_International_Student_Assessment#Rankings_comparison_2000%E2%80%932022"  # pylint: disable=line-too-long
)[5:8]

pisa_tables = [table.set_index(table.columns[0]) for table in pisa_tables]
for table in pisa_tables:
    table.index.name = "country"

pisa_wiki = pd.concat(pisa_tables, axis=1)

pisa_wiki.columns.names = ["subject", "year", "score/rank"]

# Drop rankings.
pisa_wiki = pisa_wiki.xs("Score", level="score/rank", axis=1)

# Drop undesired years and rename a couple of them.
pisa_wiki = pisa_wiki.loc[
    :, pisa_wiki.columns.get_level_values("year").isin(["2022[33]", "2018[34]", "2015"])
]
pisa_wiki.columns = pisa_wiki.columns.set_levels(
    pisa_wiki.columns.levels[1]
    .to_series()
    .replace({"2022[33]": "2022", "2018[34]": "2018"})
    .values,
    level="year",
)

# Cleanup
pisa_wiki = (
    pisa_wiki.replace({"—": pd.NA})
    .stack(level=["subject", "year"], future_stack=True)
    .reset_index()
    .rename(columns={0: "pisa_score"})
    .dropna()
    .astype({"year": int, "pisa_score": int})
    .replace({"Mathematics": "math", "Reading": "reading", "Science": "science"})
    .replace(
        {
            "China B-S-J-G[a]": "B-S-J-G (China)",
            "China B-S-J-Z[b]": "B-S-J-Z (China)",
            "Ukraine[d]": "Ukraine",
            "United States": "USA",
            "United Kingdom": "UK",
            "United Arab Emirates": "UAE",
            "Macedonia": "North Macedonia",
            "Hong Kong": "Hong Kong (China)",
            "Macau": "Macao (China)",
            "Azerbaijan Baku": "Baku (Azerbaijan)",
            "Argentina CABA[c]": "CABA (Argentina)"
        }
    )
)

pisa_wiki.to_csv("data/pisa_data.csv", index=False)
