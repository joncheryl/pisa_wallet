# John Sherrill
# June 2025
"""
Reading mean PISA scores in pdf from
https://www.oecd.org/en/publications/pisa-2022-results-volume-i_53f23881-en.html
"""

# %%
import pandas as pd
import tabula

############################
# Math scores.
############################
pisa_2022_math = pd.concat(tabula.read_pdf("data/PISA_2022_report.pdf", pages=[54, 55]))
pisa_2022_math.columns = ["mean_score", "country", "subject"]
pisa_2022_math["subject"] = "math"

pisa_2022_math["mean_score"] = pd.to_numeric(
    pisa_2022_math["mean_score"], errors="coerce"
)

############################
# Reading scores.
############################
pisa_2022_reading = pd.concat(
    tabula.read_pdf("data/PISA_2022_report.pdf", pages=[56, 57])
)
pisa_2022_reading.columns = ["mean_score", "country", "subject"]
pisa_2022_reading["subject"] = "reading"

pisa_2022_reading["mean_score"] = pd.to_numeric(
    pisa_2022_reading["mean_score"], errors="coerce"
)

############################
# Science scores.
############################
pisa_2022_science_one = pd.concat(
    tabula.read_pdf("data/PISA_2022_report.pdf", pages=[58])
)
pisa_2022_science_two = pd.concat(
    tabula.read_pdf("data/PISA_2022_report.pdf", pages=[59])
).iloc[:, [1, 2, 3]]
pisa_2022_science_one.columns = ["mean_score", "country", "subject"]
pisa_2022_science_two.columns = ["mean_score", "country", "subject"]
pisa_2022_science = pd.concat([pisa_2022_science_one, pisa_2022_science_two])
pisa_2022_science.columns = ["mean_score", "country", "subject"]
pisa_2022_science["subject"] = "science"

pisa_2022_science["mean_score"] = pd.to_numeric(
    pisa_2022_science["mean_score"], errors="coerce"
)

############################
# Put it all together.
############################
pisa_2022 = (
    pd.concat([pisa_2022_math, pisa_2022_reading, pisa_2022_science], axis=0)
    .dropna(subset="mean_score")
    .replace(
        {
            "Russian Federation": "Russia",
            "United Arab Emirates": "UAE",
            "United Kingdom*": "UK",
            "United States*": "USA",
            "Australia*": "Australia",
            "Canada*": "Canada",
            "China*": "China",
            "Denmark*": "Denmark",
            "Ghana*": "Ghana",
            "India*": "India",
            "Kenya*": "Kenya",
            "Netherlands*": "Netherlands",
            "New Zealand*": "New Zealand",
            "Russia*": "Russia",
            "South Africa*": "South Africa",
            "Turkey*": "Turkey",
        }
    )
)

# %%
pisa_2022.to_csv("data/pisa_data.csv", index=False)
