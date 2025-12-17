# Tests for hypothesis testing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import pytest

data = pd.read_csv('data/processed_ames_data.csv')


# LotArea differs by demand? t-test / Mann-Whitney
def test_lot_area_by_demand():
    high_demand = data[data["HighDemand"] ==1]["LotArea"]
    low_demand = data[data["HighDemand"] ==0]["LotArea"]
    stat, p = stats.ttest_ind(high_demand, low_demand, equal_var = False)
    assert p < 0.05, "LotArea does not differ significantly by demand"
    # Visualization
    plt.figure(figsize = (8, 5))
    plt.hist(high_demand, alpha = 0.5, label='High Demand', bins = 30)
    plt.hist(low_demand, alpha = 0.5, label = "Low Demand", bins = 30)
    plt.legend()
    plt.title("Distribution of LotArea by Demand")
    plt.xlabel("LotArea")
    plt.ylabel("Frequency")
    plt.show()
    # Normality check
    _, p_high = stats.shapiro(high_demand)
    _, p_low = stats.shapiro(low_demand)
    if p_high < 0.05 or p_low < 0.05:
        print("LotArea is not normally distributed in one or both groups; consider non-parametric tests.")

    # Homogeneity of variance
    _, p_levene = stats.levene(high_demand, low_demand)
    if p_levene < 0.05:
        print("Variances are not equal; consider using Welch's t-test.")
    return stat, p
test_lot_area_by_demand()