# Cookie Cats: A/B Testing for Mobile Game Retention

### Business Analyst Project \| Prepared by Vivek Kumar

**Author:** Vivek Kumar\
**Project Type:** Product Analytics · Business Analysis · A/B Testing\
**Domain:** Mobile Gaming / Product Growth\
**Primary Tools:** Python, Pandas, NumPy, Matplotlib, Seaborn, SciPy,
Statsmodels\
**Dataset:** `cookie_cats.csv`\
**Notebook:** `cookie_cats_ab_testing.ipynb`

------------------------------------------------------------------------

## 1. Executive Summary

This project analyzes an A/B experiment from **Cookie Cats**, a mobile
puzzle game, to understand whether moving a progression gate from level
30 to level 40 changes player engagement and retention.

Players were assigned to one of two versions:

-   **`gate_30` --- Control:** the progression gate is at level 30.
-   **`gate_40` --- Treatment:** the progression gate is at level 40.

The analysis examines three outcomes: total game rounds played, one-day
retention, and seven-day retention. Exploratory data analysis (EDA) is
followed by assumption checks and statistical hypothesis tests.

### Key results

  --------------------------------------------------------------------------------
  Metric              `gate_30`        `gate_40` Test result      Interpretation
                                                                  at α = 0.05
  ------------ ---------------- ---------------- ---------------- ----------------
  Players                44,700           45,489 Descriptive      Groups are
                                                 count            similar in size

  Average game            52.46            51.30 Mann--Whitney U, Not
  rounds                                         p ≈ 0.0502       statistically
                                                                  significant at
                                                                  5%

  Median game                17               16 Descriptive      Median is
  rounds                                         statistic        slightly higher
                                                                  for `gate_30`

  Day-1                  44.82%           44.23% Two-proportion   Not
  retention                                      z-test, p ≈      statistically
                                                 0.0744           significant at
                                                                  5%

  Day-7                  19.02%           18.20% Two-proportion   Statistically
  retention                                      z-test, p ≈      significant;
                                                 0.0015           `gate_30` is
                                                                  higher
  --------------------------------------------------------------------------------

**Business takeaway:** In this experiment, moving the gate to level 40
did not demonstrate an improvement in the measured outcomes. The
level-30 group had significantly higher day-7 retention. The evidence
supports considering the level-30 gate as the preferred option, while
reviewing experiment design and additional product metrics before making
a final rollout decision.

> Statistical note: p-values are interpreted against a 5% significance
> level. A result that is not statistically significant is not proof
> that the groups are identical. The reported values are the values
> documented in the analysis notebook; rerun the notebook to reproduce
> results in your environment.

------------------------------------------------------------------------

## 2. Business Problem

Product teams regularly change onboarding, difficulty, rewards, and
progression mechanics to improve engagement. However, a change that
appears reasonable can unintentionally reduce the number of players who
return.

The business question is:

**Does moving the progression gate from level 30 to level 40 improve
player engagement or retention?**

The team needs evidence to decide whether to keep the gate at level 30,
move it to level 40, or run a further experiment.

### Why this matters to the business

-   **Retention:** Returning players have more opportunities to continue
    playing and form a habit.
-   **Engagement:** Game rounds provide a simple measure of how much
    players play.
-   **Product decisions:** Controlled experiments reduce reliance on
    intuition alone.
-   **Player experience:** Gate placement can affect pacing, challenge,
    and the perceived reward structure.
-   **Resource allocation:** Teams can prioritize changes supported by
    evidence.

------------------------------------------------------------------------

## 3. Objectives

1.  Load and inspect the experiment dataset.
2.  Understand the dataset's structure, data types, and quality.
3.  Check missing values and experimental group sizes.
4.  Compare engagement and retention metrics between groups.
5.  Use charts to understand distributions, outliers, and group
    differences.
6.  Check assumptions relevant to statistical test selection.
7.  Test whether observed differences are statistically significant.
8.  Translate the statistical findings into a practical product
    recommendation.
9.  Document limitations and follow-up analyses.

------------------------------------------------------------------------

## 4. Dataset Description

The dataset contains **90,189 player records and five columns**.

  -----------------------------------------------------------------------
  Column                  Meaning                 Data type / role
  ----------------------- ----------------------- -----------------------
  `userid`                Unique identifier for a Integer identifier
                          player                  

  `version`               Experimental version    Categorical
                          assigned to the player  
                          (`gate_30` or           
                          `gate_40`)              

  `sum_gamerounds`        Total number of game    Numeric count
                          rounds played during    
                          the observed period     

  `retention_1`           Whether the player      Boolean outcome
                          returned one day after  
                          installation            

  `retention_7`           Whether the player      Boolean outcome
                          returned seven days     
                          after installation      
  -----------------------------------------------------------------------

### Outcome definitions

-   **Game rounds:** A measure of player activity. It is a count and can
    have a highly skewed distribution.
-   **Day-1 retention:** The proportion of players who returned one day
    after installation.
-   **Day-7 retention:** The proportion of players who returned seven
    days after installation.

Retention rates are calculated as the mean of the Boolean column: `True`
is treated as 1 and `False` as 0.

### Experimental groups

-   `gate_30`: control version.
-   `gate_40`: treatment version.

The experiment's intended comparison is between the outcomes of these
two groups. The analysis assumes the assigned versions reflect the
experiment setup represented in the dataset.

------------------------------------------------------------------------

## 5. Tools and Requirements

### Software

-   Python 3.9 or later recommended
-   Jupyter Notebook, JupyterLab, or Google Colab
-   CSV dataset file available locally

### Python libraries

  Library         Purpose
  --------------- ---------------------------------------------------------
  `pandas`        Load, clean, group, summarize, and reshape tabular data
  `numpy`         Numerical operations
  `matplotlib`    Plotting and chart customization
  `seaborn`       Statistical charts and visual exploration
  `scipy`         Shapiro--Wilk, Levene, and Mann--Whitney U tests
  `statsmodels`   Two-proportion z-tests and confidence intervals

### Install dependencies

``` bash
pip install pandas numpy matplotlib seaborn scipy statsmodels jupyter
```

### Files expected in the project folder

``` text
cookie-cats-ab-testing/
├── README.md
├── cookie_cats.csv
└── cookie_cats_ab_testing.ipynb
```

Keep the CSV in the same folder as the notebook unless you update the
path in the data-loading cell.

------------------------------------------------------------------------

## 6. How to Run the Project

1.  Download or clone the project files.

2.  Place `cookie_cats.csv` in the same directory as the notebook.

3.  Install the required packages.

4.  Open the notebook:

    ``` bash
    jupyter notebook
    ```

5.  Open `cookie_cats_ab_testing.ipynb`.

6.  Run the cells from top to bottom.

7.  Review the summary tables, plots, assumption checks, test results,
    and business conclusion.

In Google Colab, upload both the notebook and CSV, or mount the folder
containing the CSV and adjust the path if necessary.

------------------------------------------------------------------------

## 7. End-to-End EDA Process

EDA is used to understand the data before formal hypothesis testing. It
helps identify data-quality problems, understand the shape of variables,
compare groups, and select appropriate analytical methods.

### Step 1 --- Import libraries and load data

The notebook imports the data-analysis, visualization, and
statistical-testing libraries, then loads the CSV:

``` python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import shapiro, levene, mannwhitneyu
from statsmodels.stats.proportion import proportions_ztest
import statsmodels.stats.api as sms

df = pd.read_csv("cookie_cats.csv")
```

**Business purpose:** Establish a reproducible workflow and make the
experiment data available for inspection.

### Step 2 --- Preview the data

``` python
df.head()
```

This displays the first few records so the analyst can confirm column
names, example values, and the format of the experimental groups.

**What to check:** Are the expected columns present? Do the values look
sensible? Are the group labels consistent?

### Step 3 --- Check dataset dimensions

``` python
df.shape
```

**Observed result:** 90,189 rows and 5 columns.

**Interpretation:** The dataset has one record per player in the
supplied data. This is a substantial sample for comparing the two
experiment groups, although sample size alone does not guarantee that an
experiment is unbiased.

### Step 4 --- Inspect data types and structure

``` python
df.info()
```

This identifies the number of non-null values, data types, and
approximate memory usage.

**Interpretation:** `userid` and `sum_gamerounds` are numeric, `version`
is categorical in meaning, and the retention fields are Boolean
outcomes. Analysts should distinguish a numeric identifier from a
meaningful numeric measurement.

### Step 5 --- Review descriptive statistics

``` python
df.describe().T
```

This summarizes numeric variables using count, mean, standard deviation,
minimum, quartiles, and maximum.

**Key observation:** The notebook reports an overall average of
approximately 51.87 game rounds, a median of 16, and a maximum of
49,854.

**Interpretation:** The mean is much larger than the median, which
signals strong right-skew and unusually high values. Reporting only the
mean would hide the experience of a typical player. Median and
percentile summaries are useful alongside the mean.

### Step 6 --- Check missing values

``` python
df.isnull().sum()
```

**Observed result:** The notebook reports no missing observations.

**Interpretation:** No missing-value imputation is required for the
supplied columns. This does not, by itself, prove that every value is
valid; logical checks and duplicate checks can still be useful.

### Step 7 --- Review group allocation

``` python
df["version"].value_counts()
df["version"].value_counts(normalize=True)
```

**Observed counts:**

-   `gate_30`: 44,700 players
-   `gate_40`: 45,489 players

The groups are close in size. A small difference in group counts is not
inherently a problem, but the assignment mechanism and any exclusions
should be checked when validating an experiment.

### Step 8 --- Compare metrics by group

``` python
df.groupby("version").agg({
    "sum_gamerounds": ["mean", "median"],
    "retention_1": "mean",
    "retention_7": "mean"
})
```

This provides a first comparison before hypothesis testing.

  Metric                  `gate_30`   `gate_40`
  --------------------- ----------- -----------
  Average game rounds         52.46       51.30
  Median game rounds             17          16
  Day-1 retention            44.82%      44.23%
  Day-7 retention            19.02%      18.20%

**Interpretation:** The descriptive metrics favor `gate_30`, but these
differences must be evaluated with statistical tests. Descriptive
differences alone do not establish statistical significance or
causation.

### Step 9 --- Inspect quantiles

The notebook examines game-round quantiles, including the 1st, 5th,
25th, 50th, 75th, 95th, and 99th percentiles.

**Observed insight:** The median is 16 rounds, while the 99th percentile
is approximately 493 rounds. A small share of players play far more
rounds than most players.

**Business interpretation:** Player activity is heterogeneous. Typical
behavior and highly engaged players should be considered separately when
designing retention or engagement strategies.

------------------------------------------------------------------------

## 8. Chart-by-Chart Explanation

The notebook includes visualizations designed to answer different
analytical questions. The exact values shown in charts depend on the
dataset and the executed notebook cells.

### Chart 1 --- User Distribution by Version

**Chart type:** Count plot / bar chart\
**Purpose:** Compare the number of players assigned to each experimental
version.

**What is plotted:** - X-axis: `version` (`gate_30`, `gate_40`) -
Y-axis: number of players - Bar labels: group counts

**Observed result:** `gate_30` has 44,700 players and `gate_40` has
45,489 players.

**How to interpret it:** The bars should be similar in height. Similar
sample sizes make the groups easier to compare, but do not independently
prove randomization was implemented correctly.

**Business question answered:** Did the experiment produce similarly
sized groups?

### Chart 2 --- Retention Rates by Version

**Chart type:** Grouped bar chart\
**Purpose:** Compare day-1 and day-7 retention for both versions.

**What is plotted:** - X-axis: retention metric (`retention_1`,
`retention_7`) - Y-axis: retention rate - Color / legend: experimental
version - Labels: observed rates

**Observed result:** - Day-1 retention: 44.82% for `gate_30` versus
44.23% for `gate_40`. - Day-7 retention: 19.02% for `gate_30` versus
18.20% for `gate_40`.

**How to interpret it:** `gate_30` has higher observed retention for
both time horizons. The gap is more pronounced for day-7 retention. The
bar chart is descriptive; the two-proportion tests assess whether the
differences are statistically significant.

**Business question answered:** Which version has higher observed player
retention?

### Chart 3 --- Average Game Rounds by Version

**Chart type:** Bar chart\
**Purpose:** Compare average player activity.

**What is plotted:** - X-axis: experiment version - Y-axis: average
`sum_gamerounds` - Labels: group means

**Observed result:** The average is approximately 52.46 for `gate_30`
and 51.30 for `gate_40`.

**How to interpret it:** Average rounds are slightly higher in the
control group. Because the distribution has extreme values and is
strongly right-skewed, the mean should not be interpreted in isolation.
The notebook uses the Mann--Whitney U test for formal comparison.

**Business question answered:** Does the later gate appear to increase
average game activity?

### Chart 4 --- Distribution of Total Game Rounds

**Chart type:** Histogram with density curves\
**Purpose:** Examine the shape of player activity and compare the two
versions.

**What is plotted:** - X-axis: `sum_gamerounds` - Y-axis: density -
Color / legend: experiment version - Display range: limited to the 99th
percentile for readability

**Observed pattern:** The distribution is strongly right-skewed. Most
players complete relatively few rounds, while a small number complete
many more.

**Why the axis is limited:** Very large observations can compress the
majority of the histogram into a narrow region. Limiting the displayed
x-axis makes the main distribution easier to see. It is a display
choice, not evidence that the extreme observations were deleted from the
dataset.

**Business question answered:** Is average game activity representative
of most players, and are parametric-test assumptions plausible?

### Chart 5 --- Box Plot of Game Rounds by Version

**Chart type:** Box plot\
**Purpose:** Compare the median, interquartile range, spread, and
potential extreme values for game rounds.

**How to read it:** - The line inside the box represents the median. -
The box represents the middle 50% of observations (interquartile
range). - Whiskers show the range defined by the plotting convention. -
Points beyond the whiskers may be displayed as potential outliers.

**Observed insight:** The data contain extreme high values and a long
right tail. The median is a more robust summary of a typical player than
the mean.

**Business question answered:** Do the groups have similar
distributions, and are extreme observations influencing the mean?

### Chart 6 --- Game-Round Quantiles

**Chart/table type:** Quantile summary\
**Purpose:** Quantify the distribution at selected percentiles.

**Important values:** The median is 16 rounds and the 99th percentile is
approximately 493 rounds.

**Interpretation:** Most players have activity well below the maximum.
Quantiles help communicate the distribution without allowing a handful
of very active users to dominate the summary.

**Business question answered:** How does player activity vary from
typical to highly engaged users?

### Chart 7 --- Confidence Interval for Average Game Rounds

**Chart type:** Point estimate with error bars\
**Purpose:** Display each group's mean and a 95% confidence interval for
the mean.

**What is plotted:** - X-axis: version - Y-axis: average game rounds -
Point: estimated group mean - Error bars: confidence interval

**Approximate intervals documented in the notebook:** - `gate_30`: 50.08
to 54.84 - `gate_40`: 50.35 to 52.25

**How to interpret it:** The intervals overlap. However, overlap between
two separate confidence intervals is not itself a formal test of the
difference between group means. The notebook's primary game-round test
is the Mann--Whitney U test, which has p ≈ 0.0502.

**Important caveat:** These are confidence intervals for group means,
while the chosen hypothesis test is non-parametric. A confidence
interval for the difference in means or a bootstrap interval could
complement this analysis.

------------------------------------------------------------------------

## 9. Hypothesis Testing Framework

### Significance level

The notebook uses **α = 0.05**.

-   If p-value \< 0.05: reject the null hypothesis at the 5% level.
-   If p-value ≥ 0.05: fail to reject the null hypothesis at the 5%
    level.

A p-value is not the probability that the null hypothesis is true.
Statistical significance also does not automatically imply that an
effect is large or commercially important.

### Game rounds

-   **Null hypothesis (H₀):** The game-round distributions do not differ
    between `gate_30` and `gate_40` in the way tested.
-   **Alternative hypothesis (H₁):** The game-round distributions
    differ.

### Day-1 retention

-   **H₀:** Day-1 retention proportions are equal between the two
    versions.
-   **H₁:** Day-1 retention proportions differ between the two versions.

### Day-7 retention

-   **H₀:** Day-7 retention proportions are equal between the two
    versions.
-   **H₁:** Day-7 retention proportions differ between the two versions.

The notebook uses two-sided tests for these comparisons. If a team
intends to test a directional hypothesis, that direction should be
specified before examining the results.

------------------------------------------------------------------------

## 10. Assumption Checks and Test Selection

### Shapiro--Wilk normality test

**Purpose:** Assess whether the game-round data are consistent with a
normal distribution.

The notebook samples 5,000 observations from each version for the
Shapiro--Wilk test.

-   H₀: the sampled data follow a normal distribution.
-   H₁: the sampled data do not follow a normal distribution.

The notebook reports p-values below 0.05 for both groups, so normality
is not supported. This is consistent with the strongly right-skewed
histogram and extreme values.

**Caution:** With large samples, normality tests can detect small
deviations. The plot and the nature of the count data should be
considered alongside the p-value.

### Levene's test for equality of variances

**Purpose:** Test whether the groups have equal variances.

-   H₀: group variances are equal.
-   H₁: group variances differ.

The notebook reports a p-value above 0.05. This means there is
insufficient evidence to conclude that the variances differ; it does not
prove that the variances are exactly equal.

### Why use the Mann--Whitney U test?

The game-round data are highly skewed, with extreme values, and the
normality check does not support a normal distribution. The notebook
therefore uses the **Mann--Whitney U test** rather than relying on a
standard independent-samples t-test.

The Mann--Whitney U test compares the relative ranks of observations
between two independent groups. It is often described as a test of
distributions; it should not automatically be interpreted as a test of
medians unless additional assumptions about the distributions are
appropriate.

### Why use a two-proportion z-test for retention?

`retention_1` and `retention_7` are binary outcomes. A two-proportion
z-test compares the proportion of players who returned in each group.

This directly addresses the retention question and is appropriate for
comparing the two large independent groups, subject to the experiment's
design and independence assumptions.

------------------------------------------------------------------------

## 11. A/B Test Results in Detail

### Test 1 --- Total game rounds

**Test:** Mann--Whitney U\
**Reported p-value:** approximately 0.0502

Because p ≈ 0.0502 is slightly greater than 0.05, the result is not
statistically significant at the 5% level. We fail to reject the null
hypothesis.

**Business interpretation:** The data do not establish a statistically
significant difference in game-round activity at the chosen threshold.
The observed average is slightly higher in `gate_30`, but the result is
close to the threshold and should not be described as conclusive
evidence of a difference.

### Test 2 --- Day-1 retention

**Test:** Two-proportion z-test\
**Reported p-value:** approximately 0.0744

Because p \> 0.05, the difference is not statistically significant at
the 5% level.

**Business interpretation:** `gate_30` has a slightly higher observed
day-1 retention rate, but the experiment does not establish a
statistically significant difference in day-1 retention at α = 0.05.

### Test 3 --- Day-7 retention

**Test:** Two-proportion z-test\
**Reported p-value:** approximately 0.0015

Because p \< 0.05, reject the null hypothesis of equal day-7 retention
proportions at the 5% level.

**Business interpretation:** Day-7 retention is statistically
significantly higher for `gate_30` than for `gate_40` in the supplied
experiment. This is the clearest finding in the analysis.

### Results summary

  -----------------------------------------------------------------------
  Outcome                 Statistical conclusion  Product implication
  ----------------------- ----------------------- -----------------------
  Game rounds             Not significant at 5%   No conclusive evidence
                                                  of a difference from
                                                  this test

  Day-1 retention         Not significant at 5%   No conclusive evidence
                                                  of a day-1 retention
                                                  improvement

  Day-7 retention         Significant; `gate_30`  The move to level 40
                          is higher               did not improve the
                                                  measured long-term
                                                  retention outcome
  -----------------------------------------------------------------------

------------------------------------------------------------------------

## 12. Business Recommendations

### Recommendation 1 --- Do not move the gate to level 40 solely to improve retention

The experiment does not show an improvement in day-1 retention or game
rounds, and day-7 retention is significantly higher in the level-30
group.

### Recommendation 2 --- Prioritize day-7 retention as a key success metric

Day-7 retention provides a longer-term view of whether players return.
It should be monitored alongside day-1 retention, game rounds, and other
product outcomes.

### Recommendation 3 --- Review player progression and friction

Investigate where players stop progressing, whether the gate changes
perceived difficulty, and whether players experience frustration or loss
of motivation. These are hypotheses for further analysis, not causes
proven by this dataset.

### Recommendation 4 --- Validate the experiment before broad rollout

Confirm random assignment, check for sample-ratio mismatch, ensure the
observation windows are consistent, and verify that no users were
duplicated or improperly excluded.

### Recommendation 5 --- Consider follow-up experiments

Test alternative gate positions, difficulty curves, or reward
structures. Define the primary metric and decision threshold before
launching the next experiment.

### Recommendation 6 --- Evaluate practical significance

A statistically significant result should also be translated into
absolute percentage-point differences, relative lift, confidence
intervals, and potential business impact. Statistical significance alone
does not establish the financial value of a change.

------------------------------------------------------------------------

## 13. Limitations and Further Work

1.  **Limited variables:** The supplied dataset does not include session
    duration, purchase behavior, player demographics, progression
    details, or detailed event logs.
2.  **Causal interpretation depends on design:** A/B tests can support
    causal conclusions when assignment, exposure, and measurement are
    valid. The dataset alone cannot verify every design detail.
3.  **Extreme game-round values:** A small number of highly active
    players affect the mean and make the distribution strongly skewed.
4.  **Multiple outcomes:** Three outcomes are tested. If multiple
    hypotheses are treated as one family of confirmatory tests, a
    multiple-comparison adjustment may be appropriate. The reported
    p-values are interpreted as documented in the notebook without such
    an adjustment.
5.  **Confidence interval interpretation:** The notebook's separate
    intervals are for group means; overlapping intervals are not a
    substitute for a formal test of the difference.
6.  **No full uncertainty analysis for retention:** The README
    summarizes the reported proportion-test p-values. A
    production-quality analysis should also report confidence intervals
    for absolute and relative retention differences.
7.  **No segmentation:** Results are not broken down by acquisition
    channel, platform, geography, or new-player behavior because those
    fields are not included.
8.  **No long-term or revenue outcomes:** The analysis does not
    determine the effect on monetization, lifetime value, or retention
    beyond the measured periods.

### Recommended improvements

-   Report confidence intervals for the difference in day-1 and day-7
    retention.
-   Calculate absolute percentage-point change and relative lift.
-   Add a sample-ratio mismatch check.
-   Verify unique `userid` values and valid category labels.
-   Use bootstrap intervals for skewed engagement metrics.
-   Consider a pre-registered primary metric and correction for multiple
    comparisons.
-   Analyze downstream metrics such as session frequency, progression,
    purchases, and long-term retention if data are available.

------------------------------------------------------------------------

## 14. Business Analyst Skills Demonstrated

This project demonstrates practical skills relevant to Business Analyst,
Product Analyst, and Data Analyst roles:

-   Translating a product question into measurable hypotheses.
-   Defining KPIs and comparing control versus treatment groups.
-   Data loading, inspection, validation, and exploratory analysis.
-   Descriptive statistics and quantile interpretation.
-   Chart selection and communicating visual findings.
-   Understanding statistical assumptions and choosing suitable tests.
-   Interpreting p-values without overstating conclusions.
-   Translating results into business recommendations.
-   Documenting limitations, risks, and follow-up actions.

### Example interview explanation

> "I analyzed an A/B test for Cookie Cats to determine whether moving a
> progression gate from level 30 to level 40 affected engagement and
> retention. I used Python, Pandas, Seaborn, SciPy, and Statsmodels to
> inspect the data, compare group-level KPIs, visualize the distribution
> of game rounds, and test the observed differences. The game-round data
> were strongly right-skewed, so I used the Mann--Whitney U test for
> that outcome and two-proportion z-tests for day-1 and day-7 retention.
> The reported results did not show a statistically significant
> difference in game rounds or day-1 retention, but day-7 retention was
> significantly higher in the level-30 group. My recommendation was not
> to move the gate to level 40 solely to improve retention and to
> validate the experiment and investigate player progression before
> deciding on a broader rollout."

Use this as a framework and make sure you can explain each metric, test,
and limitation in your own words.

------------------------------------------------------------------------

## 15. Suggested Resume Bullet

Use only if this accurately reflects the work you completed:

**Cookie Cats A/B Testing --- Product & Retention Analytics** - Analyzed
90K+ player records using Python, Pandas, Seaborn, SciPy, and
Statsmodels to evaluate the impact of game progression-gate placement on
player engagement and retention. - Conducted EDA, distribution and
assumption checks, Mann--Whitney U testing, and two-proportion z-tests
to compare game rounds and day-1/day-7 retention across experimental
groups. - Translated statistical results into product recommendations,
identifying significantly higher day-7 retention for the level-30 gate
in the analyzed experiment.

------------------------------------------------------------------------

## 16. Final Conclusion

The Cookie Cats experiment illustrates how business decisions can be
improved by combining exploratory analysis with statistical testing. The
groups are similar in size, but the game-round distribution is strongly
right-skewed. The reported tests do not establish a statistically
significant difference in game rounds or day-1 retention at the 5%
level. Day-7 retention is significantly higher in `gate_30`.

**Final recommendation:** Based on the measured outcomes in this
dataset, do not move the progression gate to level 40 on the assumption
that it will improve retention. Keep the level-30 option as the leading
candidate for further validation, monitor key KPIs, and use follow-up
experiments to understand why the retention difference occurs.

------------------------------------------------------------------------

**Prepared by Vivek Kumar**\
*Business Analytics · Product Analytics · A/B Testing*
