# Benchmark Modification Summary

## Overview

| Domain | Task Ambiguity | Solution Error | Total |
|---|---:|---:|---:|
| Archeology | 0 | 1 | 1 |
| Astronomy | 5 | 0 | 5 |
| Environment | 4 | 0 | 4 |
| Legal | 2 | 2 | 4 |
| Wildfire | 2 | 1 | 3 |
| **Total** | **13** | **4** | **17** |

## Detailed Items

| Task ID | Issue Type | Problem | Resolution |
|---|---|---|---|
| archeology-hard-5 | Solution error | The official solution filters the wrong radiocarbon records, mishandles BP-style chronology, and rounds `Age_ky.1` before matching climate rows. | Correct answer: `66158.3691 -> 36815.2633`. |
| astronomy-easy-2 | Task ambiguity | The task does not specify rounding precision, while the official answer uses 2 decimal places. | Add: “Round your answer to 2 decimal places.” |
| astronomy-easy-3 | Evaluator metric issue | The target answer is a very small scientific-notation value (`7.95e-13`). The previous `numeric_exact` / `success` metric allowed a fixed absolute tolerance of `1e-6`, so numerically wrong answers near zero could still receive full credit even when produced by an incorrect solution path. | Change the answer type from `numeric_exact` to the new `numeric_scientific_exact` category. |
| astronomy-easy-4 | Task ambiguity | It is unclear whether peaks should be detected before or after restricting the data to 1960-2020. | Clarify that the data should first be restricted to 1960-2020 inclusive, then peaks detected with prominence 20 and distance 5. |
| astronomy-easy-6 | Task ambiguity | The task asks for two numerical rates but does not specify rounding precision. | Specify that both numbers should be rounded to 4 decimal places. |
| astronomy-hard-7 | Task ambiguity | The evaluation density window is inconsistent or underspecified. | Clarify the evaluation window as `Density: 2016-10-26 to 2016-10-29`. |
| astronomy-hard-7 | Evaluator metric issue | The target answer is a very small scientific-notation RMSE value (`1.211e-13`). Under the previous `numeric_exact` / `success` metric, the fixed `1e-6` absolute tolerance dominated the comparison and could mark substantially wrong density-forecasting results as correct. | Change the answer type from `numeric_exact` to the new `numeric_scientific_exact` category. |
| astronomy-hard-8 | Task ambiguity | The task requires train/test evaluation but does not specify the split. | Specify first 70% ordered data for training and remaining 30% for testing. |
| astronomy-hard-11 | Evaluator metric issue | The target answer is a very small scientific-notation RMSE value (`4.638e-13`). The previous `numeric_exact` / `success` metric could pass wrong answers solely because their absolute difference from the target was below `1e-6`, even if the relative error was large. | Change the answer type from `numeric_exact` to the new `numeric_scientific_exact` category. |
| environment-easy-6 | Task ambiguity | It is unclear whether the answer should be reported as a percentage value or a fraction. | Add: “Express the answer as a percentage value, not as a fraction.” |
| environment-hard-7 | Task ambiguity | It is unclear whether the rate difference should be reported as a percentage value or a fraction. | Add: “Express the answer as a percentage value, not as a fraction.” |
| environment-hard-8 | Task ambiguity | The task says “percentage,” but agents may still output a fraction without explicit scale guidance. | Add: “Express the answer as a percentage value, not as a fraction.” |
| environment-hard-19 | Task ambiguity | The task asks whether two years are “different or the same” but does not define what `True` means. | Specify: `True` means the two years are the same; `False` means they are different. |
| legal-hard-15 | Solution error | The official solution includes cross-state Micropolitan Statistical Areas, although the task asks for Metropolitan Statistical Areas. | Correct answer: `243377 -> 242682`. |
| legal-easy-19 | Task ambiguity | The denominator is ambiguous: all fraud reporters vs. fraud reporters with reported dollar loss. | Clarify denominator as fraud reporters with a reported dollar loss. |
| legal-hard-22 | Task ambiguity | The denominator is ambiguous: all reports vs. all identity theft reports. The task also has a typo: “Back Account.” | Clarify denominator as all identity theft reports whose Theft Type is Bank Account and Theft Subtype is New Accounts. |
| legal-hard-24 | Solution error | The official parser removes a legitimate comma inside the MSA name, causing the gold string to differ from the source data. | Correct answer: `Los Angeles-Long Beach-Anaheim CA Metropolitan Statistical Area -> Los Angeles-Long Beach-Anaheim, CA Metropolitan Statistical Area`. |
| wildfire-easy-9 | Solution error | The task says humidity less than 30%, but the official answer uses an inclusive `<= 30` threshold. | Correct answer: `-0.0059 -> -0.0053`. |
| wildfire-hard-16 | Task ambiguity | The task asks for chi-square statistic and p-value but does not specify rounding precision. | Specify that both numbers should be rounded to three decimal places. |
| wildfire-hard-20 | Task ambiguity | The task asks for a percentage to 2 decimals, but the official answer is a proportion with 4 decimals. | Rewrite as a proportion task and specify rounding to 4 decimal places. |

## Notes on Complex Cases 
### numeric_scientific_exact

The previous `numeric_exact` answer type uses the `success` metric. For floating-point answers, that metric includes a fixed absolute tolerance check: `abs(predicted - target) < 1e-6`. This is too permissive for scientific-notation targets around `1e-13`, because the tolerance is millions of times larger than the correct answer. As a result, answers such as `0`, `1e-12`, or even `1e-9` can be marked correct for astronomy density/RMSE tasks despite having large relative error and potentially coming from an incorrect computation path.

The new `numeric_scientific_exact` category evaluates scalar scientific-notation answers by relative error instead of a fixed absolute tolerance. The judging logic is:

If `target != 0`, compute `relative_error = abs(predicted - target) / abs(target)` and return success only when `relative_error <= 0.005`.

If `target == 0`, use a strict absolute fallback: `abs(predicted) <= 1e-15`.

Invalid values such as non-numeric text, percentages, `NaN`, `inf`, lists, and booleans are scored as incorrect. This keeps small floating-point/library differences acceptable while preventing near-zero but materially wrong answers from receiving full credit.

### archeology-hard-5
The official solution has two issues. First, the task says “in the Maltese dataset,” so the radiocarbon records should be filtered to Region == "Malta" and Culture == "Neolithic", not to all Homo sapiens records across the regional dataset. Second, the date values are BP-style ages: a smaller BP value is chronologically later. Among the northernmost Maltese Neolithic samples, 6005 is later than 6412, so the tie-break should choose date = 6005.
The official solution also rounds Age_ky.1 before finding the closest climate year, which collapses many different climate rows into the same artificial year and incorrectly takes the max aluminum across all of them. The closest climate row should be selected using the unrounded Age_ky.1 year. This gives the corrected answer: 36815.2633.

### legal-hard-15
The official solution is inconsistent with the task wording. The task specifically asks for cross-state Metropolitan Statistical Areas, but the official solution only checks whether the area spans multiple states and does not filter out Micropolitan Statistical Areas. Therefore, it incorrectly includes cross-state micropolitan areas in the total. The two incorrectly included micropolitan areas are:
LaGrange, GA-AL Micropolitan Statistical Area: 453
Lebanon-Claremont, NH-VT Micropolitan Statistical Area: 242
These add 695 extra reports.
Correct Solution Path
The correct solution should:
Load the 2024 Identity Theft MSA data; Keep only rows whose area name contains Metropolitan Statistical Area; Exclude rows whose area name contains Micropolitan Statistical Area; Among those rows, keep only cross-state areas; Sum # of Reports.
The official answer is 243377, but under the task wording the corrected answer should be: 242682

### widefire-easy-9
The task says “humidity less than 30%,” which implies a strict threshold: avrh_mean < 30. However, the official gold answer -0.0059 matches an inclusive threshold, avrh_mean <= 30. Since the dataset contains rows with avrh_mean == 30, the two interpretations produce different results. The correct answer under the stated task wording should be computed with avrh_mean < 30.

