# Instamart Customer Intelligence & Support Copilot

## Overview

This repository implements all four parts of the capstone using the assignment's fictional synthetic data: Part A data foundation, Part B customer segments and lapse prediction, Part C guarded support-ticket triage, and Part D policy retrieval and a simulated refund agent. The supplied `generate_data.py` and `mock_llm.py` files are included unchanged.

All money amounts in this README are in INR. All data is synthetic and fictional.

## Reproduce

Run these commands from the repository root in Windows PowerShell. Python 3.11 or newer is required.

1. Create an isolated environment:

   ```powershell
   python -m venv .venv
   ```

2. Activate it:

   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

3. Install the pinned libraries recorded in `requirements.txt`:

   ```powershell
   python -m pip install -r requirements.txt
   ```

4. Generate the deterministic raw data:

   ```powershell
   python generate_data.py
   ```

   Expected final line: `generated: customers=1200 products=40 orders_raw=13084 tickets=164`.

5. Audit, clean, and store the data, then run all seven SQL queries:

   ```powershell
   python part_a.py
   ```

6. Check the Part A reference checkpoints:

   ```powershell
   python validate_part_a.py
   ```

   Expected output: `ALL CHECKPOINTS PASSED`.

7. Build segments, tune and evaluate the lapse models, and write the Part B outputs:

   ```powershell
   python part_b.py
   ```

8. Run the guarded ticket-classification pipeline and write both triage CSVs:

   ```powershell
   python part_c.py
   ```

9. Evaluate policy retrieval, run the refund scenarios, and write agent checkpoints and logs:

   ```powershell
   python part_d.py
   ```

`requirements.txt` was produced with `python -m pip freeze` after installing pandas, NumPy, scikit-learn, and joblib. The environment is isolated in `.venv/`, which is excluded from Git.

## Part A

### Raw data audit

The three raw tables have these shapes and column-level types and missing-value counts. Percentages are of each table's rows; missing percentages are rounded to two decimal places. Pandas' default CSV NA parsing is used for this audit. The raw tickets table is not used by Part A; when it is read in a later part, it must use `keep_default_na=False` so empty ticket text remains an empty string.

| Table | Shape | Column | Raw dtype | Missing count | Missing % |
|---|---:|---|---|---:|---:|
| customers | 1,200 × 5 | customer_id | str | 0 | 0.00% |
| customers | 1,200 × 5 | city | str | 0 | 0.00% |
| customers | 1,200 × 5 | signup_date | str | 0 | 0.00% |
| customers | 1,200 × 5 | age | float64 | 61 | 5.08% |
| customers | 1,200 × 5 | gender | str | 30 | 2.50% |
| products | 40 × 4 | product_id | str | 0 | 0.00% |
| products | 40 × 4 | product_name | str | 0 | 0.00% |
| products | 40 × 4 | category | str | 0 | 0.00% |
| products | 40 × 4 | price_inr | int64 | 0 | 0.00% |
| orders | 13,084 × 9 | order_id | str | 0 | 0.00% |
| orders | 13,084 × 9 | customer_id | str | 0 | 0.00% |
| orders | 13,084 × 9 | product_id | str | 0 | 0.00% |
| orders | 13,084 × 9 | quantity | int64 | 0 | 0.00% |
| orders | 13,084 × 9 | order_ts | str | 0 | 0.00% |
| orders | 13,084 × 9 | status | str | 0 | 0.00% |
| orders | 13,084 × 9 | discount_pct | int64 | 0 | 0.00% |
| orders | 13,084 × 9 | order_amount | str | 53 | 0.41% |
| orders | 13,084 × 9 | delivery_minutes | float64 | 2,541 | 19.42% |

| Raw audit check | Result |
|---|---:|
| Order rows | 13,084 |
| Exact duplicate order rows | 119 |
| Pending order rows | 656 |
| Distinct order IDs | 12,309 |
| Amounts that do not convert to a number | 120 |
| Negative delivery times | 106 |
| Blank delivery times | 2,541 |
| Blank customer ages | 61 |
| Customer ages outside 10–100 | 6 |
| Distinct raw city spellings | 24 |
| Distinct nonblank raw gender spellings | 10 |
| Blank genders | 30 |

The amount audit has 53 CSV missing values, while numeric conversion flags 120 rows because it also catches nonnumeric text values.

### Cleaning rules and effects

| Table | Rule | Values or rows affected |
|---|---|---:|
| customers | Trim city whitespace and apply title case | 476 city strings standardized; 6 spellings remain |
| customers | Map case and abbreviations to `male`, `female`, or `unknown` | 950 gender values mapped; final values: female 610, male 560, unknown 30 |
| customers | Treat ages outside 10–100 as missing | 6 ages set missing |
| customers | Fill missing/invalid ages with the median of the customer's city (`groupby` + `transform`) | 67 ages filled; 0 missing remain; final ages range from 18 to 60 |
| customers | Parse signup date as datetime | 1,200 dates converted |
| products | Preserve supplied catalogue | All 40 rows retained unchanged |
| orders | Parse `order_ts` as datetime | 13,084 timestamps converted |
| orders | Drop exact duplicate rows | 119 rows removed |
| orders | Keep the latest `order_ts` record per `order_id` | 656 older rows removed; 12,309 records remain |
| orders | Convert amounts to numeric and fill missing/unparseable winning records from product price × quantity × (1 − discount / 100), rounded to two decimals | 112 amounts filled after latest-record selection; 0 missing amounts remain |
| orders | Set negative completed delivery durations to missing | 105 negative durations in the retained records set missing (106 appeared in the raw audit) |
| orders | Fill missing completed delivery durations with the median for that product category among completed orders | 774 durations filled; 0 missing among completed orders |
| orders | Leave cancelled deliveries missing | All 1,199 cancelled orders remain missing a delivery duration |

Clean customer city counts are Pune 227, Delhi 210, Bengaluru 197, Chennai 191, Hyderabad 191, and Mumbai 184. Clean orders contain 11,110 completed and 1,199 cancelled orders, no pending orders, and completed-order revenue totals **INR 2,998,668.80**.

The clean CSVs are `data/clean/customers.csv`, `data/clean/products.csv`, and `data/clean/orders.csv`. `data/instamart.db` contains SQLite tables named `customers`, `products`, and `orders`; `order_ts` is stored as text in `YYYY-MM-DD HH:MM:SS` format. The checkpoint script asserts more than eight reference values and prints exactly `ALL CHECKPOINTS PASSED` on success.

### SQL analysis

The seven statements are saved in [`sql/queries.sql`](sql/queries.sql). `part_a.py` reads each statement from that file and executes it with `pandas.read_sql_query` against `data/instamart.db`.

| Query | Result |
|---|---|
| Q1 — completed orders above INR 500; customer totals above INR 8,000 | 30 customers |
| Q2 — completed revenue by city and product category | 30 city/category rows; total INR 2,998,668.80; largest cell Bengaluru / Electronics at INR 233,909.45 |
| Q3 — products above average catalogue price | 11 products |
| Q4 — customers with no completed order | 25 customers |
| Q5 — customers above average completed spend | 550 customers; average customer spend INR 2,552.06 |
| Q6 — city spend with `RANK`, `DENSE_RANK`, and `ROW_NUMBER` | 8 rows have different `RANK` and `ROW_NUMBER` values |
| Q7 — daily completed revenue and cumulative total | 183 days; final cumulative revenue INR 2,998,668.80 |

In Q6, ties share the same `RANK`, while `ROW_NUMBER` assigns each tied customer a distinct position; this creates the eight rows where the values differ.

| City | Top customer | Completed spend |
|---|---|---:|
| Bengaluru | C0082 | INR 12,856.20 |
| Chennai | C0930 | INR 10,151.40 |
| Delhi | C0291 | INR 13,742.60 |
| Hyderabad | C0743 | INR 12,730.05 |
| Mumbai | C0693 | INR 14,766.55 |
| Pune | C0426 | INR 10,377.10 |

### pandas cross-check and exploratory analysis

The `pivot_table` reconstruction of Q2 uses `margins=True`; its grand total is INR 2,998,668.80, matching the SQL revenue. Per-city completed-order statistics are:

| City | Sum | Mean | Order count |
|---|---:|---:|---:|
| Bengaluru | INR 523,643.40 | INR 276.47 | 1,894 |
| Chennai | INR 463,629.15 | INR 272.72 | 1,700 |
| Delhi | INR 502,829.80 | INR 243.62 | 2,064 |
| Hyderabad | INR 528,589.90 | INR 284.95 | 1,855 |
| Mumbai | INR 479,058.50 | INR 297.00 | 1,613 |
| Pune | INR 500,918.05 | INR 252.48 | 1,984 |

| Completed order amount statistic | Result |
|---|---:|
| Mean | INR 269.91 |
| Median | INR 135.00 |
| Skewness | 4.61 |
| Q1 | INR 64.80 |
| Q2 (median) | INR 135.00 |
| Q3 | INR 297.00 |
| Lower 1.5 × IQR fence | INR -283.50 |
| Upper 1.5 × IQR fence | INR 645.30 |
| Outliers below lower fence | 0 |
| Outliers above upper fence | 897 |

Correlation matrix for completed orders:

| | order_amount | discount_pct | quantity | delivery_minutes |
|---|---:|---:|---:|---:|
| order_amount | 1.000 | -0.095 | 0.087 | -0.003 |
| discount_pct | -0.095 | 1.000 | -0.000 | -0.010 |
| quantity | 0.087 | -0.000 | 1.000 | -0.013 |
| delivery_minutes | -0.003 | -0.010 | -0.013 | 1.000 |

The data describes fictional grocery, personal-care, and electronics orders from six cities. Order amounts are strongly right-skewed: the INR 269.91 mean is well above the INR 135.00 median, with 897 upper-fence outliers. The largest absolute correlation is the weak negative association between amount and discount percentage (about -0.095); amount and quantity are also weakly related (about +0.087). Correlations this small do not establish causation.

## Part B

### RFM and customer segments

RFM uses completed orders only and the snapshot time `2025-10-01 00:00:00`. It contains 1,175 customers with at least one completed order.

| RFM measure | Recency (days) | Frequency (orders) | Monetary (INR) |
|---|---:|---:|---:|
| Median / mean | Median 19 | Mean 9.46 | Mean INR 2,552.06 |
| Minimum / maximum | 0 / 180 | — / 33 | — / INR 14,766.55 |

The RFM columns were standardized before K-means. Each K used `n_init=10` and `random_state=42`.

| K | Inertia | Silhouette |
|---:|---:|---:|
| 2 | 1,820.76 | 0.465 |
| 3 | 1,268.23 | 0.445 |
| 4 | 743.07 | 0.526 |
| 5 | 617.37 | 0.480 |
| 6 | 503.17 | 0.482 |
| 7 | 408.46 | 0.416 |
| 8 | 366.50 | 0.412 |

K=4 is selected because the inertia drop shows an elbow there, it has the highest silhouette score, and four groups support clear campaign actions. Inertia always falls as K grows because more centroids let observations sit closer to a center; by itself it would keep rewarding extra clusters and cannot balance fit with useful, interpretable segment sizes.

| Segment ID | Business name | Customers | Mean recency | Mean frequency | Mean monetary | Premium | Recommended action |
|---:|---|---:|---:|---:|---:|---|---|
| 0 | Steady Everyday | 436 | 28.86 days | 3.77 | INR 1,057.62 | No | Encourage a second category with a relevant cross-category recommendation. |
| 1 | Premium High-Value | 138 | 24.12 days | 6.77 | INR 7,129.70 | Yes | Offer a tailored premium bundle and protect the relationship with proactive service. |
| 2 | Frequent Essentials | 419 | 8.77 days | 19.40 | INR 3,410.02 | No | Use replenishment reminders and practical multi-buy offers. |
| 3 | Cooling-Off Occasional | 182 | 109.04 days | 2.20 | INR 685.99 | No | Send a modest win-back offer and test whether delivery experience is a barrier. |

Segment IDs are cluster labels only; they have no inherent ranking or business meaning. Premium is assigned to the segment with the highest mean monetary value. PCA on the standardized RFM matrix explains **0.6343** and **0.2180** of variance in its first two components, **0.8523** combined.

### Lapse definition and leakage controls

The cohort is customers with at least one completed order before `2025-09-01`. A customer is labelled lapsed when they have no completed order from `2025-09-01` through `2025-09-30`. The cohort has 1,156 customers, 443 lapsed (38.32%), and splits into 924 training and 232 test rows with `test_size=0.2`, stratification, and `random_state=42`.

Features are pre-cutoff recency, distinct completed-order frequency, completed monetary spend, average completed-order discount, cancelled orders divided by all final-status orders, average completed delivery time, favourite product category, city, age, gender, and tenure in days. A category tie is resolved by choosing the alphabetically first category. **No order dated on or after 2025-09-01 entered the model features, and segment labels from the whole-window RFM were not used as features.** The code asserts that no model feature contains a missing value.

The one-hot encoder (`drop="first"`) and numeric standardizer are inside the modeling pipeline and are fitted on training folds during cross-validation, then on the training partition for refit. **No encoder, scaler, or resampler was fit on or sampled from test rows**; the fitted encoder and scaler are only applied to test rows. The frequency mean learned by the selected model's training-only scaler is **8.115**, compared with the full-cohort frequency mean of **8.029**; a stratified split preserves the label proportions, not each feature's exact mean, and keeping preprocessing on training rows avoids using held-out values.

The always-not-lapsed baseline has 61.64% accuracy and lapsed-class precision, recall, and F1 of 0, 0, and 0. Its test confusion matrix (actual rows and predicted columns ordered not-lapsed, lapsed) is `[[143, 0], [89, 0]]`.

### Model tuning and test evaluation

Both searches use five-fold cross-validation and lapsed-class F1. Logistic regression uses `C ∈ {0.01, 0.1, 1, 10, 100}` with `max_iter=1000`. Random forest uses `n_estimators=100`, `random_state=42`, `max_depth ∈ {3, 5, 8}`, and `min_samples_leaf ∈ {5, 10, 20}` (45 fits). The test set was evaluated only after tuning was complete.

| Model | Best parameters | Best CV F1 | Test confusion matrix (TN, FP; FN, TP) | Lapsed precision | Lapsed recall | Lapsed F1 |
|---|---|---:|---|---:|---:|---:|
| Logistic regression | C=100 | 0.651 | `[[111, 32], [31, 58]]` | 0.644 | 0.652 | 0.648 |
| Random forest | max_depth=8, min_samples_leaf=10 | 0.681 | `[[113, 30], [31, 58]]` | 0.659 | 0.652 | 0.655 |

The random forest was selected because its cross-validated F1 is higher than logistic regression's and it also has the better held-out F1. It identifies 58 of 89 lapsing customers, while 30 loyal customers are flagged for outreach. Missing a lapsing customer can mean losing future orders and the chance to win them back; wrongly targeting a loyal customer spends offer budget and may give away an unnecessary discount. The selected model clears the required 0.55 lapsed-class test F1, with outreach costs still needing review before a real campaign.

### Class-balance experiment

The training split started with 570 not-lapsed and 354 lapsed rows. We retained all rows and added 216 randomly sampled lapsed-row duplicates (replacement sampling, `random_state=42`) to reach 570 in each class; no test row was sampled. On the unchanged test set:

| Logistic regression | Precision | Recall | F1 |
|---|---:|---:|---:|
| Original training data | 0.644 | 0.652 | 0.648 |
| Oversampled training data | 0.589 | 0.854 | 0.697 |

Recall moved most, rising by about 20.2 percentage points, while precision fell by about 5.5 points. The retention team would catch more customers who lapse, at the cost of sending more offers to loyal customers.

### Interpretation, persistence, and retention list

The five largest random-forest feature importances are:

| Feature | Importance | Business reading |
|---|---:|---|
| Frequency | 0.280 | Order cadence is the strongest separator of lapse risk in this fit. |
| Monetary | 0.271 | Historical spend adds substantial signal alongside order cadence. |
| Recency | 0.115 | Time since the last order is an important warning signal. |
| Average delivery minutes | 0.061 | Delivery experience adds a smaller contribution to risk ranking. |
| Cancellation rate | 0.052 | Cancellation history adds a smaller contribution to risk ranking. |

Three logistic-regression coefficient signs (positive means higher modeled lapse odds, holding other features fixed):

| Feature | Coefficient | Business reading |
|---|---:|---|
| Recency | +0.0396 | Longer time since the last order raises predicted lapse risk. |
| Frequency | -1.6982 | More completed orders lower predicted lapse risk. |
| Monetary | +0.2846 | Conditional on the other features, higher cumulative spend raises predicted lapse odds in this fitted sample; this can flag valuable customers for review, but is not a causal effect. |

The selected pipeline is saved at `outputs/lapse_model.joblib`; loading it and predicting the original test rows reproduced the predictions exactly. `outputs/customer_segments.csv` has columns `customer_id, recency, frequency, monetary, segment_id, segment_name, is_premium`. `outputs/retention_list.csv` has columns `customer_id, lapse_probability, actually_lapsed` and lists the 10 highest-scoring Premium-segment customers in the test set; **1 of the 10 actually lapsed**.

Because Premium comes from whole-window RFM that includes September, the retention list is illustrative and not a leakage-free evaluation.

## Part C

### Guardrails, redaction, and classification

Tickets are processed in file order. Before a model call, empty/whitespace text is marked `skipped_empty`, and text with `mock_llm.est_tokens(text) > 400` is marked `flagged_long`. Four tickets were skipped as empty and four were flagged long; each long ticket estimates to 734 tokens. Neither group was sent to the model.

The pipeline applies the required regex substitutions in order: email to `[EMAIL]`, phone to `[PHONE]`, and order ID to `[ORDER_ID]`. Reapplying the patterns to all redacted ticket texts found no remaining matches.

| Redaction | Substitutions |
|---|---:|
| Emails | 55 |
| Phone numbers | 91 |
| Order IDs | 132 |
| **Total** | **278** |
| Matches remaining after redaction | **0** |

The user message passed to the offline classifier is exactly the redacted ticket text. The code asserts that all three patterns find no match in each user prompt. No hosted model, network request, or API key is used.

Both system prompts include `CLASSIFY`, the four allowed labels (`billing`, `delivery`, `technical`, `fraud`), and a JSON-only output requirement. The zero-shot prompt has no `EXAMPLES:` line. The few-shot prompt has three labelled examples, including billing, fraud, and technical.

Every `TransientLLMError` is retried at most three times with exponential backoff and jitter using `BASE_DELAY=0.05` seconds. Each independent full run calls `mock_llm.reset()` first. Each of the zero-shot, few-shot, and final few-shot full runs had 19 transient errors, all recovered on retry number 0; the maximum logged wait was 0.056152 seconds, inside the required 0.045–0.0575-second range. The budget-limited demo had 9 retries before it exhausted its budget. The complete retry log is `outputs/triage_retry_log.txt`.

Valid JSON with an allowed category is marked `classified_json`. Invalid JSON is parsed for lowercase alphabetic words and accepted as `classified_keyword` only when exactly one allowed category appears. Valid JSON with a disallowed category, or any other unresolvable reply, is `flagged_invalid_output` and routed to human review.

| Run | Classified | Correct | Accuracy | Total tokens | Tokens per classified ticket |
|---|---:|---:|---:|---:|---:|
| Zero-shot | 147 | 126 | 85.7% | 12,832 | 87.29 |
| Few-shot | 147 | 147 | 100.0% | 23,596 | 160.52 |

Zero-shot confusion table (rows are true categories; columns are predicted categories):

| True \ Predicted | billing | delivery | technical | fraud |
|---|---:|---:|---:|---:|
| billing | 38 | 0 | 0 | 0 |
| delivery | 0 | 36 | 0 | 0 |
| technical | 10 | 0 | 29 | 0 |
| fraud | 11 | 0 | 0 | 23 |

The few-shot prompt used 160.52 tokens per classified ticket versus 87.29 for zero-shot, an increase of 73.22 tokens per ticket (10,764 total). It corrected all 21 errors in the zero-shot run, raising accuracy from 126/147 (85.7%) to 147/147 (100%). That additional prompt cost is justified when a wrong fraud or technical route is more costly than the extra token use. When human review already catches low-risk classifications, the lower-cost zero-shot prompt may be adequate.

### Budget and segment-aware routing

The zero-shot budget demonstration used `TOKEN_BUDGET=6000`; 83 tickets ended with `budget_exhausted`. This run is saved as `outputs/triage_budget_demo.csv`. The final few-shot run used `TOKEN_BUDGET=100000` and wrote 164 rows to `outputs/triage_results.csv`.

The final few-shot run produced 135 rule-validated JSON classifications and 12 keyword-validated classifications. Nine invalid outputs, four empty tickets, and four long tickets were sent to `human_review` at NORMAL priority; no eligible ticket ended with a permanent model error. Fraud tickets route to `security_team` at HIGH priority, Premium customers with other valid classifications route to `premium_desk` at HIGH priority, and the remaining valid classifications route to `support_queue` at NORMAL priority. Customers without a segment row are treated as non-Premium.

| Final route | Tickets |
|---|---:|
| support_queue | 102 |
| security_team | 34 |
| human_review | 17 |
| premium_desk | 11 |

The required output columns are `ticket_id, customer_id, status, category, segment_name, priority, route, tokens_used`; unavailable categories and segment names remain blank.

## Part D

### Policy chunking, retrieval, and answers

Each policy file is split into 40-word chunks with 10 words of overlap. The chunks are stored with their source filename and vectorized with the default `TfidfVectorizer`; queries are ranked with cosine similarity. TF measures how often a word occurs in a chunk, while IDF gives more weight to words that appear in fewer chunks and less weight to common words. TF-IDF can miss semantically equivalent phrasings that use different words, so relevant policy text may rank below a lexical match.

| Policy file | Chunks (10-word overlap) |
|---|---:|
| refund_policy.txt | 4 |
| delivery_policy.txt | 3 |
| escalation_policy.txt | 3 |
| payment_policy.txt | 3 |
| privacy_policy.txt | 2 |
| **Total** | **15** |

The top three results for `refund policy late delivery` all come from `refund_policy.txt`. `answer(question)` sends a `POLICY ASSISTANT` system instruction and a user message formatted as `CONTEXT:`, three source-labelled chunks, and `QUESTION:`. It returns the mock model's answer and the source files used. The system instruction restricts answers to retrieved context and requires the fallback `I could not find...` when the context does not contain an answer.

| Chunking | Expected document in top 3 (answerable) | Expected fact in answer (answerable) | Correct fallback (unanswerable) |
|---|---:|---:|---:|
| 40 words, 10-word overlap | 12/12 | 12/12 | 2/2 |
| 40 words, no overlap | 12/12 | 10/12 | 2/2 |

With zero overlap, `What delivery time makes a completed order eligible for a full refund?` loses the `more than 45 minutes` condition across a chunk boundary, and `How soon is money returned when a payment fails but is deducted?` loses the `48 hours` fact across a chunk boundary. The first condition is split between the chunk ending with the delivery-time phrase and the next chunk starting with the refund consequence; the answerer can return only one matching sentence fragment.

### Refund tools and ReAct agent

`part_d.py` defines the five tools in `TOOLS_SPEC` and maps them through `TOOL_MAP`: `get_order`, `search_policy`, `calculate_refund`, `issue_refund`, and `escalate_to_human`. The ledger tools append timestamped JSON lines to `outputs/refund_ledger.jsonl`; they enforce completed-order, late-delivery, and INR 1000 approval rules. All simulated LLM calls use the local `mock_llm.py`.

| Scenario | Status | Refund amount | Model calls | Tokens used |
|---|---|---:|---:|---:|
| S1 | REFUND | INR 128.25 | 5 | 1,147 |
| S2 | REFUND | INR 135.00 | 5 | 1,147 |
| S3 | ESCALATE | INR 1,999.00 | 5 | 1,134 |
| S4 | DENIED | INR 0.00 | 4 | 805 |
| S5 | DENIED | INR 0.00 | 3 | 487 |
| S6 | NOT_FOUND | INR 0.00 | 2 | 172 |
| S7 | FALLBACK_MAX_ITERATIONS | INR 0.00 | 6 | 3,300 |

S7 used all six allowed model calls and then fell back to human review. The safety demonstrations returned `FALLBACK_TOKEN_LIMIT` for S1 with a 200-token limit (2 calls; 202 tokens), and `FALLBACK_TIMEOUT` with a zero-second time limit (0 calls; 0 tokens).

After every tool observation, the run checkpoints the full messages, tokens used, step number, and executed tool names under `state/`. The S1 crash demonstration checkpointed after `get_order`, `search_policy`, then resumed with `calculate_refund`, `issue_refund`; the combined tool sequence is exactly `get_order → search_policy → calculate_refund → issue_refund`, with no repeated tool execution, and finishes `REFUND: INR 128.25`. The resumed run used 5 model calls and 1,147 tokens. `outputs/agent_runs.jsonl` records the seven scenarios and the safety/crash demonstrations; `outputs/refund_ledger.jsonl` contains four simulated issue/escalation entries.

### ReAct versus plan-and-execute

ReAct used 5 model calls for S1, 3 for S5, and 2 for S6 in these runs, paying more calls and tokens as each tool observation returns to the model. A plan-and-execute design could request a known sequence once and use fewer calls, but it must replan when an observation differs from the plan: `NOT_FOUND` should stop lookup, and a cancelled order must not proceed to refund calculation. For this refund workflow, ReAct is safer because each decision follows the observed database and policy result, while the six-call bound provides a clear human handoff for loops.

## Limitations and Next Steps

The source data is synthetic and cannot establish real customer or operational behaviour. Because Premium comes from whole-window RFM that includes September, the retention list is illustrative and not a leakage-free evaluation. Every refund action in Part D is simulated in a local JSONL ledger and does not move real money.
