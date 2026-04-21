# Paste into GitHub when opening the PR

**Title (suggested):** Strengthen IHM TLS task tests + rubric PR metadata

**Description (paste below the line):**

---

**Contributor:** Akshad Pai (avpai2), Matthew Ruth (mrruth2)

**Contribution type:** Full pipeline (Dataset + Task + Model) — follow-up on PR #3

**Paper:** Kuznetsova et al., “On the Importance of Step-wise Embeddings for Heterogeneous Clinical Time-Series,” *Journal of Machine Learning Research* (2023).  
https://jmlr.org/papers/v24/22-0850.html

**Description:**  
Addresses review feedback: extends `tests/core/test_mimic3_tls.py` so `InHospitalMortalityTLS.__call__` is exercised on synthetic `Patient` / `timeseries/*` event rows (sorting, truncation, label handling, NaN / bad numeric features, feature subsets, empty / invalid cases). PR #3 previously had an empty body; this text matches the course rubric for contributors, paper link, and file guide.

**File guide:**

- `tests/core/test_mimic3_tls.py` — additional `TestInHospitalMortalityTLSCall` cases hitting real task event logic  
- `PR_FULL_PIPELINE_FOLLOWUP.md` — copy-paste PR title/body helper for maintainers (optional to delete after merge)

**Base / head:** This PR targets `full-pipeline-4-15` so it stacks on the same branch as [PR #3](https://github.com/akshadpai/PyHealth-step-wise-embeddings/pull/3).
