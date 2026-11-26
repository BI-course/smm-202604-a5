https://smm-202604-a5.onrender.com -- link to Render endpoint


Deploying api.py to Render

**Public URL:** https://smm-202604-a5.onrender.com

## Deployment settings
- Repository and branch: BI-course/smm-202604-a5, main (commit 61c7af8)
- Build command: `pip install -r requirements/base.txt -r requirements/prod.txt -c requirements/constraints.txt`
- Start command: `gunicorn -w 4 -b 0.0.0.0:$PORT api:app`
- Plan: Free
- Environment variable: PYTHON_VERSION = 3.12.10
- First deploy succeeded on 26 November 2026. The same api.py from main was used, unchanged.

## Test results (curl from my own computer to the public URL)
| Endpoint | Result |
|---|---|
| monthly-rent-regressor | predicted_monthly_rent_kes: 89198.28 |
| employee-attrition-classifier | Yes (0.7173), No (0.2827) |
| store-segmenter | assigned_cluster: 4 |
| track-recommender (cart M10, M09) | M08 Sunday Worship 1, confidence 0.862, lift 5.6 |

## Sample request
    curl -X POST https://smm-202604-a5.onrender.com/api/v1/models/store-segmenter/predictions -H "Content-Type: application/json" -d '{"monthly_revenue_kes":1200000,"daily_foot_traffic":450,"staff_count":12,"store_size_sqm":180,"avg_basket_size_kes":1500}'

Response: {"assigned_cluster":4,"model":"K-Means (k=5, standardized features)"}
