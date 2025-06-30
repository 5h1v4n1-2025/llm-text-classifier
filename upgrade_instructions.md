# Gemini API Upgrade Instructions

## Current Free Tier Limits:
- 50 requests per day
- 30 requests per minute

## Paid Tier Benefits:
- 15 requests per second (900/minute)
- 1,500 requests per minute
- Much higher daily limits

## How to Upgrade:

### Step 1: Go to Google AI Studio
1. Visit: https://aistudio.google.com/app/apikey
2. Sign in with your Google account

### Step 2: Enable Billing
1. Click on "Billing" in the left sidebar
2. Set up a Google Cloud billing account
3. Link it to your AI Studio project

### Step 3: Check Quotas
1. Go to: https://console.cloud.google.com/apis/api/generativelanguage.googleapis.com/quotas
2. Look for "Generate Content Requests Per Day"
3. Default paid tier: 1,500 requests per day

### Step 4: Update Script
Once upgraded, update the script with higher limits:

```python
# In process_with_gemini.py, change:
self.max_requests_per_day = 1500  # Paid tier limit
```

## Cost Estimate:
- ~$0.0005 per 1K characters input
- ~$0.0015 per 1K characters output
- For 53 rows: ~$0.50-1.00 total

## Alternative: Use Different Model
Try gemini-1.5-pro for potentially higher limits:
```python
self.model = genai.GenerativeModel('gemini-1.5-pro')
``` 