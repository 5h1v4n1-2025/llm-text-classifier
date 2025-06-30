# How to Get a New Gemini API Key

## Steps:
1. Go to: https://aistudio.google.com/app/apikey
2. Sign in with a **different Google account** (or create a new one)
3. Create a new API key
4. The new account will have its own 50 requests/day quota

## Alternative: Use Different Model
Try a different Gemini model that might have separate quotas:
- gemini-1.5-pro
- gemini-1.0-pro
- gemini-1.0-pro-latest

## Quick Test:
Once you have a new key, test it with:
```python
python3 test_api.py
```

## Update Script:
Replace the API key in your batch classifier:
```python
api_key = "YOUR_NEW_API_KEY_HERE"
``` 