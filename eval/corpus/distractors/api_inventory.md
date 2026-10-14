# Inventory API v2

The Inventory API v2 base URL is https://api.example.com/inventory. Requests are authenticated with an OAuth bearer token.

# Inventory API Rate Limits

The Inventory API v2 allows 60 requests per minute per client. When the limit is exceeded, the API returns HTTP 429.

# Inventory API Webhooks

Webhook deliveries for the Inventory API are retried up to 3 times over 12 hours.
