# Reports API v2

The Reports API v2 base URL is https://api.example.com/reports. Requests are authenticated with an OAuth bearer token.

# Reports API Rate Limits

The Reports API v2 allows 300 requests per minute per client. When the limit is exceeded, the API returns HTTP 429.

# Reports API Webhooks

Webhook deliveries for the Reports API are retried up to 5 times over 12 hours.
