# Shipments API v2

The Shipments API v2 base URL is https://api.example.com/shipments. Requests are authenticated with an OAuth bearer token.

# Shipments API Rate Limits

The Shipments API v2 allows 900 requests per minute per client. When the limit is exceeded, the API returns HTTP 429.

# Shipments API Webhooks

Webhook deliveries for the Shipments API are retried up to 4 times over 48 hours.
