# Catalog API v2

The Catalog API v2 base URL is https://api.example.com/catalog. Requests are authenticated with an OAuth bearer token.

# Catalog API Rate Limits

The Catalog API v2 allows 90 requests per minute per client. When the limit is exceeded, the API returns HTTP 429.

# Catalog API Webhooks

Webhook deliveries for the Catalog API are retried up to 8 times over 6 hours.
