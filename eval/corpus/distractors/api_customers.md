# Customers API v2

The Customers API v2 base URL is https://api.example.com/customers. Requests are authenticated with an OAuth bearer token.

# Customers API Rate Limits

The Customers API v2 allows 900 requests per minute per client. When the limit is exceeded, the API returns HTTP 429.

# Customers API Webhooks

Webhook deliveries for the Customers API are retried up to 8 times over 12 hours.
