# Invoices API v2

The Invoices API v2 base URL is https://api.example.com/invoices. Requests are authenticated with an OAuth bearer token.

# Invoices API Rate Limits

The Invoices API v2 allows 450 requests per minute per client. When the limit is exceeded, the API returns HTTP 429.

# Invoices API Webhooks

Webhook deliveries for the Invoices API are retried up to 4 times over 12 hours.
