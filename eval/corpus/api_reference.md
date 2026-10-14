# Orders API v2

The Orders API v2 base URL is https://api.example.com/v2. Requests are authenticated with the X-Api-Token header.

# Rate Limits

The Orders API v2 allows 600 requests per minute per API key. The legacy Orders API v1 allowed only 120 requests per minute. When the limit is exceeded, the API returns HTTP 429 with a Retry-After header.

# Endpoints

GET /v2/orders/{order_id} returns a single order. POST /v2/orders creates an order. DELETE /v2/orders/{order_id} cancels an order that has not shipped yet.

# API Error Codes

ERR_ORDER_LOCKED means the order is being fulfilled and can no longer be changed. ERR_ORDER_NOT_FOUND means no order exists with that id. ERR_TOKEN_EXPIRED means the API token is older than 90 days and must be rotated.

# Webhooks

Webhook deliveries are retried up to 6 times over 24 hours with exponential backoff.
