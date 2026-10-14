# Severity Levels

SEV1 is a customer-facing outage. The on-call engineer must respond within 5 minutes, and the status page is updated every 30 minutes.

SEV2 is a major degradation with a workaround. A response is required within 30 minutes.

SEV3 is a minor issue that is handled on the next business day.

# Incident Roles

Every SEV1 and SEV2 incident has an incident commander, who coordinates the response and all communication.

# Postmortems

A written postmortem is required within 5 business days for every SEV1 and SEV2 incident.

# Incident INC-2024-031

INC-2024-031 was a SEV1 payments outage on 14 March 2024 that lasted 47 minutes. The root cause was an expired TLS certificate on the payment gateway. The Payments team led the response.

# Incident INC-2024-058

INC-2024-058 was a SEV2 incident on 2 September 2024 in which CSV exports timed out for 6 hours. The underlying defect was tracked as BUG-1893.
