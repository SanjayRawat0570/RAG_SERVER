"""GDPR data rights (F25): access, portability, erasure."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def export_user_data(user_id: str) -> dict[str, Any]:
    """Collect all data held for *user_id* (right to data portability)."""
    from app.rag.personalization.store import get_history, get_profile
    from app.rag.feedback.store        import get_feedback
    from app.rag.security.audit        import get_events
    from app.rag.security.apikeys      import list_api_keys
    from app.rag.memory                import store as conversations
    from app.rag.personalization.store import infer_interests

    profile  = get_profile(user_id)
    history  = get_history(user_id, limit=10_000)
    feedback = get_feedback(user_id=user_id, limit=10_000)
    audit    = get_events(user_id=user_id, limit=10_000)
    keys     = list_api_keys(user_id)

    return {
        "user_id":      user_id,
        "exported_at":  datetime.now(timezone.utc).isoformat(),
        "profile":      profile.model_dump(mode="json"),
        "query_history": [r.model_dump(mode="json") for r in history],
        # Conversation memory: short-term turns per chat, long-term interests
        # inferred from the topics of cited sources.
        "conversations": conversations.export_user(user_id),
        "inferred_interests": infer_interests(user_id, top_n=10),
        "feedback":     [f.model_dump(mode="json") for f in feedback],
        "audit_log": [
            {
                "id":        e.id,
                "action":    e.action,
                "resource":  e.resource,
                "timestamp": e.timestamp.isoformat(),
            }
            for e in audit
        ],
        "api_keys": [
            {
                "id":         k.id,
                "name":       k.name,
                "prefix":     k.prefix,
                "created_at": k.created_at.isoformat(),
            }
            for k in keys
        ],
    }


def delete_user_data(user_id: str) -> dict[str, Any]:
    """Erase all data for *user_id* (right to erasure / right to be forgotten)."""
    from app.rag.personalization.store import reset_profile
    from app.rag.security.apikeys      import delete_api_key, list_api_keys
    from app.rag.memory                import store as conversations

    reset_profile(user_id)              # profile, query history and inferred interests
    erased_conversations = conversations.erase_user(user_id)

    keys = list_api_keys(user_id)
    for k in keys:
        delete_api_key(k.id)

    return {
        "user_id": user_id,
        "deleted": {
            "profile_and_history": True,
            "conversations": erased_conversations,
            "api_keys": len(keys),
        },
        "status": "deleted",
    }


def data_inventory(user_id: str) -> dict[str, Any]:
    """Summary of data categories stored for *user_id* (right to access)."""
    from app.rag.personalization.store import get_history
    from app.rag.feedback.store        import get_feedback
    from app.rag.security.audit        import get_events
    from app.rag.security.apikeys      import list_api_keys
    from app.rag.memory                import store as conversations

    history  = get_history(user_id, limit=10_000)
    feedback = get_feedback(user_id=user_id, limit=10_000)
    audit    = get_events(user_id=user_id, limit=10_000)
    keys     = list_api_keys(user_id)

    return {
        "user_id": user_id,
        "data_categories": {
            "profile":       "Preferences and interests",
            "query_history": f"{len(history)} records",
            "conversations": f"{len(conversations.export_user(user_id))} chats (last 5 turns each)",
            "feedback":      f"{len(feedback)} ratings",
            "audit_log":     f"{len(audit)} events",
            "api_keys":      f"{len(keys)} keys",
        },
        "retention_policy": {
            "query_history": "200 records (rolling)",
            "conversations": "Last 5 turns per chat; deleted on request or erasure",
            "feedback":      "Unlimited",
            "audit_log":     "100,000 events (rolling)",
        },
        "legal_basis": {
            "query_history": "Legitimate interest (service improvement)",
            "personalization": "Consent",
            "audit_log":     "Legal obligation / security",
        },
    }
