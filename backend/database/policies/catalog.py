"""RBAC permission catalog and role→permission grants.

The catalog is the source of truth used to seed the ``permissions`` / ``roles`` /
``role_permissions`` tables. Runtime enforcement consults the database grants
first (administrators may customise them) and falls back to this catalog when the
database has not been seeded yet.

Role hierarchy (used by the fallback path and by ``RBAC.can``):
``user`` < ``analyst`` < ``dataset_manager`` < ``admin`` < ``super_admin``.
"""

from __future__ import annotations

#: Roles used by the enterprise data layer. `user`/`analyst`/`admin` keep the
#: Phase 8 names so existing route guards keep working.
SYSTEM_ROLES = {
    "user": "Farmer",
    "analyst": "Researcher",
    "dataset_manager": "Dataset Manager",
    "admin": "Administrator",
    "super_admin": "Super Admin",
}

#: Privilege ordering (higher = more privileged).
ROLE_PRIORITY = {
    "user": 0,
    "analyst": 1,
    "dataset_manager": 2,
    "admin": 3,
    "super_admin": 4,
}

#: (code, resource, action, name, description)
PERMISSIONS: list[tuple[str, str, str, str, str]] = [
    ("prediction.create", "prediction", "create", "Create prediction", "Run a crop prediction"),
    ("prediction.read", "prediction", "read", "Read prediction", "View a prediction detail"),
    ("prediction.history", "prediction", "history", "View history", "List and search prediction history"),
    ("prediction.export", "prediction", "export", "Export prediction", "Mark or export a prediction"),
    ("explain.read", "explain", "read", "Read explanation", "View explainability reports"),
    ("model.read", "model", "read", "Read models", "List model registry entries"),
    ("model.manage", "model", "manage", "Manage models", "Register/activate model versions"),
    ("dataset.read", "dataset", "read", "Read datasets", "List dataset registry entries"),
    ("dataset.manage", "dataset", "manage", "Manage datasets", "Register/validate dataset versions"),
    ("spatial.read", "spatial", "read", "Read spatial", "Query boundaries and locations"),
    ("analytics.read", "analytics", "read", "Read analytics", "View dashboard analytics"),
    ("experiment.read", "experiment", "read", "Read experiments", "List research experiments"),
    ("experiment.manage", "experiment", "manage", "Manage experiments", "Create/update experiments"),
    ("user.read", "user", "read", "Read users", "List/search user accounts"),
    ("user.manage", "user", "manage", "Manage users", "Create/update/disable users and roles"),
    ("role.manage", "role", "manage", "Manage roles", "Grant/revoke role permissions"),
    ("feedback.read", "feedback", "read", "Read feedback", "View submitted feedback"),
    ("feedback.manage", "feedback", "manage", "Manage feedback", "Resolve and triage feedback"),
    ("notifications.read", "notifications", "read", "Read notifications", "View own notifications"),
    ("audit.read", "audit", "read", "Read audit", "View the audit trail"),
    ("config.read", "config", "read", "Read config", "View application configuration"),
    ("config.manage", "config", "manage", "Manage config", "Update application configuration"),
    ("system.read", "system", "read", "Read system", "View system logs and health"),
]

#: Default grants per system role (customisable at runtime via the DB).
ROLE_PERMISSIONS: dict[str, set[str]] = {
    "user": {
        "prediction.create",
        "prediction.read",
        "prediction.history",
        "prediction.export",
        "explain.read",
        "spatial.read",
        "notifications.read",
    },
    "analyst": {
        "prediction.create",
        "prediction.read",
        "prediction.history",
        "prediction.export",
        "explain.read",
        "spatial.read",
        "notifications.read",
        "analytics.read",
        "model.read",
        "dataset.read",
        "experiment.read",
        "feedback.read",
    },
    "dataset_manager": {
        "prediction.create",
        "prediction.read",
        "prediction.history",
        "prediction.export",
        "explain.read",
        "spatial.read",
        "notifications.read",
        "analytics.read",
        "model.read",
        "dataset.read",
        "experiment.read",
        "feedback.read",
        "dataset.manage",
        "system.read",
    },
    "admin": {
        "prediction.create",
        "prediction.read",
        "prediction.history",
        "prediction.export",
        "explain.read",
        "spatial.read",
        "notifications.read",
        "analytics.read",
        "model.read",
        "model.manage",
        "dataset.read",
        "dataset.manage",
        "experiment.read",
        "experiment.manage",
        "feedback.read",
        "feedback.manage",
        "user.read",
        "user.manage",
        "audit.read",
        "config.read",
        "config.manage",
        "system.read",
    },
    "super_admin": {
        code for code, _resource, _action, _name, _desc in PERMISSIONS
    },
}
