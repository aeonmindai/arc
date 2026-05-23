//! Python trajectories (5).

use super::{assemble, assistant_call, assistant_text, src, tool, user};
use crate::trajectory::Trajectory;
use serde_json::json;

pub fn all() -> Vec<Trajectory> {
    vec![
        py_006_pandas_groupby(),
        py_008_requests_retry(),
        py_009_sqlalchemy_session(),
        py_010_django_admin(),
        super::long_trajectories::python_long(),
    ]
}

fn py_006_pandas_groupby() -> Trajectory {
    let mut turns = Vec::new();
    turns.push(user(
        "I have orders.csv with columns (order_id, customer_id, amount, created_at). Write a function in analysis.py that returns a DataFrame indexed by customer_id, with total spend and order count. Use pandas groupby.",
    ));
    turns.push(assistant_call(
        "Let me see the existing analysis module first.",
        "call_001",
        "read_file",
        json!({"path":"analysis.py"}),
        &turns,
    ));
    turns.push(tool(
        "call_001",
        "import pandas as pd\n\ndef load_orders(path: str) -> pd.DataFrame:\n    return pd.read_csv(path, parse_dates=[\"created_at\"])\n",
    ));
    turns.push(assistant_call(
        "I'll add `customer_summary` that does a groupby on customer_id and aggregates with named expressions.",
        "call_002",
        "write_file",
        json!({
            "path":"analysis.py",
            "contents":"import pandas as pd\n\ndef load_orders(path: str) -> pd.DataFrame:\n    return pd.read_csv(path, parse_dates=[\"created_at\"])\n\n\ndef customer_summary(orders: pd.DataFrame) -> pd.DataFrame:\n    return (\n        orders.groupby(\"customer_id\")\n        .agg(total_spend=(\"amount\", \"sum\"), order_count=(\"order_id\", \"count\"))\n        .sort_values(\"total_spend\", ascending=False)\n    )\n"
        }),
        &turns,
    ));
    turns.push(tool("call_002", "wrote 412 bytes to analysis.py"));
    turns.push(assistant_text(
        "Done. `customer_summary` returns a DataFrame indexed by customer_id with `total_spend` and `order_count` columns sorted by spend descending.",
        &turns,
    ));

    assemble(
        "agentperf_tuning_006",
        "python",
        src("github.com/pandas-dev/pandas", "BSD-3-Clause", "be4e0ea"),
        turns,
    )
}

#[allow(dead_code)]
fn py_007_flask_blueprint() -> Trajectory {
    let mut turns = Vec::new();
    turns.push(user(
        "Add a Flask blueprint named `health` to api/health.py that exposes GET /health returning {\"status\": \"ok\"} and register it in app.py at the /api prefix.",
    ));
    turns.push(assistant_call(
        "Reading the current app entrypoint.",
        "call_001",
        "read_file",
        json!({"path":"app.py"}),
        &turns,
    ));
    turns.push(tool(
        "call_001",
        "from flask import Flask\n\napp = Flask(__name__)\n\n@app.route(\"/\")\ndef index():\n    return {\"hello\": \"world\"}\n",
    ));
    turns.push(assistant_call(
        "Creating the blueprint.",
        "call_002",
        "write_file",
        json!({
            "path":"api/health.py",
            "contents":"from flask import Blueprint\n\nhealth_bp = Blueprint(\"health\", __name__)\n\n@health_bp.get(\"/health\")\ndef health() -> dict[str, str]:\n    return {\"status\": \"ok\"}\n"
        }),
        &turns,
    ));
    turns.push(tool("call_002", "wrote 178 bytes to api/health.py"));
    turns.push(assistant_call(
        "Now registering it.",
        "call_003",
        "write_file",
        json!({
            "path":"app.py",
            "contents":"from flask import Flask\nfrom api.health import health_bp\n\napp = Flask(__name__)\napp.register_blueprint(health_bp, url_prefix=\"/api\")\n\n@app.route(\"/\")\ndef index():\n    return {\"hello\": \"world\"}\n"
        }),
        &turns,
    ));
    turns.push(tool("call_003", "wrote 226 bytes to app.py"));
    turns.push(assistant_text(
        "Blueprint registered. GET /api/health returns {\"status\": \"ok\"}.",
        &turns,
    ));

    assemble(
        "agentperf_tuning_007",
        "python",
        src("github.com/pallets/flask", "BSD-3-Clause", "d6c5e0b"),
        turns,
    )
}

fn py_008_requests_retry() -> Trajectory {
    let mut turns = Vec::new();
    turns.push(user(
        "In client.py we use requests directly. Refactor to use a Session with a urllib3 Retry adapter that retries 5 times on 429 and 5xx with exponential backoff.",
    ));
    turns.push(assistant_call(
        "Reading.",
        "call_001",
        "read_file",
        json!({"path":"client.py"}),
        &turns,
    ));
    turns.push(tool(
        "call_001",
        "import requests\n\nBASE = \"https://api.example.com\"\n\ndef get_user(user_id: int) -> dict:\n    r = requests.get(f\"{BASE}/users/{user_id}\", timeout=5)\n    r.raise_for_status()\n    return r.json()\n",
    ));
    turns.push(assistant_call(
        "I'll build a module-level session with a Retry mounted on https://. Retry uses backoff_factor for exponential growth.",
        "call_002",
        "write_file",
        json!({
            "path":"client.py",
            "contents":"import requests\nfrom requests.adapters import HTTPAdapter\nfrom urllib3.util.retry import Retry\n\nBASE = \"https://api.example.com\"\n\n_RETRY = Retry(\n    total=5,\n    backoff_factor=0.5,\n    status_forcelist=(429, 500, 502, 503, 504),\n    allowed_methods=frozenset([\"GET\", \"HEAD\", \"OPTIONS\"]),\n    raise_on_status=False,\n)\n\n\ndef _session() -> requests.Session:\n    s = requests.Session()\n    s.mount(\"https://\", HTTPAdapter(max_retries=_RETRY))\n    s.mount(\"http://\", HTTPAdapter(max_retries=_RETRY))\n    return s\n\n\nSESSION = _session()\n\n\ndef get_user(user_id: int) -> dict:\n    r = SESSION.get(f\"{BASE}/users/{user_id}\", timeout=5)\n    r.raise_for_status()\n    return r.json()\n"
        }),
        &turns,
    ));
    turns.push(tool("call_002", "wrote 730 bytes to client.py"));
    turns.push(assistant_text(
        "Retry policy in place: 5 attempts, exponential backoff (0.5s base), retries on 429 and 5xx for idempotent methods.",
        &turns,
    ));

    assemble(
        "agentperf_tuning_008",
        "python",
        src("github.com/psf/requests", "Apache-2.0", "0e322af"),
        turns,
    )
}

fn py_009_sqlalchemy_session() -> Trajectory {
    let mut turns = Vec::new();
    turns.push(user(
        "Add a SQLAlchemy context-manager `db_session` in db.py that yields a Session, commits on success, and rolls back on exception. Use the existing engine.",
    ));
    turns.push(assistant_call(
        "Reading the file.",
        "call_001",
        "read_file",
        json!({"path":"db.py"}),
        &turns,
    ));
    turns.push(tool(
        "call_001",
        "from sqlalchemy import create_engine\nfrom sqlalchemy.orm import sessionmaker, DeclarativeBase\n\nengine = create_engine(\"sqlite:///app.db\", future=True)\nSessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)\n\nclass Base(DeclarativeBase):\n    pass\n",
    ));
    turns.push(assistant_call(
        "Adding the contextmanager.",
        "call_002",
        "write_file",
        json!({
            "path":"db.py",
            "contents":"from contextlib import contextmanager\nfrom typing import Iterator\nfrom sqlalchemy import create_engine\nfrom sqlalchemy.orm import sessionmaker, DeclarativeBase, Session\n\nengine = create_engine(\"sqlite:///app.db\", future=True)\nSessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)\n\nclass Base(DeclarativeBase):\n    pass\n\n@contextmanager\ndef db_session() -> Iterator[Session]:\n    session: Session = SessionLocal()\n    try:\n        yield session\n        session.commit()\n    except Exception:\n        session.rollback()\n        raise\n    finally:\n        session.close()\n"
        }),
        &turns,
    ));
    turns.push(tool("call_002", "wrote 596 bytes to db.py"));
    turns.push(assistant_text(
        "Done. Use as `with db_session() as s: s.add(obj)` — commit on success, rollback on exception, always close.",
        &turns,
    ));

    assemble(
        "agentperf_tuning_009",
        "python",
        src("github.com/sqlalchemy/sqlalchemy", "MIT", "1a4b0c6"),
        turns,
    )
}

fn py_010_django_admin() -> Trajectory {
    let mut turns = Vec::new();
    turns.push(user(
        "Register the Order model in orders/admin.py with list_display = (id, customer, total, created_at) and search_fields on customer email and id.",
    ));
    turns.push(assistant_call(
        "Reading the model first.",
        "call_001",
        "read_file",
        json!({"path":"orders/models.py"}),
        &turns,
    ));
    turns.push(tool(
        "call_001",
        "from django.db import models\n\nclass Order(models.Model):\n    customer = models.ForeignKey(\"customers.Customer\", on_delete=models.CASCADE)\n    total = models.DecimalField(max_digits=12, decimal_places=2)\n    created_at = models.DateTimeField(auto_now_add=True)\n\n    def __str__(self) -> str:\n        return f\"Order #{self.pk}\"\n",
    ));
    turns.push(assistant_call(
        "Now reading admin.py.",
        "call_002",
        "read_file",
        json!({"path":"orders/admin.py"}),
        &turns,
    ));
    turns.push(tool(
        "call_002",
        "from django.contrib import admin\n# Register your models here.\n",
    ));
    turns.push(assistant_call(
        "Writing the registration.",
        "call_003",
        "write_file",
        json!({
            "path":"orders/admin.py",
            "contents":"from django.contrib import admin\nfrom .models import Order\n\n@admin.register(Order)\nclass OrderAdmin(admin.ModelAdmin):\n    list_display = (\"id\", \"customer\", \"total\", \"created_at\")\n    search_fields = (\"id\", \"customer__email\")\n    list_filter = (\"created_at\",)\n    ordering = (\"-created_at\",)\n"
        }),
        &turns,
    ));
    turns.push(tool("call_003", "wrote 350 bytes to orders/admin.py"));
    turns.push(assistant_text(
        "Order is now registered with the requested list_display and search_fields plus a date filter and reverse-chronological ordering for usability.",
        &turns,
    ));

    assemble(
        "agentperf_tuning_010",
        "python",
        src("github.com/django/django", "BSD-3-Clause", "fc4e3f2"),
        turns,
    )
}
