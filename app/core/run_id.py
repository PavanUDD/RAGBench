from datetime import datetime
import uuid

def new_run_id() -> str:
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    short = uuid.uuid4().hex[:8]
    return f"run_{ts}_{short}"
