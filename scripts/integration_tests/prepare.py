# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Record the checkout revision on the login node before submitting tests."""

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

root = Path(__file__).resolve().parents[2]
record = {
    "prepared_at": datetime.now(UTC).isoformat(),
    "commit": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip(),
    "working_tree": subprocess.check_output(
        ["git", "--no-optional-locks", "status", "--short"], cwd=root, text=True
    ).splitlines(),
}
output = root / "integration_test_results/prepared_revision.json"
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(record, indent=2) + "\n")
print(f"Prepared test revision: {record['commit']}")
print(f"Revision record: {output}")
