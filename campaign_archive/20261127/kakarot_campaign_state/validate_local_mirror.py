#!/usr/bin/env python3
"""Validate a mirrored result while resolving remote absolute paths locally.

The result JSON remains untouched and is checksum-verified separately. Only
the in-memory reference to a matched-control source schedule is translated to
its corresponding local mirror before causal replay validation.
"""
import json
import pathlib
import sys

STATE = pathlib.Path(__file__).resolve().parent
REPO = pathlib.Path('/home/anurag-basistha/Projects/Untapped/FedQCL_Task')
sys.path.insert(0, str(REPO / 'scripts'))
sys.path.insert(0, str(REPO / 'system'))

import flcore.sync.decision_audit as decision_audit

original_verify = decision_audit.verify_decisions


def verify_decisions_with_local_schedule(config, *args, **kwargs):
    resolved = dict(config)
    schedule = resolved.get('sync_count_schedule')
    if schedule and not pathlib.Path(schedule).is_file():
        marker = '/research_results/'
        value = pathlib.PurePosixPath(schedule).as_posix()
        if marker in value:
            relative = value.split(marker, 1)[1]
            candidate = REPO / 'research_results' / relative
            if candidate.is_file():
                resolved['sync_count_schedule'] = str(candidate)
    return original_verify(resolved, *args, **kwargs)


decision_audit.verify_decisions = verify_decisions_with_local_schedule
import run_campaign

job = json.load(sys.stdin)
sys.exit(0 if run_campaign.valid(job) else 1)
