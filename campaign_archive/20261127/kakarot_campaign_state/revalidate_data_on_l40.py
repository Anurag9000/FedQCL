import datetime,hashlib,json,sys
from pathlib import Path
root=Path("/home/anurag/Projects/FedQCL_Task")
sys.path.insert(0,str(root/"scripts"))
import run_campaign
jobs=[json.loads(x) for x in Path("/tmp/fedqcl-data-revalidation.jsonl").read_text().splitlines() if x.strip()]
status=json.loads(Path("/tmp/fedqcl-data-status.json").read_text())
for j in jobs:
 ok=run_campaign.valid(j)
 hashes={p:run_campaign.sha_file(p) for p in j["artifacts"] if Path(p).is_file()}
 expected=status[j["id"]]["artifact_sha256"]
 match=ok and hashes==expected
 print(json.dumps({"id":j["id"],"semantic_id":j["semantic_id"],"validator_pass":ok,"artifact_hashes_match_original_status":hashes==expected,"artifact_sha256":hashes,"verified":match},sort_keys=True),flush=True)
