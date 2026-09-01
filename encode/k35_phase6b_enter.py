import json
from quant_pipeline.campaign import glm53_uniform_k35 as k35

plan = json.load(open("plan.json"))
state = json.load(open("state/state-0000.json"))
readiness = json.load(open("gss/readiness-receipt.json"))
successor = k35.enter_k35_encoding(
    plan, state, readiness_receipt_sha256=readiness["readiness_receipt_sha256"])
json.dump(successor, open(f"state/state-{successor['sequence']:04d}.json", "w"), indent=2)
print("state", successor["sequence"], successor["phase"])
