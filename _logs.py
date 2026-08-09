import json
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelSessionLogsStreamRequest

api = KaggleApi()
client = api.build_kaggle_client()
k = client.kernels.kernels_api_client

req = ApiGetKernelSessionLogsStreamRequest()
req.user_name = "gojosensa"
req.kernel_slug = "cropfusion-system-check"
req.wait_for_logs_url_seconds = 60

resp = k.get_kernel_session_logs_stream(req)
lines = json.loads(resp.content.decode("utf-8"))
keep = [l for l in lines if any(kw in l["data"] for kw in
        ("GPU", "gpu", "passed", "validation", "READY", "bootstrap", "CUDA", "torch"))]
for e in keep:
    data = e["data"]
    if len(data) > 300:
        data = data[:300] + " ..."
    print("[%s] %s" % (e["stream_name"], data.rstrip()))
