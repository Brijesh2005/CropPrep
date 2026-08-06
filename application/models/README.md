# Models (`application/models/`)

Directory for the **exported CropFusion model weights** that the Prediction
Platform serves at inference time.

R1.4 prepares the layout and naming convention only. **No model is loaded in
R1.4** — loading is deliberately deferred (see `application/inference/loaders`).

## Expected layout

| File | Status | Purpose |
| --- | --- | --- |
| `cropfusion.pt` | expected (not yet shipped) | The current trained model artifact |
| `cropfusion_v1.pt` | future | Pinned version 1 |
| `cropfusion_v2.pt` | future | Pinned version 2 |
| `cropfusion_latest.pt` | future | "Latest" copy / symlink used by default resolution |

## Naming convention

- Default artifact: `cropfusion.pt`
- Pinned versions: `cropfusion_{version}.pt` (semantic version, e.g. `v1`, `v2`)
- `latest`: `cropfusion_latest.pt`

Version resolution is the responsibility of `application/inference/versioning`
(`ModelVersionResolver`) — not implemented in R1.4.

## Rules

- Weights are **consumed only**; they are produced by the Training Platform
  export pipeline and shipped at deployment time.
- Never load weights at import time; always go through the loader port.
- Never commit large binaries; the `.gitignore` keeps them out of git.
