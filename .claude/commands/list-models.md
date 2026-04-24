# List and filter models in AI4Science Studio

Discover what models are available in this repository.

## How to respond

1. Read `models.yaml` at the repo root.
2. Parse the YAML and present the models in a table.
3. If the user provided filter criteria (in $ARGUMENTS), apply them.

## Default output (no filter)

Present ALL models in a table:

| Model | Domain | HF id | License | Tasks | Path |
|-------|--------|-------|---------|-------|------|
| ... | ... | ... | ... | ... | ... |

## Filter examples

- **By domain:** "earth_science" → show only earth_science models
- **By task:** "finetune" → show only models with `finetune` in `tasks_available`
- **By license:** "MIT" → show only MIT-licensed models
- **By HF availability:** "has HF weights" → filter where `hf_id != N/A`
- **By hardware:** read individual `model.yaml` files and filter by `validated_hardware`

## Deep dive

If the user names a specific model, read its `model.yaml` and present:
- Full metadata (HF id, license, upstream code, paper)
- Available recipes with descriptions
- Environment variables with defaults
- Container image and validated hardware

## Arguments

$ARGUMENTS
