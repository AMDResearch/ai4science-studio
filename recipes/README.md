# Recipes

In **AI4Science Studio**, recipes live **next to the model they target**, not in this top-level folder. Use this file as an **index** and mental map.

## Where recipes live

```text
<domain>/models/<model-folder>/recipes/
```

Examples of what to put there:

- Shell or Python entrypoints for **inference** or **batch scoring**
- **Fine-tuning** or **training** scripts with documented CLI arguments
- Short **Markdown** runbooks if scripts live upstream and you only document the exact versions and commands
- **Large-data** runbooks may describe **Globus** or **Hugging Face Hub CLI** staging onto a cluster’s **shared** filesystem (scratch or project space), not check-ins to git

## Domains

| Domain | Models index |
|--------|----------------|
| Earth science (climate, weather, geospatial) | [`../earth_science/models/README.md`](../earth_science/models/README.md) |
| Material science | [`../material_science/models/README.md`](../material_science/models/README.md) |
| Protein folding | [`../protein_folding/models/README.md`](../protein_folding/models/README.md) |
| Healthcare | [`../healthcare/models/README.md`](../healthcare/models/README.md) |

## Conventions

- One **Hugging Face model** (or one tightly coupled family) per `models/<model-folder>/` directory.
- Prefer **small** recipe repos: pin dependencies in comments or a local `requirements.txt` inside the model folder when needed.
- Link to the **official** Hugging Face model card and any **GitHub** test or training repo maintained by the model authors.

## Template

Copy [`../_template/`](../_template/) when adding a new model directory in any domain.
