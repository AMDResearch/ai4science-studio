# Agents for Science in AI4Science Studio: design and roadmap

| Field | Value |
|---|---|
| Status | Draft v0.3, for team review |
| Target | One AI factory (single site) |
| Replaces | [`archive/agents-for-science-v0.2.md`](archive/agents-for-science-v0.2.md) |
| Last updated | 2026-10-02 |

## 1. Objective

Run AI4Science Studio (AI4S Studio) agents on an AI factory without a person at the terminal, using open LLMs served on the same AI factory.

The first product is an automatic performance report for an ORBIT-2 or HydraGNN training job. Each claim in the report cites telemetry or tool output.

Later phases add provenance export, model routing and declarative campaigns. Applications such as an AI Scientist, the cofolding agent or matsim-agents can then launch whole campaigns through the same entry points.

## 2. Terms

| Term | Meaning |
|---|---|
| AI factory | An HPC cluster with AMD Instinct GPUs, ROCm, SLURM, Apptainer and shared storage |
| Agent | A folder in the agent registry with `agent.yaml` (settings) and, if it uses an LLM, `playbook.md` (instructions) |
| `llm` | The agent setting that names one model alias, or `none` |
| Recipe | An existing, validated script in a model's `examples/` folder |
| Model agent | An agent that runs one recipe of one model, named `<model>:<task>`; `llm: none` by default |
| Engine | The program that executes a playbook with an LLM: OpenCode or Claude Code |
| Agent runner | Runs one agent, as a SLURM job or as a step inside an existing allocation |
| Model alias | A name such as `ai4s/small` that LiteLLM maps to a served model |
| Run | One execution of an agent, with its own run record |
| Run record | The run's folder: inputs, transcript, `result.json`, status |
| Campaign | A file listing agents, their order and human gates (Phase 3) |

## 3. Scope

**In scope:** one AI factory; SLURM and Apptainer; models with a validated recipe on that AI factory; OpenCode and Claude Code as engines.

**Out of scope:**
- agents that span several sites;
- Kubernetes;
- agent-to-agent messaging and A2A;
- an AI4S Studio-owned LLM loop, unless Phase 1 shows the engines cannot do the job;
- self-healing of running jobs;
- network isolation.

## 4. Functional requirements

| ID | Requirement | Phase |
|---|---|---|
| F1 | `ai4s agent list`, `show`, `new`, `validate` manage agents from the repo and from a user directory | 1 |
| F2 | `ai4s run <agent>` runs one agent as a SLURM job, or as a step when called inside an existing allocation | 1 |
| F3 | With `llm: none`, the runner submits the agent's recipe, waits, checks its outputs and writes `result.json`; it starts no engine and calls no LLM | 1 |
| F4 | With a model alias, the runner starts the engine non-interactively, inside Apptainer, against the LiteLLM endpoint, with the agent's playbook | 1 |
| F5 | `--llm <alias>` enables an LLM on a model agent, to choose parameters or diagnose a failure; the recipe is unchanged | 1 |
| F6 | `--then <agent>` runs a follow-on agent when the current one ends, as a SLURM dependency job or as the next step | 1 |
| F7 | Every run writes a run record: inputs, git commit, engine, model, commands and outputs, token counts, timings, `result.json`, status | 1 |
| F8 | An agent may only use the tools and commands in its `allow` list; SLURM actions are limited to the user's own jobs | 1 |
| F9 | vLLM serves gpt-oss-20b and gpt-oss-120b, with a LiteLLM sidecar and the endpoint written to shared storage; `ai4s llm start`, `status`, `stop` manage it. Placement is open question 1 | 1 |
| F10 | The site maps model aliases to served models | 1 |
| F11 | The `perf-report` agent produces `perf_report.md` and `findings.json` for an ORBIT-2 or HydraGNN job; deterministic checks run first, and the LLM ranks and explains them | 1 |
| F12 | Each report is appended to a per-model index, so past runs can be compared | 1 |
| F13 | An eval runner grades the perf report against recorded benchmark runs, alongside the existing contract cases | 1 |
| F14 | The same functions are available as a stdio MCP server for Cursor, Claude Code, OpenCode and HPC Assistant | 1 |
| F15 | Run records export to Flowcept | 2 |
| F16 | A model agent exists for every model with a validated recipe; an application eval track uses the metrics in `result.json` | 2 |
| F17 | A model router in the LLM sidecar picks an alias per request; routing choices are evaluated with F13 | 2 |
| F18 | The `optimizer` agent ships, with a human approval before each job it launches | 2 |
| F19 | `ai4s campaign run`, `status`, `approve` execute a campaign file; steps pass values through `result.json` | 3 |
| F20 | An entry agent can write a campaign file and launch it through the MCP server | 3 |

## 5. Non-functional requirements

| ID | Requirement |
|---|---|
| N1 | On-prem models by default; hosted models only when the site enables them |
| N2 | No long-running service on the login node; LLM serving and agent runs are SLURM jobs or steps |
| N3 | The run record is written by the agent runner, not by the LLM |
| N4 | An agent with `llm: none` needs no engine and no LLM endpoint |
| N5 | `make check` validates agents and runs one report with a recorded LLM reply; no GPU needed |
| N6 | The engine can be replaced without changing the CLI, MCP tools, `agent.yaml` or the run record |

## 6. Architecture

![AI4S Studio architecture: coding and science agents, AI4S Studio, and the AI hardware/software stack with an LLM sidecar](img/agents-for-science-architecture.svg)

The diagram shows Phase 1. The model router (dashed) is Phase 2.

- **Top row:** coding agents (Claude Code, OpenCode), HPC Assistant, and science agents such as the cofolding agent, matsim-agents and a co-scientist. Each uses AI4S Studio through the CLI or the MCP server.
- **AI4S Studio** holds the CLI, the MCP server, the agent runner and the agent registry.
- **Agent interfaces** are skills, MCP tools and recipes. Agents use the tools and the platform only through them. AI4S Studio ships the recipes and its own skills; tool teams can ship interfaces for their tools.
- **LLM sidecar** runs on the same GPUs. Only agents with an LLM use it.

How a run executes:

- The agent runner runs each agent as a SLURM job, or as a step if called inside an allocation.
- With `llm: none`, it runs the agent's recipe directly. Otherwise it starts the engine with the agent's playbook, allow list and model alias.
- The recipe's own jobs, such as ORBIT-2 training, are ordinary SLURM jobs.
- The agent runner writes a run record for every run (F7).

Flow for the first product:

1. `ai4s run ORBIT-2:perf-analysis --then perf-report` runs the ORBIT-2 model agent with `llm: none`. It submits the training job with Omnistat on.
2. When the job ends, the `perf-report` agent runs.
3. Its engine runs the deterministic checks on the job's telemetry and traces, uses the LLM to rank and explain the findings, and writes `perf_report.md` and `findings.json` into the run record.

Where later phases attach:

- Phase 2: the model router in the LLM sidecar; run records export to Flowcept.
- Phase 3: a campaign runner above the agent runner.

AI4S Studio ships as one Python package with three entry points: the CLI, the stdio MCP server, and the skills. The MCP server is started by the host for each session, so nothing runs on the login node between sessions. Installation is with `uv` from git, or as an Lmod module.

## 7. Agents shipped with AI4S Studio

| Agent | Purpose | LLM | Phase |
|---|---|---|---|
| `perf-report` | Post-job performance report, from the existing analyst, verifier and synthesizer playbooks | Yes | 1 |
| Model agents (for example `ORBIT-2:train`, `HydraGNN:train`, `MatterGen:generate`) | Run one recipe and write `result.json`; an LLM can be enabled to choose parameters or diagnose a failure | Optional | 1 |
| `optimizer` | Proposes and runs the next configuration, from the existing perf-optimizer-loop recipes; human approval per job | Yes | 2 |
| `tempering` | Takes an existing runbook, playbook or recipe skill and hardens it for a target system in a tight loop | Yes | Later |

Provenance is recorded by the agent runner (N3), not by an agent.

## 8. Roadmap

### Phase 1: headless runs and the first report

| Item | Work |
|---|---|
| Headless runtime | CLI, agent runner (job or step), `llm: none`, run record, allow lists, OpenCode and Claude Code engines, `--then` |
| On-prem LLMs | vLLM serving with LiteLLM sidecar; `ai4s llm start`, `status`, `stop`; endpoint file; two aliases |
| Model agents | ORBIT-2 and HydraGNN model agents with `result.json` |
| Performance report | `perf-report` agent; per-model report index |
| Evaluation | Three to five recorded benchmark runs; eval runner; existing contract cases |
| Entry points | CLI, stdio MCP server, skills |

Exit criteria:
- An ORBIT-2 and a HydraGNN training job each produce a report automatically, using an on-prem model.
- Every claim in each report cites telemetry or tool output.
- The report finds the known problem in each benchmark run, and raises no high-severity finding on the healthy runs.
- A model agent with `llm: none` runs with no LLM service up.
- The same report can be started from the CLI and from an MCP host.

### Phase 2: provenance, coverage, routing

| Item | Work |
|---|---|
| Provenance | Flowcept export of run records |
| Coverage | Model agents for every model with a validated recipe; application eval track |
| Model routing | Model router in the LLM sidecar, measured against the eval runner |
| Agents | `optimizer` with human approval |

Exit criteria:
- Run records appear in Flowcept.
- Routing beats the static aliases on the eval set at equal or lower cost.

### Phase 3: campaigns

| Item | Work |
|---|---|
| Campaign runner | Campaign files with steps, order, human gates, and values passed through `result.json` |
| Entry agents | Cursor, Claude Code or HPC Assistant writes a campaign file and launches it through MCP |
| Agent cards | Only once a consumer for them exists |

Exit criterion: a campaign of training, report, human review and optimization runs end to end.

### After Phase 3

Applications such as an AI Scientist, the cofolding agent, matsim-agents or a co-scientist build their own agents, and launch campaigns through the CLI or MCP.

## 9. Open questions

1. LLM serving placement: one shared serving job with an idle timeout, one serving job per agent run, or a serving step in the same allocation as the agent steps? The trade-offs are GPUs held while idle, queue and model-load wait per run, and allocation length.
2. Can a login node reach a compute node over HTTP? If not, interactive MCP sessions on the login node need another route to the endpoint.
3. Do OpenCode and Claude Code call tools reliably with gpt-oss through LiteLLM? This decides whether an AI4S Studio-owned loop is ever needed (N6).
4. Where does the per-model report index live, and who can read it?
5. Which benchmark runs are recorded first, and who records them?
6. Should LLM agents and `llm: none` agents stay under one runner? If reviewers find the term "agent" misleading for a recipe run, the recipe path can split into its own runner without changing the run record.
