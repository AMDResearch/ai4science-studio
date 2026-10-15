import { useEffect, useState } from 'react'
import { useStore } from '../store'
import { api } from '../api'
import { Spinner, StepHeader } from '../components/ui'

// ── Domain meta ───────────────────────────────────────────────────────────────
const DOMAINS = [
  { id: 'all',                label: 'All Models',        icon: '⊞' },
  { id: 'earth_science',      label: 'Earth Science',     icon: '🌍' },
  { id: 'material_science',   label: 'Materials',         icon: '⚛️' },
  { id: 'healthcare',         label: 'Healthcare',        icon: '💊' },
  { id: 'physics_simulation', label: 'Physics Sim.',      icon: '🔬' },
  { id: 'protein_folding',    label: 'Protein Folding',   icon: '🧬' },
  { id: 'methods',            label: 'Methods & AMD Stack', icon: '🔩' },
  { id: 'add',                label: 'Add Your Model',    icon: '＋' },
]

const DOMAIN_COLORS = {
  earth_science:      '#38bdf8',
  material_science:   '#a78bfa',
  healthcare:         '#34d399',
  physics_simulation: '#fb923c',
  protein_folding:    '#f472b6',
}

// License badge color
function licColor(lic) {
  if (!lic || lic.toLowerCase().includes('see')) return '#52525b'
  if (lic.startsWith('MIT') || lic.startsWith('Apache') || lic.startsWith('BSD')) return '#21c77a'
  if (lic.includes('NC') || lic.includes('NonCommercial')) return '#f5a524'
  return '#52525b'
}

// ── Model card ────────────────────────────────────────────────────────────────
function ModelCard({ m, onSelect }) {
  const dc = DOMAIN_COLORS[m.domain] || '#52525b'
  return (
    <div className="card clickable" onClick={() => onSelect(m)}
      style={{ padding: '1rem', display: 'flex', flexDirection: 'column', gap: '.5rem',
        cursor: 'pointer', transition: 'transform .1s', userSelect: 'none' }}
      onMouseEnter={e => { e.currentTarget.style.transform = 'translateY(-2px)' }}
      onMouseLeave={e => { e.currentTarget.style.transform = '' }}>

      {/* Domain + name row */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: '.5rem' }}>
        <span className="badge" style={{ background: dc + '22', color: dc, border: `1px solid ${dc}55`,
          fontSize: '.65rem', whiteSpace: 'nowrap' }}>
          {m.domain?.replace(/_/g, ' ')}
        </span>
        {m.vram_gb && (
          <span style={{ fontSize: '.65rem', color: '#f5a524', fontWeight: 700, flexShrink: 0 }}>
            {m.vram_gb} GB
          </span>
        )}
      </div>

      <div style={{ fontWeight: 800, fontSize: '.95rem', color: '#f5f5f7', lineHeight: 1.25 }}>
        {m.name || m.slug}
      </div>

      {m.task && (
        <div style={{ fontSize: '.75rem', color: '#a1a1aa', lineHeight: 1.4, flexGrow: 1 }}>
          {m.task.length > 80 ? m.task.slice(0, 80) + '…' : m.task}
        </div>
      )}

      {/* Tasks available */}
      {m.tasks_available?.length > 0 && (
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '.3rem' }}>
          {m.tasks_available.map(t => (
            <span key={t} className="badge badge-info"
              style={{ fontSize: '.62rem', padding: '.1rem .4rem' }}>{t}</span>
          ))}
        </div>
      )}

      {/* Hardware + license */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '.3rem' }}>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '.25rem' }}>
          {(m.validated_hardware || []).slice(0, 2).map(h => (
            <span key={h} className="badge badge-amd"
              style={{ fontSize: '.6rem', padding: '.1rem .35rem' }}>{h}</span>
          ))}
        </div>
        <span className="badge" style={{ fontSize: '.6rem', padding: '.1rem .35rem',
          background: licColor(m.license) + '22', color: licColor(m.license) }}>
          {m.license?.length > 20 ? m.license.slice(0, 18) + '…' : (m.license || '—')}
        </span>
      </div>

      {/* HF ID */}
      {m.hf_id && m.hf_id !== 'N/A' && (
        <div style={{ fontSize: '.6rem', color: '#52525b', overflow: 'hidden',
          textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
          🤗 {m.hf_id}
        </div>
      )}

      <div style={{ fontSize: '.68rem', color: '#ED1C24', fontWeight: 600, marginTop: 'auto' }}>
        Launch → Configure
      </div>
    </div>
  )
}

// ── Methods & AMD Stack page ──────────────────────────────────────────────────
function Card({ title, color = '#38bdf8', children }) {
  return (
    <div className="card" style={{ padding: '1rem' }}>
      <div style={{ fontWeight: 800, fontSize: '.9rem', color, marginBottom: '.65rem',
        borderBottom: `1px solid ${color}33`, paddingBottom: '.4rem' }}>{title}</div>
      {children}
    </div>
  )
}

function Tag({ children, color = '#52525b' }) {
  return (
    <span style={{ display: 'inline-block', padding: '.12rem .45rem', borderRadius: '9999px',
      fontSize: '.67rem', fontWeight: 700, background: color + '22', color,
      border: `1px solid ${color}55`, marginRight: '.25rem', marginBottom: '.25rem' }}>
      {children}
    </span>
  )
}

function Row({ label, value, sub }) {
  return (
    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline',
      padding: '.25rem 0', borderBottom: '1px solid #1a1a1c', fontSize: '.8rem', gap: '1rem' }}>
      <span style={{ color: '#a1a1aa', flexShrink: 0 }}>{label}</span>
      <div style={{ textAlign: 'right' }}>
        <span style={{ color: '#f5f5f7', fontWeight: 600 }}>{value}</span>
        {sub && <span style={{ color: '#52525b', fontSize: '.7rem', marginLeft: '.35rem' }}>{sub}</span>}
      </div>
    </div>
  )
}

function MethodsPage() {
  const AMD = '#ED1C24'
  const BLUE = '#38bdf8'
  const GREEN = '#21c77a'
  const AMBER = '#f5a524'
  const PURPLE = '#a78bfa'

  // Cross-list models to methods
  const MODEL_METHOD_MAP = [
    { model: 'ORBIT-2 / Aurora / GenCast / PanguWeather', methods: ['Physics-constrained ML', 'Vision Foundation Model', 'Diffusion (generative ensemble)'] },
    { model: 'StormCast / ArchesWeather', methods: ['Convection-allowing emulation', 'Autoregressive rollout'] },
    { model: 'HydraGNN', methods: ['Graph Neural Network (GNN)', 'Training from scratch', 'Fine-tuning'] },
    { model: 'MatterGen', methods: ['Diffusion (crystal generation)', 'Generative materials design'] },
    { model: 'GP-MoLFormer', methods: ['Autoregressive LLM', 'Pair-tuning (PEFT)', 'Molecule generation'] },
    { model: 'SwinUNETR', methods: ['3D Medical segmentation', 'Swin Transformer', 'Transfer learning'] },
    { model: 'REINVENT4', methods: ['Reinforcement learning', 'Molecular optimization'] },
    { model: 'MATEY / Walrus', methods: ['Spatiotemporal surrogate', 'Foundation model (continuum)'] },
  ]

  return (
    <div>
      <div className="section-label" style={{ marginBottom: '1rem' }}>AI-for-Science Methods</div>
      <p style={{ color: '#a1a1aa', fontSize: '.85rem', lineHeight: 1.65, marginBottom: '1.5rem', maxWidth: 780 }}>
        AI-for-science encompasses a spectrum of approaches — from pure data-driven surrogates to
        physics-constrained hybrid models. Understanding which method fits which problem is key to
        deploying them effectively on AMD hardware.
      </p>

      {/* Method taxonomy */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill,minmax(290px,1fr))', gap: '1rem', marginBottom: '1.5rem' }}>
        <Card title="Physics-Based Models" color={BLUE}>
          <p style={{ color: '#a1a1aa', fontSize: '.78rem', lineHeight: 1.5, marginBottom: '.6rem' }}>
            Solve PDEs (Navier-Stokes, Schrödinger, Maxwell) via finite-element, finite-volume, or
            spectral methods. Ground truth for surrogates; interpretable; computationally expensive
            at scale.
          </p>
          <div style={{ fontSize: '.75rem', color: '#71717a' }}>
            <strong style={{ color: '#f5f5f7' }}>Use when:</strong> high physical fidelity required,
            no training data available, or as a validation baseline.
          </div>
          <div style={{ marginTop: '.5rem' }}><Tag color={BLUE}>CFD</Tag><Tag color={BLUE}>FEM/FVM</Tag><Tag color={BLUE}>MD</Tag><Tag color={BLUE}>DFT</Tag></div>
        </Card>

        <Card title="Surrogate / Emulation Models" color={GREEN}>
          <p style={{ color: '#a1a1aa', fontSize: '.78rem', lineHeight: 1.5, marginBottom: '.6rem' }}>
            Trained on physics-simulation output to reproduce outputs orders of magnitude faster.
            Examples: weather emulators (GenCast, ORBIT-2), MLIP energy models (HydraGNN). Trade-off:
            fast but distribution-limited.
          </p>
          <div style={{ fontSize: '.75rem', color: '#71717a' }}>
            <strong style={{ color: '#f5f5f7' }}>Use when:</strong> ensemble/uncertainty quantification,
            interactive exploration, or real-time inference.
          </div>
          <div style={{ marginTop: '.5rem' }}><Tag color={GREEN}>GNN</Tag><Tag color={GREEN}>Transformer</Tag><Tag color={GREEN}>ViT</Tag><Tag color={GREEN}>Diffusion</Tag></div>
        </Card>

        <Card title="Training from Scratch" color={AMBER}>
          <p style={{ color: '#a1a1aa', fontSize: '.78rem', lineHeight: 1.5, marginBottom: '.6rem' }}>
            Build task-specific models on domain data (e.g., DFT datasets). Full control over
            architecture and featurizer. Requires large labeled datasets and significant compute.
            HydraGNN trained on 268k Alexandria DFT structures is a studio example.
          </p>
          <div style={{ marginTop: '.5rem' }}>
            <Tag color={AMBER}>1-GPU baseline</Tag><Tag color={AMBER}>8-GPU DDP</Tag><Tag color={AMBER}>Multi-node</Tag>
          </div>
        </Card>

        <Card title="Fine-Tuning / PEFT" color={PURPLE}>
          <p style={{ color: '#a1a1aa', fontSize: '.78rem', lineHeight: 1.5, marginBottom: '.6rem' }}>
            Adapt a pre-trained foundation model to a new domain with minimal data and compute.
            GP-MoLFormer pair-tuning trains only soft-prompt tokens (backbone frozen) to steer
            generation toward QED / logP / DRD2. Efficient and sample-efficient.
          </p>
          <div style={{ marginTop: '.5rem' }}>
            <Tag color={PURPLE}>Pair-tuning</Tag><Tag color={PURPLE}>LoRA</Tag><Tag color={PURPLE}>Prompt tuning</Tag><Tag color={PURPLE}>Adapter</Tag>
          </div>
        </Card>

        <Card title="Generative Models" color="#f472b6">
          <p style={{ color: '#a1a1aa', fontSize: '.78rem', lineHeight: 1.5, marginBottom: '.6rem' }}>
            Diffusion and autoregressive models for de-novo design: crystal structures (MatterGen),
            molecules (GP-MoLFormer), ensemble weather forecasts (GenCast). Sample from a learned
            distribution rather than minimize a single-point loss.
          </p>
          <div style={{ marginTop: '.5rem' }}>
            <Tag color="#f472b6">Diffusion</Tag><Tag color="#f472b6">Autoregressive LM</Tag><Tag color="#f472b6">SMILES</Tag>
          </div>
        </Card>

        <Card title="Physics-Constrained Hybrid" color="#fb923c">
          <p style={{ color: '#a1a1aa', fontSize: '.78rem', lineHeight: 1.5, marginBottom: '.6rem' }}>
            Embed physical laws (conservation of mass/energy, symmetries, equivariance) directly
            into the model architecture or loss. Better extrapolation than pure data-driven;
            harder to train. NeuralGCM, E3-equivariant GNNs.
          </p>
          <div style={{ marginTop: '.5rem' }}>
            <Tag color="#fb923c">Equivariance</Tag><Tag color="#fb923c">PDE-constrained</Tag><Tag color="#fb923c">Hybrid GCM</Tag>
          </div>
        </Card>
      </div>

      {/* Model ↔ method cross-list */}
      <div className="section-label" style={{ marginBottom: '.75rem' }}>Studio Models by Method</div>
      <div className="card" style={{ padding: '.75rem', marginBottom: '1.5rem', overflowX: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '.78rem' }}>
          <thead>
            <tr style={{ color: '#52525b' }}>
              <th style={{ textAlign: 'left', padding: '.3rem .5rem', borderBottom: '1px solid #27272a' }}>Model</th>
              <th style={{ textAlign: 'left', padding: '.3rem .5rem', borderBottom: '1px solid #27272a' }}>Methods</th>
            </tr>
          </thead>
          <tbody>
            {MODEL_METHOD_MAP.map(({ model, methods }) => (
              <tr key={model} style={{ borderBottom: '1px solid #1a1a1c' }}>
                <td style={{ padding: '.4rem .5rem', color: '#a1a1aa', verticalAlign: 'top', whiteSpace: 'nowrap' }}>{model}</td>
                <td style={{ padding: '.4rem .5rem' }}>
                  {methods.map(m => <Tag key={m} color={BLUE}>{m}</Tag>)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* AMD Hardware Stack */}
      <div className="section-label" style={{ marginBottom: '.75rem', color: AMD }}>AMD Hardware Ecosystem</div>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill,minmax(280px,1fr))', gap: '1rem', marginBottom: '1.5rem' }}>

        <Card title="AMD Instinct — Data Center / HPC" color={AMD}>
          <Row label="MI350X (gfx950)" value="288 GB HBM3e" sub="~2.6 PF BF16" />
          <Row label="MI430X (gfx1030)" value="Next-gen Instinct" sub="2026 roadmap" />
          <Row label="MI455 (codename)" value="Helios platform" sub="Rack-scale" />
          <Row label="MI300X (gfx942)" value="192 GB HBM3" sub="~1.3 PF FP8" />
          <Row label="MI300A (gfx942)" value="128 GB HBM3 (APU)" sub="CPU+GPU unified" />
          <Row label="MI250X (gfx90a)" value="128 GB HBM2e" sub="Frontier (exascale)" />
          <div style={{ marginTop: '.5rem', fontSize: '.7rem', color: '#52525b' }}>
            Covers: AI training, HPC simulation, mixed-precision inference. All supported by ROCm.
          </div>
        </Card>

        <Card title="Helios Rack System" color={AMD}>
          <p style={{ color: '#a1a1aa', fontSize: '.78rem', lineHeight: 1.5, marginBottom: '.5rem' }}>
            AMD's rack-scale AI infrastructure platform pairing MI455 GPUs with EPYC Zen 5 CPUs and
            high-bandwidth Infinity Fabric interconnect. Designed for large-scale training and
            multi-node inference clusters.
          </p>
          <Row label="GPUs per rack" value="Up to 72 MI455" sub="fully interconnected" />
          <Row label="CPU" value="EPYC Turin (Zen 5)" sub="192-core / socket" />
          <Row label="Interconnect" value="Infinity Fabric 4.0" sub="896 GB/s bidirectional" />
        </Card>

        <Card title="AMD Radeon PRO — Workstation" color={AMBER}>
          <Row label="Radeon PRO W7900X" value="48 GB GDDR6" sub="Workstation GPU" />
          <Row label="Radeon PRO W7800" value="32 GB GDDR6" sub="Mid-range pro" />
          <div style={{ marginTop: '.5rem', fontSize: '.7rem', color: '#52525b' }}>
            ROCm 7+ supports Radeon PRO for inference and development workflows.
            Full training workloads preferred on Instinct.
          </div>
        </Card>

        <Card title="AMD FPGA — Edge Inference" color={GREEN}>
          <Row label="Alveo U55C" value="HBM2 8 GB" sub="Datacenter inference" />
          <Row label="Alveo V80" value="HBM2e 32 GB" sub="Networking + AI" />
          <Row label="Versal AI Core" value="AI Engines (400 TOPS)" sub="Embedded / edge" />
          <div style={{ marginTop: '.5rem', fontSize: '.7rem', color: '#52525b' }}>
            Ideal for ultra-low-latency inference, sensor fusion, and network-line-rate ML.
          </div>
        </Card>

        <Card title="AMD EPYC — CPU Compute" color={PURPLE}>
          <Row label="EPYC 9965 (Turin)" value="192 cores" sub="Zen 5, 2 sockets" />
          <Row label="EPYC 9654 (Genoa)" value="96 cores" sub="Zen 4, HPC workhorse" />
          <Row label="EPYC 7763 (Milan)" value="64 cores" sub="Frontier nodes" />
          <div style={{ marginTop: '.5rem', fontSize: '.7rem', color: '#52525b' }}>
            Used as the host processor on every AMD Instinct node. Critical for data-loading,
            MPI communication, and pre/post-processing pipelines.
          </div>
        </Card>

        <Card title="Deployment Tiers" color={BLUE}>
          <Row label="Laptop / Edge" value="Ryzen AI Max (Strix)" sub="NPU XDNA 2, 50 TOPS" />
          <Row label="Halo Mini-PC (Dev Platform)" value="Ryzen AI Max+ 395" sub="128 GB LPDDR5x · 60 FP16 TFLOPS · 50 NPU TOPS · ROCm" />
          <Row label="Halo (coming soon)" value="Ryzen AI Max+ PRO 495" sub="192 GB unified memory" />
          <Row label="Workstation" value="Radeon PRO W7900X" sub="48 GB VRAM, ROCm dev" />
          <Row label="Small Cluster" value="4–16 × MI300X/MI350X" sub="Vultr Lux (this cluster)" />
          <Row label="Supercomputer" value="Frontier (37,888 × MI250X)" sub="1.1 ExaFLOP Rmax, ORNL" />
          <Row label="Rack System" value="Helios (MI455 / MI430X)" sub="Multi-exaFLOP scale" />
        </Card>
      </div>

      {/* AMD Software Stack */}
      <div className="section-label" style={{ marginBottom: '.75rem' }}>AMD ROCm Software Stack</div>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill,minmax(280px,1fr))', gap: '1rem', marginBottom: '1.5rem' }}>

        <Card title="ROCm 7.14 (July 2026)" color={AMD}>
          <p style={{ color: '#a1a1aa', fontSize: '.78rem', lineHeight: 1.5, marginBottom: '.6rem' }}>
            "TheRock" modular build system (7.9+). Single release covering Linux + Windows.
            Modular domain-specific SDKs for AI, data science, and HPC.
          </p>
          <Row label="PyTorch" value="2.12.0" sub="native ROCm" />
          <Row label="JAX" value="0.10.0" sub="XLA + PJRT plugin" />
          <Row label="vLLM" value="0.23.0" sub="production inference" />
          <Row label="TensorFlow" value="2.18" sub="ROCm upstream" />
          <Row label="Production stream" value="ROCm 6.4.4" sub="Sept 2025, widely deployed" />
        </Card>

        <Card title="Math & Communication Libraries" color={BLUE}>
          <div>
            {['rocBLAS (GEMM/BLAS)', 'rocFFT (FFT)', 'rocSPARSE (sparse BLAS)',
              'rocRAND (RNG)', 'rocSOLVER (dense linear algebra)',
              'MIOpen (DNN primitives)', 'RCCL (collective comms)',
              'hipBLASLt (LtGEMM, flash attention)'].map(lib => (
              <div key={lib} style={{ padding: '.18rem 0', borderBottom: '1px solid #1a1a1c',
                fontSize: '.75rem', color: '#a1a1aa' }}>{lib}</div>
            ))}
          </div>
        </Card>

        <Card title="Profiling & Optimization" color={GREEN}>
          <Row label="ROCprofiler-SDK" value="rocprof v3" sub="GPU trace + counters" />
          <Row label="Omniperf" value="system-wide" sub="roofline, bottlenecks" />
          <Row label="Omnistat" value="time-series metrics" sub="VictoriaMetrics / PromQL" />
          <Row label="TraceLens" value="kineto trace" sub="op-level analysis" />
          <Row label="TunableOp" value="GEMM auto-select" sub="MI300X+ only" />
          <Row label="rocm-smi" value="CLI" sub="GPU health, utilization" />
        </Card>

        <Card title="Containerization" color={AMBER}>
          <p style={{ color: '#a1a1aa', fontSize: '.78rem', lineHeight: 1.5, marginBottom: '.6rem' }}>
            This studio uses Apptainer (formerly Singularity) with ext3 overlay images for
            unprivileged HPC execution. Docker is supported for single-node dev workflows.
          </p>
          <Row label="Base image" value="ROCm PyTorch 2.10" sub="rocm7.2.2 / ubuntu24.04" />
          <Row label="Overlay pattern" value="NFS staging + tar" sub="fuse2fs write" />
          <Row label="Container runtime" value="Apptainer + Docker" sub="dual-path scripts" />
        </Card>

        <Card title="MPI & Distributed" color={PURPLE}>
          <Row label="OpenMPI" value="5.x" sub="PMIx-based launch" />
          <Row label="RCCL" value="allreduce" sub="XGMI + ionic fabric" />
          <Row label="ANP plugin" value="librccl-anp.so" sub="RoCEv2 native transport" />
          <Row label="ADIOS2" value="2.12.1" sub="parallel scientific I/O" />
          <div style={{ marginTop: '.5rem', fontSize: '.7rem', color: '#52525b' }}>
            Multi-node: srun --mpi=pmix inside Apptainer; RCCL for GPU collective, MPI for CPU coordination.
          </div>
        </Card>

        <Card title="AMD AI for Science — Genesis Mission" color={AMD}>
          <p style={{ color: '#a1a1aa', fontSize: '.78rem', lineHeight: 1.5, marginBottom: '.6rem' }}>
            AMD's initiative to accelerate open AI-for-science on AMD Instinct hardware.
            In-house AMD Research and AMD AI4Science team provide:
          </p>
          <div>
            {[
              'HPC cluster access (Frontier, Lux)',
              'ROCm porting and optimization support',
              'Performance profiling (Omnistat / TraceLens)',
              'Recipe curation (this studio)',
              'Collaboration on publications and benchmarks',
              'Direct engineering engagement for Genesis Mission projects',
            ].map(item => (
              <div key={item} style={{ display: 'flex', gap: '.4rem', padding: '.18rem 0',
                borderBottom: '1px solid #1a1a1c', fontSize: '.75rem', color: '#a1a1aa' }}>
                <span style={{ color: AMD, flexShrink: 0 }}>▸</span>{item}
              </div>
            ))}
          </div>
        </Card>
      </div>

      {/* How to succeed on AMD */}
      <div className="section-label" style={{ marginBottom: '.75rem' }}>Using This Studio for Your Science</div>
      <div className="card" style={{ padding: '1rem', marginBottom: '1rem' }}>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill,minmax(220px,1fr))', gap: '.75rem' }}>
          {[
            { step: '1', title: 'Pick your method', body: 'Start with the taxonomy above — are you training a surrogate, fine-tuning a foundation model, or running generative design?' },
            { step: '2', title: 'Find a recipe', body: 'Browse the Model Catalog. Each model card links to inference, training, and fine-tuning recipes validated on AMD Instinct.' },
            { step: '3', title: 'Run the demo', body: 'Use Demo mode for instant results (no GPU). Switch to Live for real SLURM jobs on MI355X nodes.' },
            { step: '4', title: 'Adapt to your data', body: 'Clone the model\'s examples/ scripts, swap in your dataset, and re-run. Every script is parameterized via env vars.' },
            { step: '5', title: 'Optimize', body: 'Use the perf-analysis recipe to profile with Omnistat + TraceLens, then iterate with the perf-optimizer-loop.' },
          ].map(({ step, title, body }) => (
            <div key={step} style={{ display: 'flex', gap: '.6rem', alignItems: 'flex-start' }}>
              <div style={{ width: 24, height: 24, borderRadius: '50%', background: AMD, color: '#fff',
                fontWeight: 800, fontSize: '.8rem', flexShrink: 0,
                display: 'flex', alignItems: 'center', justifyContent: 'center' }}>{step}</div>
              <div>
                <div style={{ fontWeight: 700, fontSize: '.85rem', color: '#f5f5f7', marginBottom: '.2rem' }}>{title}</div>
                <div style={{ fontSize: '.75rem', color: '#a1a1aa', lineHeight: 1.5 }}>{body}</div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

// ── "Add your model" guide ────────────────────────────────────────────────────
function AddModelGuide() {
  return (
    <div style={{ maxWidth: 780 }}>
      <div className="section-label" style={{ marginBottom: '.75rem' }}>How to Add Your Model</div>
      <p style={{ color: '#a1a1aa', lineHeight: 1.7, marginBottom: '1.5rem', fontSize: '.88rem' }}>
        AI4Science Studio uses an <strong style={{ color: '#f5f5f7' }}>agent-first recipe</strong> pattern —
        your model is described in machine-readable YAML manifests that the studio reads to discover,
        configure, and launch it. No framework code to modify.
      </p>

      {[
        { n: 1, title: 'Pick a domain folder',
          body: 'Choose the closest scientific domain: earth_science/, material_science/, healthcare/, physics_simulation/, or protein_folding/.' },
        { n: 2, title: 'Copy the template',
          body: null,
          code: 'cp -r _template/ <domain>/models/<your-slug>/' },
        { n: 3, title: 'Fill in README.md',
          body: 'Required fields: Hugging Face model id (or "N/A" with a fetch snippet), task description, license (SPDX id), upstream code URL, and paper citation.' },
        { n: 4, title: 'Add recipes',
          body: 'Create one subfolder per task under recipes/ — e.g. recipes/inference/, recipes/finetune/ — each with a README.md walkthrough.' },
        { n: 5, title: 'Add example scripts',
          body: 'Place docker_run.sh, run_<task>.sh, sbatch_<task>_amd.sh, and preflight_<slug>.py in examples/. All scripts must be chmod +x.',
          code: 'chmod +x examples/*.sh examples/*.py' },
        { n: 6, title: 'Create model.yaml',
          code: `name: YourModel
hf_id: org/model-name
license: Apache-2.0
task: "What your model does"
domain: material_science
container_image: rocm/pytorch:rocm7.2.2_ubuntu24.04_py3.12_pytorch_release_2.10.0
validated_hardware: [MI300X]
tasks_available: [inference, finetune]` },
        { n: 7, title: 'Register in models.yaml',
          body: 'Add your model slug to the root models.yaml index so the studio can discover it.',
          code: `- slug: YourModel
  domain: material_science` },
        { n: 8, title: 'Conventions to follow',
          body: null,
          items: [
            'Link upstream code, don\'t vendor it',
            'Non-HF weights: set hf_id: N/A and add a fetch snippet',
            'AMD/ROCm notes go in recipes only when validated',
            'Never commit checkpoints, datasets, or .env files',
            'HCLS models must include a research-only disclaimer',
          ] },
      ].map(({ n, title, body, code, items }) => (
        <div key={n} style={{ marginBottom: '1.25rem', display: 'flex', gap: '1rem', alignItems: 'flex-start' }}>
          <div style={{ width: 28, height: 28, borderRadius: '50%', background: '#ED1C24',
            color: '#fff', fontWeight: 800, fontSize: '.82rem', flexShrink: 0,
            display: 'flex', alignItems: 'center', justifyContent: 'center' }}>{n}</div>
          <div style={{ flex: 1 }}>
            <div style={{ fontWeight: 700, fontSize: '.9rem', color: '#f5f5f7', marginBottom: '.3rem' }}>{title}</div>
            {body && <p style={{ color: '#a1a1aa', fontSize: '.82rem', lineHeight: 1.5, margin: 0, marginBottom: code || items ? '.5rem' : 0 }}>{body}</p>}
            {code && (
              <pre style={{ background: 'rgba(5,5,6,0.8)', border: '1px solid #27272a',
                borderRadius: '.4rem', padding: '.6rem .9rem', fontSize: '.75rem',
                color: '#7dd3fc', overflowX: 'auto', margin: 0 }}>{code}</pre>
            )}
            {items && (
              <ul style={{ color: '#a1a1aa', fontSize: '.82rem', lineHeight: 1.6,
                margin: 0, paddingLeft: '1.2rem' }}>
                {items.map((it, i) => <li key={i}>{it}</li>)}
              </ul>
            )}
          </div>
        </div>
      ))}

      <div className="card" style={{ padding: '1rem', marginTop: '1.5rem' }}>
        <div style={{ fontWeight: 700, color: '#f5f5f7', marginBottom: '.4rem' }}>Directory layout after setup</div>
        <pre style={{ color: '#7dd3fc', fontSize: '.75rem', margin: 0, lineHeight: 1.6 }}>{
`<domain>/models/<slug>/
  model.yaml          ← machine-readable manifest (agent entry point)
  README.md           ← HF id, license, upstream links, citation
  recipes/
    inference/README.md
    finetune/README.md  (if applicable)
  examples/
    docker_run.sh
    sbatch_inference_amd.sh
    run_inference.sh
    preflight_<slug>.py`
        }</pre>
      </div>
    </div>
  )
}

// ── Usage Models overview ─────────────────────────────────────────────────────
// The five ways this platform is used, with the key software components for each.
const USAGE_MODELS = [
  {
    title: 'Surrogate Models',
    lead: 'ML surrogate training on modsim and experimental data',
    points: [
      'Key software components: prebuilt ML surrogates',
      '"HuggingFace of ML4Sci models" to support model sharing across teams',
    ],
  },
  {
    title: 'Inference',
    lead: 'vLLM-based containers · multi-node inference of fine-tuned models',
    points: [
      'Key software components: RAG, API and web-server LLM access',
      'Multi-node & elastic serving',
    ],
  },
  {
    title: 'Agentic AI',
    lead: 'AI agents that couple computational simulation & analysis workflows',
    points: [
      'Key software components: Autogen, LangGraph, MCP',
      'Langchain, CrewAI',
    ],
  },
  {
    title: 'Model Training',
    lead: 'Primus · optimized for large-scale pre-training',
    points: [
      'Strong RCCL support and performance',
    ],
  },
  {
    title: 'ModSim',
    lead: 'CCL / MPI requirements',
    points: [
      'Numerical routines across a range of precision datatypes',
      'MCP tool serving of GPU executables',
    ],
  },
]

function UsageModels() {
  return (
    <div style={{ marginBottom: '2rem' }}>
      <div className="section-label" style={{ marginBottom: '.4rem' }}>Usage Models</div>
      <p style={{ color: '#a1a1aa', fontSize: '.82rem', lineHeight: 1.6, marginBottom: '1rem', maxWidth: 820 }}>
        Five ways teams build on this AMD AI-for-science platform — from training fast
        surrogates to standing up agentic workflows that drive simulation and analysis.
      </p>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit,minmax(210px,1fr))', gap: '1rem' }}>
        {USAGE_MODELS.map(u => (
          <div key={u.title} className="card" style={{ padding: '1rem', display: 'flex',
            flexDirection: 'column', gap: '.6rem' }}>
            <div style={{ fontWeight: 800, fontSize: '1rem', color: '#38bdf8',
              borderBottom: '1px solid #38bdf833', paddingBottom: '.4rem' }}>{u.title}</div>
            <div style={{ fontSize: '.82rem', fontWeight: 700, color: '#f5f5f7', lineHeight: 1.4 }}>
              {u.lead}
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '.35rem' }}>
              {u.points.map((p, i) => (
                <div key={i} style={{ display: 'flex', gap: '.4rem', fontSize: '.75rem',
                  color: '#a1a1aa', lineHeight: 1.4 }}>
                  <span style={{ color: '#38bdf8', flexShrink: 0 }}>▸</span>{p}
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

// ── Main catalog component ────────────────────────────────────────────────────
export function ModelCatalog() {
  const { setDomain, setModel, setStep, setView, setTask } = useStore()
  const [allModels, setAllModels] = useState([])
  const [loading, setLoading] = useState(true)
  const [tab, setTab] = useState('all')

  useEffect(() => {
    const domainIds = ['earth_science', 'material_science', 'healthcare', 'physics_simulation', 'protein_folding']
    // Wrap each fetch with a 10-second timeout so a stalled request doesn't block the catalog.
    const withTimeout = (promise, ms = 10000) =>
      Promise.race([promise, new Promise((_, rej) => setTimeout(() => rej(new Error('timeout')), ms))])
    Promise.all(domainIds.map(d => withTimeout(api.domainModels(d)).catch(() => [])))
      .then(results => setAllModels(results.flat().filter(Boolean)))
      .finally(() => setLoading(false))
  }, [])

  function selectModel(m) {
    setDomain(m.domain)
    setModel(m)
    setTask('inference')
    setView('wizard')
    setStep(2)
  }

  const filtered = tab === 'all' || tab === 'add' || tab === 'methods'
    ? allModels
    : allModels.filter(m => m.domain === tab)

  const tabBar = (
    <div style={{ display: 'flex', gap: '.4rem', flexWrap: 'wrap', marginBottom: '1.25rem' }}>
      {DOMAINS.map(d => (
        <button key={d.id}
          onClick={() => setTab(d.id)}
          style={{
            padding: '.3rem .75rem', borderRadius: '9999px', fontSize: '.75rem', fontWeight: 700,
            border: '1px solid', cursor: 'pointer', transition: 'all .12s',
            background: tab === d.id ? '#ED1C24' : 'transparent',
            borderColor: tab === d.id ? '#ED1C24' : '#3f3f46',
            color: tab === d.id ? '#fff' : '#a1a1aa',
          }}>
          {d.icon} {d.label}
          {d.id !== 'all' && d.id !== 'add' && allModels.filter(m => m.domain === d.id).length > 0
            && <span style={{ marginLeft: '.3rem', opacity: .7 }}>
                ({allModels.filter(m => m.domain === d.id).length})
               </span>}
          {d.id === 'all' && allModels.length > 0
            && <span style={{ marginLeft: '.3rem', opacity: .7 }}>({allModels.length})</span>}
        </button>
      ))}
    </div>
  )

  return (
    <div>
      <StepHeader
        title="Model Catalog"
        sub="Browse all AI-for-science models — click any card to launch it in the wizard"
      />

      {tabBar}

      {loading && <Spinner size={28} />}

      {tab === 'methods' ? (
        <MethodsPage />
      ) : tab === 'add' ? (
        <AddModelGuide />
      ) : (
        <>
          {/* Usage-model overview shown on the full catalog (All Models) view */}
          {tab === 'all' && <UsageModels />}
          {!loading && filtered.length === 0 && (
            <div style={{ color: '#52525b', textAlign: 'center', padding: '2rem' }}>
              No models in this domain yet.
            </div>
          )}
          {tab === 'all' && (
            <div className="section-label" style={{ marginBottom: '.75rem' }}>Model Catalog</div>
          )}
          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fill, minmax(260px, 1fr))',
            gap: '1rem',
          }}>
            {filtered.map(m => (
              <ModelCard key={`${m.domain}-${m.slug}`} m={m} onSelect={selectModel} />
            ))}
          </div>
        </>
      )}
    </div>
  )
}
