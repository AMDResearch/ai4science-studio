import { create } from 'zustand'

export const useStore = create((set, get) => ({
  // Navigation
  step: 0,
  setStep: (s) => set({ step: s }),

  // Demo/Live mode
  mode: 'demo',
  setMode: (m) => set({ mode: m }),

  // Domain selection
  domain: null,
  setDomain: (d) => set({ domain: d, model: null, prompt: '', customPrompt: '', runId: null, result: null }),

  // Model selection
  model: null,
  setModel: (m) => set({ model: m, prompt: '', customPrompt: '', runId: null, result: null }),

  // Task (inference vs train-scaling) — drives which demo/live path runs
  task: 'inference',
  setTask: (t) => set({ task: t }),

  // Trained-model variant for live inference (1gpu | 8gpu)
  modelVariant: '8gpu',
  setModelVariant: (v) => set({ modelVariant: v }),

  // Prompt
  prompt: '',
  setPrompt: (p) => set({ prompt: p }),
  customPrompt: '',
  setCustomPrompt: (p) => set({ customPrompt: p }),
  promptError: '',
  setPromptError: (e) => set({ promptError: e }),

  // Params (from model env_vars)
  params: {},
  setParams: (p) => set({ params: p }),
  setParam: (k, v) => set((s) => ({ params: { ...s.params, [k]: v } })),

  // SLURM
  partition: 'lux',
  setPartition: (p) => set({ partition: p }),

  // Run
  runId: null,
  setRunId: (id) => set({ runId: id }),
  runState: 'idle',
  setRunState: (s) => set({ runState: s }),

  // Results
  result: null,
  setResult: (r) => set({ result: r }),
  outputFiles: [],
  setOutputFiles: (f) => set({ outputFiles: f }),

  // App view — 'wizard' (default) or 'catalog' (model-gallery tab)
  view: 'wizard',
  setView: (v) => set({ view: v }),

  // Add-model modal
  addModelOpen: false,
  setAddModelOpen: (v) => set({ addModelOpen: v }),

  // Reset a run
  resetRun: () => set({ runId: null, runState: 'idle', result: null, outputFiles: [] }),
}))
