import { create } from 'zustand'

// Read-only demo deployment: when served from the demo hostname (or with the
// VITE_DEMO_LOCK build flag), the app is locked to DEMO mode — the Live toggle is
// grayed out and mode can never be switched to 'live'. One frontend build serves
// both the full site and the demo site; the hostname decides behavior at runtime.
export const DEMO_LOCK = (() => {
  try {
    if (import.meta.env?.VITE_DEMO_LOCK === '1') return true
    const h = (typeof window !== 'undefined' && window.location?.hostname) || ''
    return /(^|\.)ai4science-studio-demo\./i.test(h) || h.includes('-studio-demo')
  } catch { return false }
})()

export const useStore = create((set, get) => ({
  // Navigation
  step: 0,
  setStep: (s) => set({ step: s }),

  // Demo/Live mode. Demo-locked deployments can never leave 'demo'.
  mode: 'demo',
  demoLock: DEMO_LOCK,
  setMode: (m) => set({ mode: DEMO_LOCK ? 'demo' : m }),

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

  // Prompt — changing the case/prompt invalidates any prior run so the Analyze
  // page can't show stale results for a different case (demo-critical).
  prompt: '',
  setPrompt: (p) => set({ prompt: p, runId: null, runState: 'idle', result: null, outputFiles: [] }),
  customPrompt: '',
  setCustomPrompt: (p) => set({ customPrompt: p }),
  promptError: '',
  setPromptError: (e) => set({ promptError: e }),

  // Params (from model env_vars). Changing a param (e.g. dc_event) also
  // invalidates prior run results.
  params: {},
  setParams: (p) => set({ params: p }),
  setParam: (k, v) => set((s) => ({ params: { ...s.params, [k]: v },
    runId: null, runState: 'idle', result: null, outputFiles: [] })),

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

// Expose the SAME store instance the component tree uses, for Playwright demo
// automation (window.__studioStore). This must live here in store.js — not in a
// separate import in main.jsx — so it can never bind to a second module copy.
// (A query-tagged entry point, e.g. main.jsx?v=..., can cause the dev server to
// evaluate store.js twice; exposing from within this module guarantees the demo
// recorder drives the exact store the UI renders from.)
if (typeof window !== 'undefined') window.__studioStore = useStore
