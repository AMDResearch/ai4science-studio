import { useStore } from './store'
import { AppShell } from './components/AppShell'
import { AddModelModal } from './components/AddModelModal'
import { StepDomain } from './steps/0_Domain'
import { StepModel } from './steps/1_Model'
import { StepConfigure } from './steps/2_Configure'
import { StepRun } from './steps/3_Run'
import { StepAnalyze } from './steps/4_Analyze'

const STEPS = [StepDomain, StepModel, StepConfigure, StepRun, StepAnalyze]

export default function App() {
  const { step, addModelOpen } = useStore()
  const StepComponent = STEPS[Math.min(step, STEPS.length - 1)]
  return (
    <AppShell>
      <StepComponent />
      {addModelOpen && <AddModelModal />}
    </AppShell>
  )
}
