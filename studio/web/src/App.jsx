import { Routes, Route, NavLink } from 'react-router-dom'
import Home from './pages/Home.jsx'
import Catalog from './pages/Catalog.jsx'
import ModelDetail from './pages/ModelDetail.jsx'
import ClusterSetup from './pages/ClusterSetup.jsx'
import Run from './pages/Run.jsx'
import Jobs from './pages/Jobs.jsx'
import AddModel from './pages/AddModel.jsx'

const NAV = [
  { to: '/', label: 'Home', end: true },
  { to: '/catalog', label: 'Catalog' },
  { to: '/cluster', label: 'Cluster setup' },
  { to: '/run', label: 'Run' },
  { to: '/jobs', label: 'Jobs' },
  { to: '/add-model', label: 'Add model' },
]

export default function App() {
  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">AI4Science Studio</div>
        <nav className="nav">
          {NAV.map((n) => (
            <NavLink key={n.to} to={n.to} end={n.end}
              className={({ isActive }) => (isActive ? 'nav-link active' : 'nav-link')}>
              {n.label}
            </NavLink>
          ))}
        </nav>
      </header>
      <main className="content">
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/catalog" element={<Catalog />} />
          <Route path="/models/:slug" element={<ModelDetail />} />
          <Route path="/cluster" element={<ClusterSetup />} />
          <Route path="/run" element={<Run />} />
          <Route path="/jobs" element={<Jobs />} />
          <Route path="/add-model" element={<AddModel />} />
        </Routes>
      </main>
    </div>
  )
}
