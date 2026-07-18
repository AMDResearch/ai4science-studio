import React from 'react'
import ReactDOM from 'react-dom/client'
import './theme.css'
import App from './App'

const root = ReactDOM.createRoot(document.getElementById('root'))
root.render(<React.StrictMode><App /></React.StrictMode>)

// Expose store for Playwright automation
import { useStore } from './store'
window.__studioStore = useStore
