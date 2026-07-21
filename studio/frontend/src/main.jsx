import React from 'react'
import ReactDOM from 'react-dom/client'
import './theme.css'
import App from './App'

const root = ReactDOM.createRoot(document.getElementById('root'))
root.render(<React.StrictMode><App /></React.StrictMode>)

// window.__studioStore is exposed from store.js itself (a single module instance
// shared with the whole component tree) so the Playwright demo recorder drives
// the exact store the UI renders from.
