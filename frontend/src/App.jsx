import React, { useState } from 'react'
import FetchForm from './components/FetchForm'
import Result from './components/Result'
import './styles.css'

export default function App() {
  const [result, setResult] = useState(null)

  return (
    <div className="container">
      <header>
        <h1>Fingrow Hybrid</h1>
        <p className="subtitle">Stock predictions — IndianAPI + yfinance</p>
      </header>

      <main className="main">
        <FetchForm onResult={setResult} />
        <Result data={result} />
      </main>
    </div>
  )
}
