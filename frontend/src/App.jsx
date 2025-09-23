import React, {useState} from 'react'
import FetchForm from './components/FetchForm'
import Result from './components/Result'

export default function App(){
  const [result, setResult] = useState(null)

  return (
    <div className="container">
      <header className="header">
        <h1>Fingrow Hybrid</h1>
        <p className="subtitle">Enter a ticker (no suffix) and get prediction & signals.</p>
      </header>

      <main className="main">
        <FetchForm onResult={setResult} />
        <Result data={result} />
      </main>

      <footer className="footer">
        <small>Powered by IndianAPI (primary) & yfinance (fallback)</small>
      </footer>
    </div>
  )
}
