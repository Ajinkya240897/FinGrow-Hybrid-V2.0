// src/components/FetchForm.jsx
import React, {useState, useRef} from 'react'
import {predict} from '../api'

export default function FetchForm({onResult}){
  const [symbol,setSymbol] = useState('')
  const [interval,setInterval] = useState('3-15d')
  const [indianKey,setIndianKey] = useState('')
  const [loading,setLoading] = useState(false)
  const [error,setError] = useState(null)
  const abortRef = useRef(null) // to cancel if needed

  const submit = async (e) => {
    e.preventDefault()
    setError(null)

    const s = String(symbol || '').trim()
    if (!s) {
      setError('Please enter a ticker symbol (e.g., RELIANCE)')
      return
    }

    // prevent double submit
    if (loading) return
    setLoading(true)
    // abort previous request if any
    if (abortRef.current) {
      try { abortRef.current.abort() } catch {}
    }
    abortRef.current = new AbortController()

    try {
      const res = await predict(s, interval, indianKey)
      onResult(res)
    } catch (err) {
      console.error('Fetch error', err)
      if (err && err.detail) {
        setError(JSON.stringify(err.detail))
      } else if (err && err.raw) {
        setError(String(err.raw).slice(0, 400))
      } else {
        setError(String(err.message || err))
      }
      onResult(null)
    } finally {
      setLoading(false)
      abortRef.current = null
    }
  }

  return (
    <div className="card">
      <form onSubmit={submit} className="form" autoComplete="off">
        <label className="label">
          Ticker (no suffix)
          <input
            value={symbol}
            onChange={e=>setSymbol(e.target.value)}
            placeholder="e.g. RELIANCE or TCS"
            aria-label="ticker"
          />
        </label>

        <label className="label">
          Interval
          <select value={interval} onChange={e=>setInterval(e.target.value)} aria-label="interval">
            <option value='3-15d'>3-15 days</option>
            <option value='1-3m'>1-3 months</option>
            <option value='3-6m'>3-6 months</option>
            <option value='1-3y'>1-3 years</option>
          </select>
        </label>

        <label className="label">
          IndianAPI Key (optional)
          <input
            value={indianKey}
            onChange={e=>setIndianKey(e.target.value)}
            placeholder="Optional: per-request IndianAPI key"
            aria-label="indianapi-key"
          />
        </label>

        <div className="actions">
          <button type="submit" disabled={loading}>{loading ? 'Working...' : 'Get Output'}</button>
        </div>
      </form>

      {error && <div className="error" role="alert">{error}</div>}
      <div className="hint">Tip: if you leave IndianAPI Key empty, the server env var will be used (INDIANAPI_KEY).</div>
    </div>
  )
}
