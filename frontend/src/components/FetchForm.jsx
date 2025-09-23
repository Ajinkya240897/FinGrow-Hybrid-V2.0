// frontend/src/components/FetchForm.jsx
import React, {useState} from 'react'
import {predict} from '../api'

export default function FetchForm({onResult}){
  const [symbol,setSymbol]=useState('')
  const [interval,setInterval]=useState('3-15d')
  const [indianKey,setIndianKey]=useState('')
  const [loading,setLoading]=useState(false)
  const [error,setError]=useState(null)

  const submit = async (e) => {
    e.preventDefault()
    setError(null)
    if (!symbol || symbol.trim()==='') {
      setError('Enter a ticker (e.g., RELIANCE)')
      return
    }
    if (loading) return
    setLoading(true)
    try {
      const res = await predict(symbol, interval, indianKey)
      onResult(res)
    } catch (err) {
      setError(String(err.message || err))
      onResult(null)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="card">
      <form onSubmit={submit} className="form" autoComplete="off">
        <label>Ticker
          <input value={symbol} onChange={e=>setSymbol(e.target.value)} placeholder="e.g. RELIANCE" />
        </label>
        <label>Interval
          <select value={interval} onChange={e=>setInterval(e.target.value)}>
            <option value='3-15d'>3-15 days</option>
            <option value='1-3m'>1-3 months</option>
            <option value='3-6m'>3-6 months</option>
            <option value='1-3y'>1-3 years</option>
          </select>
        </label>
        <label>IndianAPI Key (optional)
          <input value={indianKey} onChange={e=>setIndianKey(e.target.value)} placeholder="Optional per-request key" />
        </label>
        <div className="actions">
          <button type="submit" disabled={loading}>{loading ? 'Working...' : 'Get Output'}</button>
        </div>
        {error && <div className="error">{error}</div>}
      </form>
    </div>
  )
}
