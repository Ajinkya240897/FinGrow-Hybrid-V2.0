import React, {useState} from 'react'
import {predict} from '../api'
export default function FetchForm({onResult}){
  const [symbol,setSymbol]=useState('AAPL')
  const [alphaKey,setAlphaKey]=useState('')
  const [interval,setInterval]=useState('3-15d')
  const [loading,setLoading]=useState(false)
  const [error,setError]=useState(null)
  const submit=async(e)=>{e.preventDefault(); setError(null); setLoading(true); try{ const res = await predict(symbol, interval, alphaKey); onResult(res);}catch(err){ setError(err.message); onResult(null);}finally{ setLoading(false)}}
  return (
    <div className='card'>
      <form onSubmit={submit} className='form'>
        <input value={symbol} onChange={e=>setSymbol(e.target.value)} placeholder='Ticker (no suffix)' />
        <select value={interval} onChange={e=>setInterval(e.target.value)}>
          <option value='3-15d'>3-15 days</option>
          <option value='1-3m'>1-3 months</option>
          <option value='3-6m'>3-6 months</option>
          <option value='1-3y'>1-3 years</option>
        </select>
        <input value={alphaKey} onChange={e=>setAlphaKey(e.target.value)} placeholder='Alpha Vantage API Key (optional)' />
        <div className='actions'>
          <button type='submit' disabled={loading}>{loading?'Working...':'Get Output'}</button>
        </div>
      </form>
      {error && <div className='error'>{error}</div>}
    </div>
  )
}
