import React from 'react'
export default function Dashboard({data}){
  if(!data) return <div className='card'>No data. Enter ticker and press Get Output.</div>
  const show = (v)=> (v===null || v===undefined || v==='NA') ? 'NA' : v
  return (
    <div className='card'>
      <h2>{show(data.symbol)} <small className='provider'>{data.provider}</small></h2>
      <div className='grid'>
        <div><strong>Current Price</strong><div>{show(data.current_price)}</div></div>
        <div><strong>Predicted Price</strong><div>{show(data.predicted_price)}</div></div>
        <div><strong>Implied Return %</strong><div>{show(data.implied_return_pct)}</div></div>
        <div><strong>Confidence %</strong><div>{show(data.confidence_pct)}</div></div>
        <div><strong>Momentum %</strong><div>{show(data.momentum_pct)}</div></div>
        <div><strong>Fundamentals Score</strong><div>{show(data.fundamentals_score)}</div></div>
      </div>
      <h3>Recommendation: {show(data.recommendation && data.recommendation.action)}</h3>
      <p><strong>Target Price:</strong> {show(data.recommendation && data.recommendation.target_price)}</p>
      <p><strong>Why:</strong> {show(data.recommendation && data.recommendation.explanation)}</p>
      <details><summary>Raw Response</summary><pre style={{maxHeight:300,overflow:'auto'}}>{JSON.stringify(data,null,2)}</pre></details>
    </div>
  )
}
