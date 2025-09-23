import React from 'react'

function Row({k, v}) {
  return (
    <div className="row">
      <div className="label">{k}</div>
      <div className="value">{v === null || v === undefined ? 'NA' : String(v)}</div>
    </div>
  )
}

export default function Result({data}) {
  if (!data) return <div className="card"><div className="empty">No result yet</div></div>
  if (data.raw && !data.current_price) {
    return <div className="card"><pre>{String(data.raw).slice(0,1000)}</pre></div>
  }
  return (
    <div className="card result">
      <h3>Result</h3>
      <Row k="Symbol" v={data.resolved_symbol || data.symbol} />
      <Row k="Provider" v={data.provider} />
      <Row k="Current Price" v={data.current_price} />
      <Row k="Predicted Price" v={data.predicted_price} />
      <Row k="Implied Return (%)" v={data.implied_return_pct} />
      <Row k="Confidence (%)" v={data.confidence_pct} />
      <Row k="Momentum (%)" v={data.momentum_pct} />
      <Row k="Fundamentals Score" v={data.fundamentals_score} />
      <div style={{marginTop:10}}>
        <strong>Recommendation</strong>
        <div>Action: {data.recommendation?.action || 'NA'}</div>
        <div>Target: {data.recommendation?.target_price || 'NA'}</div>
        <div>{data.recommendation?.explanation || ''}</div>
      </div>
    </div>
  )
}
