// src/components/Result.jsx
import React from 'react'

function ShowItem({label, value}) {
  const val = value === null || value === undefined ? 'NA' : value
  return (
    <div className="row">
      <div className="label">{label}</div>
      <div className="value">{typeof val === 'number' ? val : String(val)}</div>
    </div>
  )
}

export default function Result({data}) {
  if (!data) {
    return (
      <div className="card result">
        <div className="empty">No result yet — enter a ticker and click Get Output.</div>
      </div>
    )
  }

  // If backend returned raw text wrap
  if (data.raw && !data.current_price) {
    return (
      <div className="card result">
        <div className="error"><strong>Raw response:</strong> <pre>{String(data.raw).slice(0,1000)}</pre></div>
      </div>
    )
  }

  return (
    <div className="card result">
      <h3>Result</h3>
      <ShowItem label="Resolved Symbol" value={data.resolved_symbol || data.symbol || 'NA'} />
      <ShowItem label="Provider" value={data.provider || 'NA'} />
      <ShowItem label="Current Price" value={data.current_price ?? 'NA'} />
      <ShowItem label="Predicted Price" value={data.predicted_price ?? 'NA'} />
      <ShowItem label="Implied Return (%)" value={data.implied_return_pct ?? 'NA'} />
      <ShowItem label="Confidence (%)" value={data.confidence_pct ?? 'NA'} />
      <ShowItem label="Momentum (%)" value={data.momentum_pct ?? 'NA'} />
      <ShowItem label="Fundamentals Score" value={data.fundamentals_score ?? 'NA'} />

      <div className="recommendation">
        <h4>Recommendation</h4>
        <div><strong>Action:</strong> { (data.recommendation && data.recommendation.action) || 'NA' }</div>
        <div><strong>Target:</strong> { (data.recommendation && data.recommendation.target_price) || 'NA' }</div>
        <div className="explain">{ (data.recommendation && data.recommendation.explanation) || 'NA' }</div>
      </div>
    </div>
  )
}
