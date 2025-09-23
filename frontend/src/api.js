// src/api.js
// Sends a single well-formed JSON object to the backend predict endpoint.
// Reads backend URL from Vite env variable VITE_API_URL

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

export async function predict(symbol, interval, indianApiKey){
  // Build a single clean JSON object
  const payload = {
    symbol: String(symbol || '').trim().toUpperCase(),
    interval: String(interval || '3-15d').trim()
  }
  if (indianApiKey && String(indianApiKey).trim() !== '') {
    payload.indianapi_key = String(indianApiKey).trim()
  }

  // single JSON.stringify call - prevents "extra data" errors
  const body = JSON.stringify(payload)

  const res = await fetch(`${API_URL}/model/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body
  })

  // Read text first, then parse to handle non-JSON responses gracefully
  const text = await res.text()
  try {
    const json = JSON.parse(text)
    if (!res.ok) {
      // Return structured error
      const err = new Error('Server error')
      err.detail = json
      throw err
    }
    return json
  } catch (err) {
    // If parsing fails but status ok, wrap raw text
    if (res.ok) {
      return { raw: text }
    }
    // otherwise throw the error for UI to show
    const e = new Error(text || (err && err.message) || 'Request failed')
    e.raw = text
    throw e
  }
}
