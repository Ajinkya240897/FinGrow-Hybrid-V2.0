const API_URL = import.meta.env.VITE_API_URL || 'https://fingrow-hybrid-v2-0.onrender.com'

export async function predict(symbol, interval, indianApiKey){
  const payload = {
    symbol: String(symbol || '').trim().toUpperCase(),
    interval: String(interval || '3-15d').trim()
  }
  if (indianApiKey && String(indianApiKey).trim() !== '') {
    payload.indianapi_key = String(indianApiKey).trim()
  }
  const res = await fetch(`${API_URL}/model/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload)
  })
  const text = await res.text()
  try {
    const json = JSON.parse(text)
    if (!res.ok) throw new Error(JSON.stringify(json))
    return json
  } catch (err) {
    if (res.ok) return { raw: text }
    throw new Error(text || err.message)
  }
}
