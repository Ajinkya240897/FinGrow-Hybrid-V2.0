const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

export async function predict(symbol, interval, indianApiKey){
  const body = { symbol, interval, indianapi_key: indianApiKey || null }
  const res = await fetch(`${API_URL}/model/predict`, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(body)
  })
  if(!res.ok){
    const t = await res.text()
    throw new Error(t)
  }
  return res.json()
}
