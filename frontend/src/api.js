const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'
export async function predict(symbol, interval, alphaKey){
  const res = await fetch(`${API_URL}/model/predict`,{
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({symbol, interval, alpha_key: alphaKey})
  })
  if(!res.ok){ const txt = await res.text(); throw new Error(txt) }
  return res.json()
}
