# lightweight main to expose health and simple fetch
from fastapi import FastAPI
from providers import fetch_quote
app = FastAPI()
@app.get('/health')
async def health():
    return {'status':'ok'}
@app.post('/fetch')
async def fetch(q: dict):
    s = q.get('symbol')
    k = q.get('alpha_key')
    return fetch_quote(s, alpha_key=k)
