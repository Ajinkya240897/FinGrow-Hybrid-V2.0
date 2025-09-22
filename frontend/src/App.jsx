import React, {useState} from 'react'
import FetchForm from './components/FetchForm'
import Dashboard from './components/Dashboard'
export default function App(){
  const [result,setResult] = useState(null)
  return (
    <div className='app'>
      <header className='topbar'><h1>Fingrow-Hybrid v2.0</h1><p className='tag'>Real-time quotes • Predictions • Beginner-friendly recommendations</p></header>
      <main className='content'><FetchForm onResult={setResult} /><Dashboard data={result} /></main>
    </div>
  )
}
