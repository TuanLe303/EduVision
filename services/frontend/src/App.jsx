import { useState } from 'react'
import Sidebar from './components/Sidebar'
import Dashboard from './pages/Dashboard'
import Sessions from './pages/Sessions'
import SessionDetail from './pages/SessionDetail'
import Reports from './pages/Reports'
import Students from './pages/Students'
import Settings from './pages/Settings'

export default function App() {
  const [page, setPage] = useState('dashboard')
  const [selectedSession, setSelectedSession] = useState(null)

  const handleSelectSession = (id) => {
    setSelectedSession(id)
    setPage('session-detail')
  }

  const handleBackToSessions = () => {
    setSelectedSession(null)
    setPage('sessions')
  }

  const renderPage = () => {
    switch (page) {
      case 'dashboard':     return <Dashboard />
      case 'sessions':      return <Sessions onSelectSession={handleSelectSession} />
      case 'session-detail': return <SessionDetail sessionId={selectedSession} onBack={handleBackToSessions} />
      case 'reports':       return <Reports />
      case 'students':      return <Students />
      case 'settings':      return <Settings />
      default:              return <Dashboard />
    }
  }

  return (
    <div className="flex h-screen bg-slate-950 text-slate-100 overflow-hidden">
      <Sidebar page={page === 'session-detail' ? 'sessions' : page} setPage={setPage} />
      <main className="flex-1 overflow-y-auto p-5">
        {renderPage()}
      </main>
    </div>
  )
}
