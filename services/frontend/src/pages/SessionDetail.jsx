import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { api } from '../api'
import BehaviorBadge from '../components/BehaviorBadge'
import { BEHAVIOR_META } from '../constants'

const TABS = [
  { id: 'attendance', label: 'Điểm danh' },
  { id: 'behavior',   label: 'Hành vi' },
  { id: 'students',   label: 'Từng sinh viên' },
]

function AttendanceTab({ sessionId }) {
  const { data, isLoading } = useQuery({
    queryKey: ['attendance', sessionId],
    queryFn: () => api.getAttendance(sessionId),
  })
  if (isLoading) return <p className="text-slate-400 text-sm">Đang tải...</p>
  if (!data?.length) return <p className="text-slate-500 text-sm py-4">Không có dữ liệu</p>
  return (
    <table className="w-full text-sm">
      <thead>
        <tr className="border-b border-slate-800 text-left">
          <th className="pb-2 text-slate-400 font-medium">Mã SV</th>
          <th className="pb-2 text-slate-400 font-medium">Tên</th>
          <th className="pb-2 text-slate-400 font-medium">Vào lớp</th>
          <th className="pb-2 text-slate-400 font-medium">Ra lớp</th>
          <th className="pb-2 text-slate-400 font-medium">Thời gian</th>
        </tr>
      </thead>
      <tbody className="divide-y divide-slate-800">
        {data.map(r => (
          <tr key={r.student_id}>
            <td className="py-2.5 font-mono text-slate-300">{r.student_id}</td>
            <td className="py-2.5 text-white">{r.name}</td>
            <td className="py-2.5 text-slate-300">
              {r.entry_time ? new Date(r.entry_time * 1000).toLocaleTimeString('vi-VN') : '—'}
            </td>
            <td className="py-2.5 text-slate-300">
              {r.exit_time ? new Date(r.exit_time * 1000).toLocaleTimeString('vi-VN') : (
                <span className="text-green-400 text-xs">Trong lớp</span>
              )}
            </td>
            <td className="py-2.5 text-slate-300">
              {r.duration_min !== undefined ? `${Math.round(r.duration_min)} phút` : '—'}
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  )
}

function BehaviorTab({ sessionId }) {
  const { data: summary, isLoading } = useQuery({
    queryKey: ['summary', sessionId],
    queryFn: () => api.getSummary(sessionId),
  })
  const { data: events = [] } = useQuery({
    queryKey: ['events', sessionId],
    queryFn: () => api.getEvents(sessionId),
  })

  if (isLoading) return <p className="text-slate-400 text-sm">Đang tải...</p>

  const dist = summary?.behavior_distribution ?? {}
  const total = Object.values(dist).reduce((a, b) => a + b, 0)

  return (
    <div className="flex flex-col gap-6">
      {/* distribution */}
      {total > 0 && (
        <div>
          <h4 className="text-sm font-medium text-slate-300 mb-3">Phân bố hành vi</h4>
          <div className="space-y-2">
            {Object.entries(dist).map(([state, count]) => {
              const meta = BEHAVIOR_META[state] ?? { label: state, color: '#94a3b8' }
              const pct = Math.round((count / total) * 100)
              return (
                <div key={state} className="flex items-center gap-3 text-sm">
                  <span className="w-24 text-slate-400 text-xs">{meta.label}</span>
                  <div className="flex-1 h-2.5 bg-slate-800 rounded-full overflow-hidden">
                    <div className="h-full rounded-full transition-all" style={{
                      backgroundColor: meta.color, width: `${pct}%`
                    }} />
                  </div>
                  <span className="text-xs text-slate-300 w-10 text-right">{pct}% ({count})</span>
                </div>
              )
            })}
          </div>
          {summary?.avg_attention_pct !== undefined && (
            <p className="text-sm text-slate-400 mt-3">
              Chú ý trung bình: <span className="text-indigo-400 font-semibold">{summary.avg_attention_pct}%</span>
            </p>
          )}
        </div>
      )}

      {/* event log */}
      {events.length > 0 && (
        <div>
          <h4 className="text-sm font-medium text-slate-300 mb-3">Nhật ký sự kiện ({events.length})</h4>
          <div className="max-h-64 overflow-y-auto space-y-1.5">
            {events.map((ev, i) => {
              const meta = BEHAVIOR_META[ev.state] ?? { color: '#94a3b8', label: ev.state }
              return (
                <div key={i} className="flex gap-3 text-xs">
                  <span className="text-slate-500 font-mono flex-shrink-0">
                    {ev.ts ? new Date(ev.ts * 1000).toLocaleTimeString('vi-VN') : ''}
                  </span>
                  <span className="text-slate-300">
                    <span className="font-medium" style={{ color: meta.color }}>
                      {ev.name ?? `T${ev.track_id}`}
                    </span>
                    {' → '}
                    <span style={{ color: meta.color }}>{meta.label}</span>
                    {ev.duration_s && <span className="text-slate-500 ml-1">({ev.duration_s}s)</span>}
                  </span>
                </div>
              )
            })}
          </div>
        </div>
      )}
    </div>
  )
}

function StudentsTab({ sessionId }) {
  const { data: events = [] } = useQuery({
    queryKey: ['events', sessionId],
    queryFn: () => api.getEvents(sessionId),
  })

  const byStudent = events.reduce((acc, ev) => {
    const key = ev.student_id ?? ev.track_id
    if (!acc[key]) acc[key] = { name: ev.name ?? `Track ${key}`, events: [] }
    acc[key].events.push(ev)
    return acc
  }, {})

  const entries = Object.entries(byStudent)
  if (!entries.length) return <p className="text-slate-500 text-sm py-4">Không có dữ liệu</p>

  return (
    <div className="space-y-3">
      {entries.map(([id, { name, events: evs }]) => {
        const stateMap = evs.reduce((a, e) => { a[e.state] = (a[e.state] ?? 0) + 1; return a }, {})
        const focusedCount = stateMap['focused'] ?? 0
        const focusPct = Math.round((focusedCount / evs.length) * 100)
        return (
          <div key={id} className="bg-slate-800/50 rounded-lg p-3">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm font-medium text-white">{name}</span>
              <span className="text-xs text-indigo-400">{focusPct}% chú ý</span>
            </div>
            <div className="flex flex-wrap gap-1">
              {Object.entries(stateMap).map(([state, count]) => {
                const meta = BEHAVIOR_META[state] ?? { label: state, color: '#94a3b8' }
                return (
                  <span key={state} className="text-xs px-2 py-0.5 rounded-full"
                    style={{ backgroundColor: meta.color + '22', color: meta.color }}>
                    {meta.label}: {count}
                  </span>
                )
              })}
            </div>
          </div>
        )
      })}
    </div>
  )
}

export default function SessionDetail({ sessionId, onBack }) {
  const [tab, setTab] = useState('attendance')
  const { data: session, isLoading } = useQuery({
    queryKey: ['session', sessionId],
    queryFn: () => api.getSession(sessionId),
    enabled: !!sessionId,
  })

  return (
    <div className="card flex flex-col gap-4">
      {/* back + header */}
      <div className="flex items-center gap-3">
        <button onClick={onBack} className="btn-secondary text-sm px-3 py-1.5">
          ← Quay lại
        </button>
        <div>
          <h2 className="sec-title mb-0">
            {isLoading ? 'Đang tải...' : (session?.class_name ?? `Phiên #${sessionId}`)}
          </h2>
          {session?.start_time && (
            <p className="text-xs text-slate-500 mt-0.5">
              {new Date(session.start_time).toLocaleString('vi-VN')}
            </p>
          )}
        </div>
      </div>

      {/* tabs */}
      <div className="flex gap-1 border-b border-slate-800 pb-0">
        {TABS.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)}
            className={`tab-btn ${tab === t.id ? 'active' : ''}`}>
            {t.label}
          </button>
        ))}
      </div>

      {/* content */}
      <div className="min-h-48">
        {tab === 'attendance' && <AttendanceTab sessionId={sessionId} />}
        {tab === 'behavior'   && <BehaviorTab   sessionId={sessionId} />}
        {tab === 'students'   && <StudentsTab   sessionId={sessionId} />}
      </div>
    </div>
  )
}
