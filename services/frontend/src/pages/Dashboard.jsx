import { useState, useCallback, useRef } from 'react'
import { useQuery } from '@tanstack/react-query'
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip } from 'recharts'
import VideoStream from '../components/VideoStream'
import BehaviorBadge from '../components/BehaviorBadge'
import StatCard from '../components/StatCard'
import { useWebSocket } from '../hooks/useWebSocket'
import { BEHAVIOR_META, GAZE_ARROW, OBJECT_EMOJI, WS_EVENTS } from '../constants'
import { api } from '../api'

const MAX_EVENTS = 50

export default function Dashboard() {
  const [tracks, setTracks]   = useState([])
  const [frameW, setFrameW]   = useState(1280)
  const [frameH, setFrameH]   = useState(720)
  const [events, setEvents]   = useState([])
  const [session, setSession] = useState(null)
  const [wsStreamStatus, setWsStreamStatus] = useState('disconnected')

  // behavior event websocket
  const onEvent = useCallback((msg) => {
    if (msg.type === 'frame') {
      setTracks(msg.tracks ?? [])
      if (msg.frame_w) setFrameW(msg.frame_w)
      if (msg.frame_h) setFrameH(msg.frame_h)
      if (msg.session) setSession(msg.session)
    } else if (msg.type === 'behavior_event') {
      setEvents(prev => [msg, ...prev].slice(0, MAX_EVENTS))
    }
  }, [])

  const { status: evStatus } = useWebSocket(WS_EVENTS, { onMessage: onEvent })

  // fetch active session if not available via WS
  const { data: sessions = [] } = useQuery({
    queryKey: ['sessions'],
    queryFn: () => api.getSessions(),
    refetchInterval: 5000,
  })
  
  const activeSession = session || sessions.find(s => !s.end_time)

  // derived stats
  const focused    = tracks.filter(t => t.state === 'focused').length
  const total      = tracks.length
  const attPct     = total > 0 ? Math.round((focused / total) * 100) : 0
  const stateCounts = tracks.reduce((acc, t) => {
    acc[t.state] = (acc[t.state] ?? 0) + 1; return acc
  }, {})

  return (
    <div className="flex flex-col gap-4 h-full overflow-hidden">
      {/* stat bar */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3 flex-shrink-0">
        <StatCard label="Đang theo dõi"   value={total}     icon="👥" />
        <StatCard label="Chú ý"           value={`${attPct}%`} icon="🎯" color="text-indigo-400" />
        <StatCard label="Mất tập trung"   value={total - focused} icon="⚠️" color="text-amber-400" />
        <StatCard label="Trạng thái WS"   value={evStatus}  icon="🔗"
          color={evStatus === 'connected' ? 'text-green-400' : 'text-slate-400'} />
      </div>

      {/* main split: video left, side panel right */}
      <div className="flex gap-4 flex-1 min-h-0">
        {/* video */}
        <div className="flex-1 min-w-0 flex flex-col gap-4">
          <VideoStream tracks={tracks} frameW={frameW} frameH={frameH} wsStatus={evStatus} session={activeSession} />

          {/* attention bar */}
          <div className="card">
            <p className="text-xs text-slate-400 mb-2">Chú ý toàn lớp</p>
            <div className="progress-bar">
              <div className="progress-fill" style={{ width: `${attPct}%` }} />
            </div>
            <p className="text-right text-xs text-slate-400 mt-1">{attPct}%</p>
          </div>
        </div>

        {/* right panel */}
        <div className="w-80 flex-shrink-0 flex flex-col gap-4 overflow-y-auto">
          {/* student tracks */}
          <div className="card flex flex-col gap-2">
            <h3 className="sec-title">Sinh viên ({total})</h3>
            {total === 0 ? (
              <p className="text-sm text-slate-500 py-4 text-center">Chưa phát hiện ai</p>
            ) : (
              <div className="space-y-2 max-h-72 overflow-y-auto pr-1">
                {tracks.map(t => {
                  const meta = BEHAVIOR_META[t.state] ?? { color: '#94a3b8', label: t.state }
                  const objs = t.objects ?? []
                  return (
                    <div key={t.track_id}
                      className="flex items-start gap-2 bg-slate-800/50 rounded-lg p-2 text-sm">
                      <div className="w-8 h-8 rounded-full flex items-center justify-center text-xs font-bold flex-shrink-0"
                        style={{ backgroundColor: meta.color + '33', color: meta.color }}>
                        {t.name ? t.name[0] : `T${t.track_id}`}
                      </div>
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center justify-between gap-1">
                          <span className="font-medium text-white truncate">
                            {t.name ?? `Track ${t.track_id}`}
                          </span>
                          <BehaviorBadge state={t.state} />
                        </div>
                        <div className="flex items-center gap-2 mt-0.5 text-xs text-slate-400">
                          {t.gaze_direction && (
                            <span title="Hướng nhìn">{GAZE_ARROW[t.gaze_direction] ?? '?'} {t.gaze_direction}</span>
                          )}
                          {t.focused_fraction !== undefined && (
                            <span>{Math.round(t.focused_fraction * 100)}% focus</span>
                          )}
                        </div>
                        {objs.length > 0 && (
                          <div className="flex gap-1 mt-1">
                            {objs.map(o => (
                              <span key={o} title={o} className="text-base leading-none">
                                {OBJECT_EMOJI[o] ?? '📦'}
                              </span>
                            ))}
                          </div>
                        )}
                      </div>
                    </div>
                  )
                })}
              </div>
            )}
          </div>

          {/* behavior distribution chart */}
          {total > 0 && (
            <div className="card h-48 flex-shrink-0 flex flex-col">
              <h3 className="sec-title mb-0">Phân bố hành vi</h3>
              <div className="flex-1 min-h-0 relative">
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie
                      data={Object.entries(stateCounts).map(([k, v]) => ({ name: BEHAVIOR_META[k]?.label || k, value: v, fill: BEHAVIOR_META[k]?.color || '#94a3b8' }))}
                      cx="50%" cy="50%" innerRadius={40} outerRadius={60}
                      paddingAngle={5} dataKey="value" stroke="none"
                    >
                      {Object.entries(stateCounts).map(([k], index) => (
                        <Cell key={`cell-${index}`} fill={BEHAVIOR_META[k]?.color || '#94a3b8'} />
                      ))}
                    </Pie>
                    <Tooltip contentStyle={{ backgroundColor: '#0f172a', borderColor: '#1e293b', fontSize: '12px', borderRadius: '8px' }} itemStyle={{ color: '#fff' }} />
                  </PieChart>
                </ResponsiveContainer>
              </div>
            </div>
          )}

          {/* live events */}
          <div className="card flex flex-col gap-2 flex-1">
            <div className="flex items-center justify-between">
              <h3 className="sec-title">Sự kiện live</h3>
              {events.length > 0 && (
                <button onClick={() => setEvents([])} className="text-xs text-slate-500 hover:text-slate-300">
                  Xóa
                </button>
              )}
            </div>
            <div className="space-y-1.5 max-h-64 overflow-y-auto pr-1">
              {events.length === 0 ? (
                <p className="text-sm text-slate-500 py-2 text-center">Chưa có sự kiện</p>
              ) : events.map((ev, i) => {
                const meta = BEHAVIOR_META[ev.state] ?? { color: '#94a3b8', label: ev.state }
                return (
                  <div key={i} className="flex gap-2 text-xs">
                    <span className="text-slate-500 flex-shrink-0 font-mono">
                      {ev.ts ? new Date(ev.ts * 1000).toLocaleTimeString() : '--:--:--'}
                    </span>
                    <span className="text-slate-300 flex-1">
                      <span className="font-medium" style={{ color: meta.color }}>
                        {ev.name ?? `T${ev.track_id}`}
                      </span>
                      {' → '}
                      <span style={{ color: meta.color }}>{meta.label}</span>
                    </span>
                  </div>
                )
              })}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
