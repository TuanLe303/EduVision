import { useEffect, useRef, useState } from 'react'
import { useCamera } from '../hooks/useCamera'
import { BEHAVIOR_META, GAZE_ARROW, OBJECT_EMOJI } from '../constants'

function drawTracks(canvas, video, tracks) {
  const ctx = canvas.getContext('2d')
  const W = canvas.width  = video.videoWidth  || canvas.offsetWidth
  const H = canvas.height = video.videoHeight || canvas.offsetHeight
  ctx.clearRect(0, 0, W, H)

  tracks.forEach(t => {
    const [x1, y1, x2, y2] = t.bbox
    const meta = BEHAVIOR_META[t.state] ?? { color: '#94a3b8', label: t.state }
    const bw = x2 - x1
    const bh = y2 - y1

    ctx.strokeStyle = meta.color
    ctx.lineWidth = 2
    ctx.strokeRect(x1, y1, bw, bh)

    // header pill
    const label = t.name ? `${t.name} (${t.track_id})` : `T${t.track_id}`
    ctx.fillStyle = meta.color
    ctx.fillRect(x1, y1 - 20, ctx.measureText(label).width + 10, 20)
    ctx.fillStyle = '#000'
    ctx.font = '12px Inter, sans-serif'
    ctx.fillText(label, x1 + 5, y1 - 5)
  })
}

export default function VideoStream({ tracks = [], frameW, frameH, wsStatus }) {
  const videoRef = useRef(null)
  const canvasRef = useRef(null)
  const { status: camStatus, error: camError, start, stop } = useCamera()
  const [useWs, setUseWs] = useState(false)

  // draw bbox overlay on every tracks change
  useEffect(() => {
    const video  = videoRef.current
    const canvas = canvasRef.current
    if (!video || !canvas || tracks.length === 0) return
    drawTracks(canvas, video, tracks)
  }, [tracks])

  const handleToggle = async () => {
    const video = videoRef.current
    if (camStatus === 'on') { stop(video) }
    else                    { await start(video) }
  }

  const camActive = camStatus === 'on'

  return (
    <div className="card flex flex-col gap-3">
      {/* header bar */}
      <div className="flex items-center justify-between">
        <h2 className="sec-title mb-0">Live Monitor</h2>
        <div className="flex items-center gap-2">
          {wsStatus && (
            <span className={`text-xs px-2 py-0.5 rounded-full ${
              wsStatus === 'connected' ? 'bg-green-900 text-green-300' :
              wsStatus === 'connecting' ? 'bg-yellow-900 text-yellow-300' :
              'bg-slate-700 text-slate-400'
            }`}>
              WS {wsStatus}
            </span>
          )}
          <button
            onClick={handleToggle}
            className={camActive ? 'btn-danger' : 'btn-primary'}
          >
            {camStatus === 'requesting' ? 'Đang kết nối…' : camActive ? 'Dừng camera' : 'Bật camera'}
          </button>
        </div>
      </div>

      {/* video + canvas stack */}
      <div className="relative bg-slate-900 rounded-lg overflow-hidden" style={{ aspectRatio: '16/9' }}>
        <video
          ref={videoRef}
          autoPlay
          muted
          playsInline
          className="w-full h-full object-cover"
          style={{ display: camActive ? 'block' : 'none' }}
        />
        {!camActive && (
          <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 text-slate-500">
            <svg className="w-16 h-16" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1}
                d="M15 10l4.553-2.069A1 1 0 0121 8.882v6.236a1 1 0 01-1.447.894L15 14M3 8a2 2 0 012-2h10a2 2 0 012 2v8a2 2 0 01-2 2H5a2 2 0 01-2-2V8z" />
            </svg>
            <span className="text-sm">Camera chưa bật</span>
            {camError && <span className="text-xs text-red-400">{camError}</span>}
          </div>
        )}
        {/* transparent canvas overlay for bbox drawing */}
        <canvas
          ref={canvasRef}
          className="absolute inset-0 w-full h-full pointer-events-none"
        />
        {/* track count badge */}
        {tracks.length > 0 && (
          <div className="absolute top-2 left-2 bg-black/60 text-white text-xs px-2 py-1 rounded">
            {tracks.length} người phát hiện
          </div>
        )}
      </div>
    </div>
  )
}
