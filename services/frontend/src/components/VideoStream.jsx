import { useEffect, useRef, useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { useCamera } from '../hooks/useCamera'
import { BEHAVIOR_META, GAZE_ARROW, OBJECT_EMOJI } from '../constants'
import { api } from '../api'

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

export default function VideoStream({ tracks = [], frameW, frameH, wsStatus, session }) {
  const videoRef = useRef(null)
  const canvasRef = useRef(null)
  const qc = useQueryClient()
  const { status: camStatus, error: camError, start, stop } = useCamera()
  
  // Pipeline form states
  const [mode, setMode] = useState('browser') // 'browser' | 'rtsp' | 'webcam' | 'mp4'
  const [rtspUrl, setRtspUrl] = useState('rtsp://100.86.84.22:8554/live.sdp')
  const [mp4File, setMp4File] = useState(null)
  const [targetFps, setTargetFps] = useState(8)

  // Pipeline API
  const { data: pipelineStatus } = useQuery({
    queryKey: ['pipeline'],
    queryFn: () => api.getPipelineStatus(),
    refetchInterval: 3000,
  })

  const startPipeline = useMutation({
    mutationFn: async () => {
      let source = '0'
      if (mode === 'rtsp') source = rtspUrl
      if (mode === 'mp4') {
        if (!mp4File) throw new Error("Vui lòng chọn file MP4 trước khi bật AI!")
        const res = await api.uploadVideo(mp4File)
        source = res.source
      }
      return api.startPipeline(session?.id ?? 1, source, targetFps)
    },
    onSuccess: () => qc.invalidateQueries(['pipeline'])
  })

  const stopPipeline = useMutation({
    mutationFn: () => api.stopPipeline(),
    onSuccess: () => qc.invalidateQueries(['pipeline'])
  })

  // draw bbox overlay on every tracks change
  useEffect(() => {
    const video  = videoRef.current
    const canvas = canvasRef.current
    if (!video || !canvas || tracks.length === 0) return
    drawTracks(canvas, video, tracks)
  }, [tracks])

  const handleToggleBrowserCam = async () => {
    const video = videoRef.current
    if (camStatus === 'on') { stop(video) }
    else                    { await start(video) }
  }

  const handleTogglePipeline = () => {
    if (pipelineStatus?.is_running) {
      stopPipeline.mutate()
    } else {
      if (!session) {
        alert("Vui lòng tạo hoặc chọn một phiên học đang diễn ra trước khi bật AI!")
        return
      }
      startPipeline.mutate()
    }
  }

  const camActive = camStatus === 'on'
  const isPipelineRunning = pipelineStatus?.is_running

  return (
    <div className="card flex flex-col gap-3">
      {/* header bar */}
      <div className="flex flex-col gap-3 p-3 bg-slate-900 border-b border-slate-800 rounded-t-xl">
        <div className="flex items-center justify-between">
          <h2 className="sec-title mb-0">Luồng Video & AI</h2>
          <div className="flex items-center gap-2 text-xs">
            {wsStatus && (
              <span className={`px-2 py-0.5 rounded-full ${
                wsStatus === 'connected' ? 'bg-green-900 text-green-300' :
                wsStatus === 'connecting' ? 'bg-yellow-900 text-yellow-300' :
                'bg-slate-700 text-slate-400'
              }`}>
                WS {wsStatus}
              </span>
            )}
          </div>
        </div>

        {/* Controls */}
        <div className="flex flex-wrap items-end gap-3 bg-slate-950 p-3 rounded-lg border border-slate-800">
          <div>
            <label className="block text-xs text-slate-400 mb-1">Nguồn Camera</label>
            <select 
              value={mode} 
              onChange={e => setMode(e.target.value)}
              className="input-field text-xs py-1"
              disabled={isPipelineRunning}
            >
              <option value="browser">Camera Máy Tính (Trình duyệt)</option>
              <option value="webcam">Camera Máy Tính (Chạy AI)</option>
              <option value="rtsp">Link RTSP Điện Thoại (Chạy AI)</option>
              <option value="mp4">Upload Video MP4 (Chạy AI)</option>
            </select>
          </div>

          {mode === 'rtsp' && (
            <div className="flex-1 min-w-[200px]">
              <label className="block text-xs text-slate-400 mb-1">Link RTSP</label>
              <input 
                type="text" 
                value={rtspUrl}
                onChange={e => setRtspUrl(e.target.value)}
                className="input-field w-full text-xs py-1"
                placeholder="rtsp://100.x.x.x:8554/live.sdp"
                disabled={isPipelineRunning}
              />
            </div>
          )}

          {mode === 'mp4' && (
            <div className="flex-1 min-w-[200px]">
              <label className="block text-xs text-slate-400 mb-1">Chọn file MP4</label>
              <input 
                type="file" 
                accept="video/mp4"
                onChange={e => setMp4File(e.target.files[0])}
                className="input-field w-full text-xs py-1 text-slate-300"
                disabled={isPipelineRunning}
              />
            </div>
          )}

          {mode !== 'browser' && (
            <div className="w-32">
              <label className="block text-xs text-slate-400 mb-1">Target FPS: {targetFps}</label>
              <input 
                type="range" 
                min="1" max="30" 
                value={targetFps}
                onChange={e => setTargetFps(parseInt(e.target.value))}
                className="w-full accent-indigo-500"
                disabled={isPipelineRunning}
              />
            </div>
          )}

          <div>
            {mode === 'browser' ? (
              <button
                onClick={handleToggleBrowserCam}
                className={camActive ? 'btn-danger' : 'btn-primary'}
              >
                {camStatus === 'requesting' ? 'Đang kết nối…' : camActive ? 'Dừng Camera' : 'Bật Camera'}
              </button>
            ) : (
              <button
                onClick={handleTogglePipeline}
                className={isPipelineRunning ? 'btn-danger' : 'btn-primary'}
                disabled={startPipeline.isPending || stopPipeline.isPending}
              >
                {startPipeline.isPending ? 'Đang bật AI...' : 
                 stopPipeline.isPending ? 'Đang tắt AI...' :
                 isPipelineRunning ? 'Dừng Pipeline AI' : 'Bật Pipeline AI'}
              </button>
            )}
          </div>
        </div>
        {isPipelineRunning && (
          <p className="text-xs text-indigo-400 animate-pulse mt-1">
            Pipeline AI đang chạy ngầm trên backend. Cửa sổ OpenCV sẽ hiện lên để bạn theo dõi.
          </p>
        )}
      </div>

      {/* video + canvas stack */}
      <div className="relative bg-slate-900 overflow-hidden rounded-b-xl" style={{ aspectRatio: '16/9' }}>
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
