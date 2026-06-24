import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { api } from '../api'

function MarkdownRenderer({ content }) {
  // Simple markdown rendering via innerHTML + basic conversion
  const html = content
    .replace(/^### (.+)$/gm, '<h3 class="text-base font-semibold text-slate-200 mt-4 mb-1">$1</h3>')
    .replace(/^## (.+)$/gm, '<h2 class="text-lg font-semibold text-white mt-5 mb-2">$1</h2>')
    .replace(/^# (.+)$/gm, '<h1 class="text-xl font-bold text-white mt-6 mb-2">$1</h1>')
    .replace(/\*\*(.+?)\*\*/g, '<strong class="text-white">$1</strong>')
    .replace(/\*(.+?)\*/g, '<em>$1</em>')
    .replace(/^- (.+)$/gm, '<li class="ml-4 list-disc text-slate-300">$1</li>')
    .replace(/^(\d+)\. (.+)$/gm, '<li class="ml-4 list-decimal text-slate-300">$2</li>')
    .replace(/\n\n/g, '</p><p class="mb-2 text-slate-300">')

  return (
    <div
      className="text-slate-300 text-sm leading-relaxed"
      dangerouslySetInnerHTML={{ __html: `<p class="mb-2">${html}</p>` }}
    />
  )
}

export default function Reports() {
  const qc = useQueryClient()
  const [sessionId, setSessionId] = useState('')
  const [provider, setProvider]   = useState('google')
  const [language, setLanguage]   = useState('vi')
  const [viewId, setViewId]       = useState(null)

  const { data: sessions = [] } = useQuery({
    queryKey: ['sessions'],
    queryFn: () => api.getSessions(),
  })

  const { data: report, isLoading: reportLoading } = useQuery({
    queryKey: ['report', viewId],
    queryFn: () => api.getReport(viewId),
    enabled: !!viewId,
  })

  const generate = useMutation({
    mutationFn: () => api.generateReport(sessionId, provider, language),
    onSuccess: (data) => {
      qc.invalidateQueries(['sessions'])
      if (data?.session_id) setViewId(data.session_id)
    },
  })

  return (
    <div className="flex gap-4 h-full">
      {/* sidebar */}
      <div className="w-72 flex-shrink-0 flex flex-col gap-4">
        <div className="card flex flex-col gap-3">
          <h3 className="sec-title">Tạo báo cáo mới</h3>
          <div>
            <label className="text-xs text-slate-400 block mb-1">Phiên học</label>
            <select
              className="input-field w-full"
              value={sessionId}
              onChange={e => setSessionId(e.target.value)}
            >
              <option value="">-- Chọn phiên --</option>
              {sessions.map(s => (
                <option key={s.id} value={s.id}>
                  #{s.id} {s.class_name ? `— ${s.class_name}` : ''}
                </option>
              ))}
            </select>
          </div>
          <div>
            <label className="text-xs text-slate-400 block mb-1">LLM Provider</label>
            <select className="input-field w-full" value={provider} onChange={e => setProvider(e.target.value)}>
              <option value="google">Google Gemini</option>
              <option value="openai">OpenAI</option>
              <option value="anthropic">Anthropic</option>
            </select>
          </div>
          <div>
            <label className="text-xs text-slate-400 block mb-1">Ngôn ngữ</label>
            <select className="input-field w-full" value={language} onChange={e => setLanguage(e.target.value)}>
              <option value="vi">Tiếng Việt</option>
              <option value="en">English</option>
            </select>
          </div>
          <button
            className="btn-primary w-full"
            onClick={() => generate.mutate()}
            disabled={!sessionId || generate.isPending}
          >
            {generate.isPending ? 'Đang tạo...' : 'Tạo báo cáo'}
          </button>
          {generate.error && (
            <p className="text-xs text-red-400">{generate.error.message}</p>
          )}
        </div>

        {/* session list with reports */}
        <div className="card flex-1 overflow-y-auto">
          <h3 className="sec-title mb-2">Báo cáo có sẵn</h3>
          <div className="space-y-1.5">
            {sessions.filter(s => s.has_report).map(s => (
              <button
                key={s.id}
                onClick={() => setViewId(s.id)}
                className={`w-full text-left px-3 py-2 rounded-lg text-sm transition-colors ${
                  viewId === s.id ? 'bg-indigo-600 text-white' : 'hover:bg-slate-800 text-slate-300'
                }`}
              >
                <div className="font-medium">#{s.id} {s.class_name ?? ''}</div>
                {s.start_time && (
                  <div className="text-xs opacity-70 mt-0.5">
                    {new Date(s.start_time).toLocaleDateString('vi-VN')}
                  </div>
                )}
              </button>
            ))}
            {sessions.filter(s => s.has_report).length === 0 && (
              <p className="text-xs text-slate-500">Chưa có báo cáo nào</p>
            )}
          </div>
        </div>
      </div>

      {/* report content */}
      <div className="flex-1 card overflow-y-auto">
        {!viewId && (
          <div className="h-full flex flex-col items-center justify-center text-slate-500 gap-3">
            <svg className="w-12 h-12" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1}
                d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
            </svg>
            <p className="text-sm">Chọn hoặc tạo báo cáo để xem</p>
          </div>
        )}
        {viewId && reportLoading && (
          <p className="text-slate-400 text-sm">Đang tải báo cáo...</p>
        )}
        {viewId && report && (
          <div className="flex flex-col gap-4">
            <div className="flex items-start justify-between">
              <div>
                <h2 className="sec-title">Báo cáo phiên #{viewId}</h2>
                {report.generated_at && (
                  <p className="text-xs text-slate-500">
                    Tạo lúc {new Date(report.generated_at).toLocaleString('vi-VN')}
                    {report.provider && ` · ${report.provider}`}
                  </p>
                )}
              </div>
              <button
                onClick={() => {
                  const blob = new Blob([report.content ?? ''], { type: 'text/markdown' })
                  const a = document.createElement('a')
                  a.href = URL.createObjectURL(blob)
                  a.download = `report-session-${viewId}.md`
                  a.click()
                }}
                className="btn-secondary text-xs px-3 py-1.5"
              >
                ↓ Tải xuống
              </button>
            </div>
            <div className="border-t border-slate-800 pt-4">
              <MarkdownRenderer content={report.content ?? ''} />
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
