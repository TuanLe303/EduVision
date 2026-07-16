import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { api } from '../api'

function CreateSessionModal({ onClose, onCreated }) {
  const qc = useQueryClient()
  const [className, setClassName] = useState('')

  const create = useMutation({
    mutationFn: () => api.startSession(className),
    onSuccess: (data) => {
      qc.invalidateQueries(['sessions'])
      onCreated(data.id)
    },
  })

  return (
    <div className="fixed inset-0 bg-black/70 flex items-center justify-center z-50">
      <div className="card w-96 flex flex-col gap-4">
        <div className="flex items-center justify-between">
          <h3 className="sec-title mb-0">Tạo phiên học mới</h3>
          <button onClick={onClose} className="text-slate-400 hover:text-white">✕</button>
        </div>
        
        <div>
          <label className="text-xs text-slate-400 block mb-1">Tên lớp học</label>
          <input
            className="input-field w-full"
            placeholder="VD: DPL302m - Nhóm 1"
            value={className}
            onChange={e => setClassName(e.target.value)}
            autoFocus
          />
        </div>

        {create.error && <p className="text-xs text-red-400">{create.error.message}</p>}

        <div className="flex gap-2 mt-2">
          <button onClick={onClose} className="btn-secondary flex-1">Hủy</button>
          <button
            onClick={() => create.mutate()}
            disabled={!className || create.isPending}
            className="btn-primary flex-1"
          >
            {create.isPending ? 'Đang tạo...' : 'Tạo phiên'}
          </button>
        </div>
      </div>
    </div>
  )
}

export default function Sessions({ onSelectSession }) {
  const [search, setSearch] = useState('')
  const [showModal, setShowModal] = useState(false)
  const { data: sessions = [], isLoading, error } = useQuery({
    queryKey: ['sessions'],
    queryFn: () => api.getSessions(),
  })

  const filtered = sessions.filter(s =>
    !search || s.class_name?.toLowerCase().includes(search.toLowerCase()) ||
    s.id?.toString().includes(search)
  )

  const qc = useQueryClient()
  const deleteSession = useMutation({
    mutationFn: (id) => api.deleteSession(id),
    onSuccess: () => qc.invalidateQueries(['sessions'])
  })

  const handleDelete = (e, id) => {
    e.stopPropagation()
    if (confirm(`Bạn có chắc chắn muốn xóa phiên học #${id} và toàn bộ báo cáo, dữ liệu đi kèm không?`)) {
      deleteSession.mutate(id)
    }
  }

  return (
    <div className="card flex flex-col gap-4">
      <div className="flex items-center justify-between">
        <h2 className="sec-title">Phiên học</h2>
        <div className="flex gap-2">
          <input
            className="input-field w-64"
            placeholder="Tìm kiếm..."
            value={search}
            onChange={e => setSearch(e.target.value)}
          />
          <button className="btn-primary" onClick={() => setShowModal(true)}>
            + Tạo phiên mới
          </button>
        </div>
      </div>

      {isLoading && <p className="text-slate-400 text-sm">Đang tải...</p>}
      {error && <p className="text-red-400 text-sm">Lỗi: {error.message}</p>}

      {!isLoading && filtered.length === 0 && (
        <p className="text-slate-500 text-sm text-center py-8">Chưa có phiên học nào</p>
      )}

      {filtered.length > 0 && (
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-slate-800 text-left">
                <th className="pb-2 text-slate-400 font-medium">ID</th>
                <th className="pb-2 text-slate-400 font-medium">Lớp</th>
                <th className="pb-2 text-slate-400 font-medium">Thời gian bắt đầu</th>
                <th className="pb-2 text-slate-400 font-medium">Kết thúc</th>
                <th className="pb-2 text-slate-400 font-medium">Sinh viên</th>
                <th className="pb-2 text-slate-400 font-medium">Chú ý</th>
                <th className="pb-2 text-slate-400 font-medium">Báo cáo</th>
                <th className="pb-2 text-slate-400 font-medium text-right">Thao tác</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800">
              {filtered.map(s => (
                <tr
                  key={s.id}
                  onClick={() => onSelectSession?.(s.id)}
                  className="hover:bg-slate-800/50 cursor-pointer transition-colors"
                >
                  <td className="py-2.5 font-mono text-slate-300">#{s.id}</td>
                  <td className="py-2.5 text-white">{s.class_name ?? '—'}</td>
                  <td className="py-2.5 text-slate-300">
                    {s.start_time ? new Date(s.start_time).toLocaleString('vi-VN') : '—'}
                  </td>
                  <td className="py-2.5 text-slate-300">
                    {s.end_time ? new Date(s.end_time).toLocaleString('vi-VN') : (
                      <span className="text-green-400 text-xs">Đang diễn ra</span>
                    )}
                  </td>
                  <td className="py-2.5 text-slate-300">{s.student_count ?? '—'}</td>
                  <td className="py-2.5">
                    {s.attention_pct !== undefined ? (
                      <div className="flex items-center gap-2">
                        <div className="w-16 h-1.5 bg-slate-700 rounded-full overflow-hidden">
                          <div className="h-full bg-indigo-500 rounded-full"
                            style={{ width: `${s.attention_pct}%` }} />
                        </div>
                        <span className="text-slate-300 text-xs">{s.attention_pct}%</span>
                      </div>
                    ) : '—'}
                  </td>
                  <td className="py-2.5">
                    {s.has_report ? (
                      <span className="text-xs text-green-400 bg-green-900/40 px-2 py-0.5 rounded">Có</span>
                    ) : (
                      <span className="text-xs text-slate-500">Chưa có</span>
                    )}
                  </td>
                  <td className="py-2.5 text-right">
                    <button
                      onClick={(e) => handleDelete(e, s.id)}
                      className="text-slate-500 hover:text-red-400 p-1 rounded transition-colors"
                      title="Xóa phiên học"
                    >
                      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                      </svg>
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {showModal && (
        <CreateSessionModal 
          onClose={() => setShowModal(false)} 
          onCreated={onSelectSession} 
        />
      )}
    </div>
  )
}
