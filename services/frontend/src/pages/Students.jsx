import { useState, useRef } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { api } from '../api'

function EnrollModal({ onClose }) {
  const qc = useQueryClient()
  const [form, setForm] = useState({ student_id: '', name: '', email: '' })
  const [imageFile, setImageFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const fileRef = useRef()

  const enroll = useMutation({
    mutationFn: () => {
      const fd = new FormData()
      Object.entries(form).forEach(([k, v]) => fd.append(k, v))
      if (imageFile) fd.append('image', imageFile)
      return api.enrollStudent(fd)
    },
    onSuccess: () => { qc.invalidateQueries(['students']); onClose() },
  })

  const handleFile = (e) => {
    const f = e.target.files[0]
    if (!f) return
    setImageFile(f)
    const reader = new FileReader()
    reader.onload = ev => setPreview(ev.target.result)
    reader.readAsDataURL(f)
  }

  return (
    <div className="fixed inset-0 bg-black/70 flex items-center justify-center z-50">
      <div className="card w-96 flex flex-col gap-4">
        <div className="flex items-center justify-between">
          <h3 className="sec-title mb-0">Đăng ký sinh viên mới</h3>
          <button onClick={onClose} className="text-slate-400 hover:text-white">✕</button>
        </div>

        {/* avatar upload */}
        <div className="flex flex-col items-center gap-2">
          <div
            onClick={() => fileRef.current?.click()}
            className="w-24 h-24 rounded-full bg-slate-700 border-2 border-dashed border-slate-600 flex items-center justify-center cursor-pointer hover:border-indigo-500 overflow-hidden transition-colors"
          >
            {preview
              ? <img src={preview} alt="preview" className="w-full h-full object-cover" />
              : <span className="text-3xl text-slate-500">📷</span>
            }
          </div>
          <input ref={fileRef} type="file" accept="image/*" className="hidden" onChange={handleFile} />
          <p className="text-xs text-slate-500">Click để upload ảnh khuôn mặt</p>
        </div>

        <div className="flex flex-col gap-3">
          {[
            { k: 'student_id', label: 'Mã sinh viên', placeholder: 'VD: SV001' },
            { k: 'name',       label: 'Họ và tên',    placeholder: 'Nguyễn Văn A' },
            { k: 'email',      label: 'Email',         placeholder: 'sv@email.com' },
          ].map(({ k, label, placeholder }) => (
            <div key={k}>
              <label className="text-xs text-slate-400 block mb-1">{label}</label>
              <input
                className="input-field w-full"
                placeholder={placeholder}
                value={form[k]}
                onChange={e => setForm(f => ({ ...f, [k]: e.target.value }))}
              />
            </div>
          ))}
        </div>

        {enroll.error && <p className="text-xs text-red-400">{enroll.error.message}</p>}

        <div className="flex gap-2">
          <button onClick={onClose} className="btn-secondary flex-1">Hủy</button>
          <button
            onClick={() => enroll.mutate()}
            disabled={!form.student_id || !form.name || enroll.isPending}
            className="btn-primary flex-1"
          >
            {enroll.isPending ? 'Đang lưu...' : 'Đăng ký'}
          </button>
        </div>
      </div>
    </div>
  )
}

export default function Students() {
  const [showModal, setShowModal] = useState(false)
  const [search, setSearch] = useState('')

  const { data: students = [], isLoading, error } = useQuery({
    queryKey: ['students'],
    queryFn: () => api.getStudents(),
  })

  const filtered = students.filter(s =>
    !search ||
    s.name?.toLowerCase().includes(search.toLowerCase()) ||
    s.student_id?.toLowerCase().includes(search.toLowerCase())
  )

  return (
    <div className="card flex flex-col gap-4">
      <div className="flex items-center justify-between">
        <h2 className="sec-title">Sinh viên ({students.length})</h2>
        <div className="flex gap-2">
          <input
            className="input-field w-56"
            placeholder="Tìm kiếm..."
            value={search}
            onChange={e => setSearch(e.target.value)}
          />
          <button className="btn-primary" onClick={() => setShowModal(true)}>
            + Đăng ký
          </button>
        </div>
      </div>

      {isLoading && <p className="text-slate-400 text-sm">Đang tải...</p>}
      {error && <p className="text-red-400 text-sm">Lỗi: {error.message}</p>}

      {!isLoading && filtered.length === 0 && (
        <div className="py-16 flex flex-col items-center gap-3 text-slate-500">
          <span className="text-4xl">👥</span>
          <p className="text-sm">Chưa có sinh viên nào</p>
          <button className="btn-primary" onClick={() => setShowModal(true)}>Đăng ký đầu tiên</button>
        </div>
      )}

      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-3">
        {filtered.map(s => (
          <div key={s.student_id ?? s.id} className="bg-slate-800/60 rounded-xl p-3 flex flex-col items-center gap-2 hover:bg-slate-800 transition-colors">
            <div className="w-16 h-16 rounded-full bg-slate-700 overflow-hidden flex items-center justify-center">
              {s.avatar_url
                ? <img src={s.avatar_url} alt={s.name} className="w-full h-full object-cover" />
                : <span className="text-2xl">{s.name?.[0] ?? '?'}</span>
              }
            </div>
            <div className="text-center">
              <p className="text-sm font-medium text-white leading-tight">{s.name}</p>
              <p className="text-xs text-slate-400 font-mono mt-0.5">{s.student_id}</p>
            </div>
            <span className={`text-xs px-2 py-0.5 rounded-full ${
              s.enrolled
                ? 'bg-green-900/40 text-green-400'
                : 'bg-slate-700 text-slate-400'
            }`}>
              {s.enrolled ? 'Đã đăng ký' : 'Chưa đăng ký'}
            </span>
          </div>
        ))}
      </div>

      {showModal && <EnrollModal onClose={() => setShowModal(false)} />}
    </div>
  )
}
