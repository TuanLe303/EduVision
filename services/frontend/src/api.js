import axios from 'axios'
import {
  MOCK_SESSIONS, MOCK_STUDENTS, MOCK_ATTENDANCE,
  MOCK_EVENTS, MOCK_SUMMARY, MOCK_REPORT,
} from './mock'

const http = axios.create({ baseURL: '/api', timeout: 5_000 })

// Fall back to mock data when backend is unreachable (network error or 5xx)
function withMock(apiFn, mockFn) {
  return async (...args) => {
    try {
      return await apiFn(...args)
    } catch (e) {
      const isNetworkOrServer = !e.response || e.response.status >= 500
      if (isNetworkOrServer) return mockFn(...args)
      throw e
    }
  }
}

export const api = {
  getSessions: withMock(
    (params) => http.get('/sessions', { params }).then(r => r.data),
    () => MOCK_SESSIONS,
  ),
  getSession: withMock(
    (id) => http.get(`/sessions/${id}`).then(r => r.data),
    (id) => MOCK_SESSIONS.find(s => s.id === Number(id)) ?? null,
  ),
  getAttendance: withMock(
    (id) => http.get(`/sessions/${id}/attendance`).then(r => r.data),
    () => MOCK_ATTENDANCE,
  ),
  getEvents: withMock(
    (id) => http.get(`/sessions/${id}/events`).then(r => r.data),
    () => MOCK_EVENTS,
  ),
  getSummary: withMock(
    (id) => http.get(`/sessions/${id}/summary`).then(r => r.data),
    () => MOCK_SUMMARY,
  ),
  getStudents: withMock(
    () => http.get('/students').then(r => r.data),
    () => MOCK_STUDENTS,
  ),
  getStudent: withMock(
    (id) => http.get(`/students/${id}`).then(r => r.data),
    (id) => MOCK_STUDENTS.find(s => s.student_id === id) ?? null,
  ),
  enrollStudent: (data) => http.post('/students', data).then(r => r.data),
  deleteStudent: (id) => http.delete(`/students/${id}`),

  // Session lifecycle — no mock fallback (must reach real backend)
  startSession: (className) =>
    http.post('/sessions/start', { class_name: className }).then(r => r.data),
  endSession: (id) =>
    http.post(`/sessions/${id}/end`).then(r => r.data),
  deleteSession: (id) =>
    http.delete(`/sessions/${id}`),

  // Pipeline Management
  uploadVideo: (file) => {
    const fd = new FormData()
    fd.append('file', file)
    return http.post('/upload_video', fd, {
      headers: { 'Content-Type': 'multipart/form-data' }
    }).then(r => r.data)
  },
  getPipelineStatus: () => http.get('/pipeline/status').then(r => r.data),
  startPipeline: (sessionId, source, targetFps) => 
    http.post(`/pipeline/start/${sessionId}`, { source, target_fps: targetFps }).then(r => r.data),
  stopPipeline: () => http.post('/pipeline/stop').then(r => r.data),

  generateReport: withMock(
    (sessionId, provider, language) =>
      http.post('/reports/generate', { session_id: sessionId, provider, language }).then(r => r.data),
    (sessionId) => ({ ...MOCK_REPORT, session_id: sessionId }),
  ),
  getReport: withMock(
    (sessionId) => http.get(`/reports/${sessionId}`).then(r => r.data),
    () => MOCK_REPORT,
  ),
}
