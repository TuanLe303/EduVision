import { useState, useRef, useCallback } from 'react'

export function useCamera() {
  const [status, setStatus] = useState('off') // off | requesting | on | error
  const [error, setError]   = useState(null)
  const streamRef = useRef(null)

  const start = useCallback(async (videoEl) => {
    setStatus('requesting')
    setError(null)
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'environment' },
        audio: false,
      })
      streamRef.current = stream
      if (videoEl) { videoEl.srcObject = stream }
      setStatus('on')
    } catch (e) {
      setError(e.name === 'NotAllowedError' ? 'Bị từ chối quyền camera' : e.message)
      setStatus('error')
    }
  }, [])

  const stop = useCallback((videoEl) => {
    streamRef.current?.getTracks().forEach(t => t.stop())
    streamRef.current = null
    if (videoEl) { videoEl.srcObject = null }
    setStatus('off')
    setError(null)
  }, [])

  return { status, error, start, stop }
}
