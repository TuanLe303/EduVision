import { useEffect, useRef, useState, useCallback } from 'react'

export function useWebSocket(url, { enabled = true, onMessage } = {}) {
  const wsRef = useRef(null)
  const [status, setStatus] = useState('disconnected') // connecting | connected | disconnected | error

  const connect = useCallback(() => {
    if (!url || !enabled) return
    setStatus('connecting')
    const ws = new WebSocket(url)
    wsRef.current = ws

    ws.onopen  = () => setStatus('connected')
    ws.onclose = () => setStatus('disconnected')
    ws.onerror = () => setStatus('error')
    ws.onmessage = (e) => {
      try { onMessage?.(JSON.parse(e.data)) }
      catch { onMessage?.(e.data) }
    }
    return ws
  }, [url, enabled, onMessage])

  useEffect(() => {
    const ws = connect()
    return () => ws?.close()
  }, [connect])

  const disconnect = () => wsRef.current?.close()

  return { status, disconnect, reconnect: connect }
}
