export const BEHAVIOR_META = {
  focused:        { label: 'Focused',  badgeCls: 'badge-focused',  color: '#4ade80', hex: '#16a34a' },
  drowsy:         { label: 'Drowsy',   badgeCls: 'badge-drowsy',   color: '#fbbf24', hex: '#d97706' },
  using_phone:    { label: 'Phone',    badgeCls: 'badge-phone',    color: '#f87171', hex: '#dc2626' },
  off_task:       { label: 'Off-task', badgeCls: 'badge-offtask',  color: '#fb923c', hex: '#ea580c' },
  away_from_seat: { label: 'Away',     badgeCls: 'badge-away',     color: '#94a3b8', hex: '#475569' },
  side_talking:   { label: 'Talking',  badgeCls: 'badge-talking',  color: '#c084fc', hex: '#7c3aed' },
}

export const GAZE_ARROW = { center: '↑', down: '↓', left: '←', right: '→', up: '↑' }

export const OBJECT_EMOJI = {
  'cell phone': '📱', laptop: '💻', book: '📖', bottle: '🍶',
  backpack: '🎒', cup: '☕', sandwich: '🥪', pizza: '🍕', cake: '🎂',
}

export const API_BASE = '/api'
export const WS_STREAM = 'ws://localhost:8000/ws/stream'
export const WS_EVENTS = 'ws://localhost:8000/ws/events'
