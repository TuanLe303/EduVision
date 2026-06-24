import { BEHAVIOR_META } from '../constants'

export default function BehaviorBadge({ state }) {
  const meta = BEHAVIOR_META[state]
  if (!meta) return null
  return <span className={`badge ${meta.badgeCls}`}>{meta.label}</span>
}
