import { useState } from 'react'

function Section({ title, children }) {
  return (
    <div className="card flex flex-col gap-4">
      <h3 className="sec-title">{title}</h3>
      {children}
    </div>
  )
}

function Field({ label, hint, children }) {
  return (
    <div className="flex items-start justify-between gap-4">
      <div className="flex-1">
        <p className="text-sm text-slate-300">{label}</p>
        {hint && <p className="text-xs text-slate-500 mt-0.5">{hint}</p>}
      </div>
      <div className="w-48 flex-shrink-0">{children}</div>
    </div>
  )
}

function NumField({ value, onChange, min, max, step = 1 }) {
  return (
    <input type="number" min={min} max={max} step={step}
      className="input-field w-full text-right"
      value={value}
      onChange={e => onChange(parseFloat(e.target.value))} />
  )
}

function SelectField({ value, onChange, options }) {
  return (
    <select className="input-field w-full" value={value} onChange={e => onChange(e.target.value)}>
      {options.map(o => (
        <option key={o.value} value={o.value}>{o.label}</option>
      ))}
    </select>
  )
}

function Toggle({ value, onChange }) {
  return (
    <button
      onClick={() => onChange(!value)}
      className={`w-12 h-6 rounded-full transition-colors relative ${value ? 'bg-indigo-600' : 'bg-slate-700'}`}
    >
      <span className={`absolute top-0.5 w-5 h-5 bg-white rounded-full shadow transition-transform ${value ? 'translate-x-6' : 'translate-x-0.5'}`} />
    </button>
  )
}

const DEFAULT_CFG = {
  // Camera / Detection
  detector_model: 'yolo11n',
  detector_conf: 0.25,
  detector_iou: 0.45,
  input_size: 640,

  // Tracker
  tracker_type: 'bytetrack',
  track_high_thresh: 0.5,
  track_low_thresh: 0.1,
  new_track_thresh: 0.6,
  track_buffer: 30,
  match_thresh: 0.8,

  // Face Detection
  face_detector: 'scrfd_10g',
  face_conf: 0.5,

  // Face Recognition
  rec_model: 'buffalo_s',
  similarity_threshold: 0.35,
  confirmation_hits: 3,
  history_size: 5,
  switch_margin: 0.08,

  // Head Pose
  head_pose_backend: 'mediapipe',
  yaw_side_threshold: 30.0,
  pitch_down_threshold: 25.0,

  // Pose Estimation
  pose_model: 'yolo11n-pose',
  pose_conf: 0.3,

  // Object Detection
  obj_detector_model: 'yolo11n',
  obj_conf: 0.35,
  obj_iou: 0.45,

  // Behavior
  behavior_history_size: 12,
  min_state_frames: 3,

  // Report / LLM
  report_provider: 'google',
  report_model: 'gemini-2.0-flash',
  report_language: 'vi',
  report_temperature: 0.7,
  report_max_tokens: 2048,
}

export default function Settings() {
  const [cfg, setCfg] = useState(DEFAULT_CFG)
  const set = (key) => (val) => setCfg(c => ({ ...c, [key]: val }))

  return (
    <div className="flex flex-col gap-4 overflow-y-auto pb-6">
      <div className="flex items-center justify-between">
        <h2 className="sec-title">Cài đặt hệ thống</h2>
        <div className="flex gap-2">
          <button className="btn-secondary" onClick={() => setCfg(DEFAULT_CFG)}>Đặt lại mặc định</button>
          <button className="btn-primary">Lưu cài đặt</button>
        </div>
      </div>

      {/* 1. Person Detection */}
      <Section title="1. Person Detection (YOLO)">
        <Field label="Model" hint="YOLO11n nhẹ nhất, YOLO11s cân bằng hơn">
          <SelectField value={cfg.detector_model} onChange={set('detector_model')} options={[
            { value: 'yolo11n', label: 'YOLO11n (nhanh)' },
            { value: 'yolo11s', label: 'YOLO11s (cân bằng)' },
            { value: 'yolo26n', label: 'YOLO26n' },
            { value: 'yolo26s', label: 'YOLO26s' },
          ]} />
        </Field>
        <Field label="Confidence threshold" hint="Ngưỡng xác suất để chấp nhận detection">
          <NumField value={cfg.detector_conf} onChange={set('detector_conf')} min={0.1} max={0.9} step={0.05} />
        </Field>
        <Field label="IoU threshold" hint="Non-max suppression threshold">
          <NumField value={cfg.detector_iou} onChange={set('detector_iou')} min={0.1} max={0.9} step={0.05} />
        </Field>
        <Field label="Input size (px)" hint="640 là chuẩn YOLO">
          <SelectField value={cfg.input_size} onChange={v => set('input_size')(parseInt(v))} options={[
            { value: 320, label: '320' }, { value: 480, label: '480' },
            { value: 640, label: '640 (mặc định)' }, { value: 1280, label: '1280' },
          ]} />
        </Field>
      </Section>

      {/* 2. Multi-Object Tracking */}
      <Section title="2. Multi-Object Tracking">
        <Field label="Tracker" hint="ByteTrack (nhanh) hoặc BoT-SORT (dùng appearance)">
          <SelectField value={cfg.tracker_type} onChange={set('tracker_type')} options={[
            { value: 'bytetrack', label: 'ByteTrack' },
            { value: 'botsort',   label: 'BoT-SORT' },
          ]} />
        </Field>
        <Field label="High confidence threshold" hint="track_high_thresh">
          <NumField value={cfg.track_high_thresh} onChange={set('track_high_thresh')} min={0.1} max={0.99} step={0.05} />
        </Field>
        <Field label="Low confidence threshold" hint="track_low_thresh">
          <NumField value={cfg.track_low_thresh} onChange={set('track_low_thresh')} min={0.01} max={0.5} step={0.05} />
        </Field>
        <Field label="New track threshold" hint="new_track_thresh">
          <NumField value={cfg.new_track_thresh} onChange={set('new_track_thresh')} min={0.1} max={0.99} step={0.05} />
        </Field>
        <Field label="Track buffer (frames)" hint="Số frame giữ track khi mất dấu">
          <NumField value={cfg.track_buffer} onChange={set('track_buffer')} min={1} max={120} />
        </Field>
        <Field label="Match threshold" hint="match_thresh — IoU để ghép track">
          <NumField value={cfg.match_thresh} onChange={set('match_thresh')} min={0.1} max={0.99} step={0.05} />
        </Field>
      </Section>

      {/* 3. Face Detection */}
      <Section title="3. Face Detection">
        <Field label="Detector" hint="SCRFD nhanh hơn, RetinaFace chính xác hơn">
          <SelectField value={cfg.face_detector} onChange={set('face_detector')} options={[
            { value: 'scrfd_10g',    label: 'SCRFD-10G (mặc định)' },
            { value: 'scrfd_2.5g',  label: 'SCRFD-2.5G (nhanh)' },
            { value: 'retinaface',   label: 'RetinaFace' },
          ]} />
        </Field>
        <Field label="Confidence threshold">
          <NumField value={cfg.face_conf} onChange={set('face_conf')} min={0.1} max={0.99} step={0.05} />
        </Field>
      </Section>

      {/* 4. Face Recognition */}
      <Section title="4. Face Recognition (InsightFace)">
        <Field label="Model" hint="buffalo_s nhẹ, buffalo_l chính xác hơn">
          <SelectField value={cfg.rec_model} onChange={set('rec_model')} options={[
            { value: 'buffalo_s', label: 'buffalo_s (mặc định)' },
            { value: 'buffalo_l', label: 'buffalo_l (chính xác hơn)' },
          ]} />
        </Field>
        <Field label="Similarity threshold" hint="Cosine similarity tối thiểu để nhận dạng (0–1)">
          <NumField value={cfg.similarity_threshold} onChange={set('similarity_threshold')} min={0.1} max={0.99} step={0.01} />
        </Field>
        <Field label="Confirmation hits" hint="Số lần nhận dạng liên tiếp để xác nhận">
          <NumField value={cfg.confirmation_hits} onChange={set('confirmation_hits')} min={1} max={10} />
        </Field>
        <Field label="History size" hint="Số lịch sử nhận dạng để xét chuyển đổi">
          <NumField value={cfg.history_size} onChange={set('history_size')} min={1} max={20} />
        </Field>
        <Field label="Switch margin" hint="Biên để chuyển giữa 2 người (phòng chớp nháy)">
          <NumField value={cfg.switch_margin} onChange={set('switch_margin')} min={0.01} max={0.5} step={0.01} />
        </Field>
      </Section>

      {/* 5. Head Pose */}
      <Section title="5. Head Pose Estimation">
        <Field label="Backend" hint="MediaPipe + solvePnP hoặc 6DRepNet">
          <SelectField value={cfg.head_pose_backend} onChange={set('head_pose_backend')} options={[
            { value: 'mediapipe', label: 'MediaPipe + solvePnP (mặc định)' },
            { value: '6drepnet',  label: '6DRepNet' },
          ]} />
        </Field>
        <Field label="Yaw side threshold (°)" hint="Góc lệch ngang để xem là nhìn sang bên">
          <NumField value={cfg.yaw_side_threshold} onChange={set('yaw_side_threshold')} min={10} max={60} step={1} />
        </Field>
        <Field label="Pitch down threshold (°)" hint="Góc cúi đầu để xem là nhìn xuống">
          <NumField value={cfg.pitch_down_threshold} onChange={set('pitch_down_threshold')} min={10} max={50} step={1} />
        </Field>
      </Section>

      {/* 6. Pose Estimation */}
      <Section title="6. Pose Estimation (YOLO-Pose)">
        <Field label="Model">
          <SelectField value={cfg.pose_model} onChange={set('pose_model')} options={[
            { value: 'yolo11n-pose', label: 'YOLO11n-Pose (mặc định)' },
            { value: 'yolo11s-pose', label: 'YOLO11s-Pose' },
          ]} />
        </Field>
        <Field label="Confidence threshold">
          <NumField value={cfg.pose_conf} onChange={set('pose_conf')} min={0.1} max={0.99} step={0.05} />
        </Field>
      </Section>

      {/* 7. Object Detection */}
      <Section title="7. Object Detection">
        <Field label="Model">
          <SelectField value={cfg.obj_detector_model} onChange={set('obj_detector_model')} options={[
            { value: 'yolo11n', label: 'YOLO11n' },
            { value: 'yolo11s', label: 'YOLO11s' },
          ]} />
        </Field>
        <Field label="Confidence threshold">
          <NumField value={cfg.obj_conf} onChange={set('obj_conf')} min={0.1} max={0.99} step={0.05} />
        </Field>
        <Field label="IoU threshold">
          <NumField value={cfg.obj_iou} onChange={set('obj_iou')} min={0.1} max={0.99} step={0.05} />
        </Field>
        <p className="text-xs text-slate-500">
          Phát hiện: 📱 cell phone · 💻 laptop · 📖 book · 🍶 bottle · 🎒 backpack · ☕ cup · 🥪 sandwich · 🍕 pizza · 🎂 cake
        </p>
      </Section>

      {/* 8. Behavior Analysis */}
      <Section title="8. Behavior Analysis">
        <Field label="History size (frames)" hint="Số frame lịch sử để phân tích xu hướng">
          <NumField value={cfg.behavior_history_size} onChange={set('behavior_history_size')} min={3} max={60} />
        </Field>
        <Field label="Min state frames" hint="Số frame tối thiểu để xác nhận trạng thái mới">
          <NumField value={cfg.min_state_frames} onChange={set('min_state_frames')} min={1} max={20} />
        </Field>
        <div className="text-xs text-slate-500 space-y-0.5">
          <p>Trạng thái: <span className="text-green-400">focused</span> · <span className="text-amber-400">drowsy</span> · <span className="text-red-400">using_phone</span> · <span className="text-orange-400">off_task</span> · <span className="text-slate-400">away_from_seat</span> · <span className="text-purple-400">side_talking</span></p>
        </div>
      </Section>

      {/* Report / LLM */}
      <Section title="Báo cáo & LLM">
        <Field label="Provider">
          <SelectField value={cfg.report_provider} onChange={set('report_provider')} options={[
            { value: 'google',    label: 'Google Gemini' },
            { value: 'openai',    label: 'OpenAI' },
            { value: 'anthropic', label: 'Anthropic' },
          ]} />
        </Field>
        <Field label="Model" hint="Model mặc định: gemini-2.0-flash">
          <input className="input-field w-full" value={cfg.report_model}
            onChange={e => set('report_model')(e.target.value)} />
        </Field>
        <Field label="Ngôn ngữ báo cáo">
          <SelectField value={cfg.report_language} onChange={set('report_language')} options={[
            { value: 'vi', label: 'Tiếng Việt' },
            { value: 'en', label: 'English' },
          ]} />
        </Field>
        <Field label="Temperature" hint="Độ sáng tạo của LLM (0.0–1.0)">
          <NumField value={cfg.report_temperature} onChange={set('report_temperature')} min={0} max={1} step={0.1} />
        </Field>
        <Field label="Max output tokens">
          <NumField value={cfg.report_max_tokens} onChange={set('report_max_tokens')} min={512} max={8192} step={256} />
        </Field>
      </Section>
    </div>
  )
}
