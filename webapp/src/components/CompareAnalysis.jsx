import { useEffect, useState } from 'react'
import {
  BarChart, Bar, LineChart, Line,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
} from 'recharts'
import './CompareAnalysis.css'

// ── colours per model (consistent across every chart) ────────────────────────
const MODEL_COLOR = {
  efficientnet: '#22c55e',
  convnext:     '#38bdf8',
  vit_moe_v3:  '#a78bfa',
  yolo:         '#fb923c',
}

const MODEL_LABEL = {
  efficientnet: 'EfficientNet',
  convnext:     'ConvNeXt',
  vit_moe_v3:  'ViT + MoE',
  yolo:         'YOLO v11',
}

const ORDER = ['efficientnet', 'vit_moe_v3', 'convnext', 'yolo']

// ── shared tooltip ────────────────────────────────────────────────────────────
function DarkTooltip({ active, payload, label, suffix = '%' }) {
  if (!active || !payload?.length) return null
  return (
    <div className="chart-tooltip">
      <p className="chart-tooltip__label">{label}</p>
      {payload.map((p) => (
        <p key={p.dataKey} style={{ color: p.color }}>
          {p.name}: <strong>{p.value != null ? `${p.value}${suffix}` : '—'}</strong>
        </p>
      ))}
    </div>
  )
}

// ── tab buttons ───────────────────────────────────────────────────────────────
function Tabs({ tabs, active, onChange }) {
  return (
    <div className="ca-tabs">
      {tabs.map((t) => (
        <button
          key={t.id}
          className={`ca-tab ${active === t.id ? 'ca-tab--active' : ''}`}
          onClick={() => onChange(t.id)}
        >
          {t.icon} {t.label}
        </button>
      ))}
    </div>
  )
}

// ── section header ────────────────────────────────────────────────────────────
function SectionHead({ title, sub }) {
  return (
    <div className="ca-section-head">
      <h3 className="ca-section-title">{title}</h3>
      {sub && <p className="ca-section-sub">{sub}</p>}
    </div>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// TAB 1 — Overview: best-metric bar chart + stat cards
// ─────────────────────────────────────────────────────────────────────────────
function OverviewTab({ summary }) {
  // Bar chart data: one entry per metric
  const metrics = [
    { key: 'best_acc',  label: 'Accuracy' },
    { key: 'best_prec', label: 'Precision' },
    { key: 'best_rec',  label: 'Recall' },
    { key: 'best_f1',   label: 'F1 Score' },
  ]

  const barData = metrics.map(({ key, label }) => {
    const row = { metric: label }
    summary.forEach((m) => {
      row[MODEL_LABEL[m.model_id] ?? m.name] = m[key] != null ? +m[key].toFixed(2) : null
    })
    return row
  })

  // Stat cards
  const statCards = ORDER
    .map((id) => summary.find((m) => m.model_id === id))
    .filter(Boolean)

  return (
    <div className="ca-tab-content">
      {/* Stat cards */}
      <div className="ca-stat-grid">
        {statCards.map((m) => (
          <div key={m.model_id} className="ca-stat-card" style={{ '--accent': MODEL_COLOR[m.model_id] }}>
            <div className="ca-stat-name">{MODEL_LABEL[m.model_id] ?? m.name}</div>
            <div className="ca-stat-acc">{m.best_acc?.toFixed(2)}%</div>
            <div className="ca-stat-meta">
              {m.best_f1 != null && <span>F1 {m.best_f1?.toFixed(1)}%</span>}
              <span>Ep {m.best_epoch}</span>
            </div>
          </div>
        ))}
      </div>

      <SectionHead
        title="Best Performance Comparison"
        sub="Accuracy, Precision, Recall and F1 at each model's best epoch — YOLO excluded from P/R/F1 (metrics not logged)"
      />

      <div className="ca-chart-wrap ca-chart-wrap--tall">
        <ResponsiveContainer width="100%" height={320}>
          <BarChart data={barData} margin={{ top: 10, right: 20, left: 0, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(74,222,128,0.08)" />
            <XAxis dataKey="metric" tick={{ fill: 'rgba(134,239,172,0.6)', fontSize: 12 }} axisLine={false} tickLine={false} />
            <YAxis domain={[75, 100]} tickFormatter={(v) => `${v}%`} tick={{ fill: 'rgba(134,239,172,0.5)', fontSize: 11 }} axisLine={false} tickLine={false} />
            <Tooltip content={<DarkTooltip />} />
            <Legend wrapperStyle={{ paddingTop: 12 }} formatter={(v) => <span style={{ color: 'rgba(134,239,172,0.7)', fontSize: 12 }}>{v}</span>} />
            {ORDER.map((id) => {
              const label = MODEL_LABEL[id]
              return summary.find((m) => m.model_id === id) ? (
                <Bar key={id} dataKey={label} fill={MODEL_COLOR[id]} radius={[4, 4, 0, 0]} maxBarSize={40} />
              ) : null
            })}
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// TAB 2 — Training Curves: val_acc + val_loss over epochs
// ─────────────────────────────────────────────────────────────────────────────
function CurvesTab({ curves }) {
  // Merge by epoch (max 30 epochs)
  const maxEpoch = Math.max(...Object.values(curves).map((d) => d.length))
  const accData = Array.from({ length: maxEpoch }, (_, i) => {
    const row = { epoch: i + 1 }
    ORDER.forEach((id) => {
      const entry = curves[id]?.[i]
      row[MODEL_LABEL[id]] = entry?.val_acc ?? null
    })
    return row
  })

  const lossData = Array.from({ length: maxEpoch }, (_, i) => {
    const row = { epoch: i + 1 }
    ORDER.forEach((id) => {
      const entry = curves[id]?.[i]
      row[MODEL_LABEL[id]] = entry?.val_loss ?? null
    })
    return row
  })

  const trainAccData = Array.from({ length: maxEpoch }, (_, i) => {
    const row = { epoch: i + 1 }
    ORDER.filter((id) => id !== 'yolo').forEach((id) => {
      const entry = curves[id]?.[i]
      row[MODEL_LABEL[id]] = entry?.train_acc ?? null
    })
    return row
  })

  const lineProps = (id) => ({
    type: 'monotone',
    dataKey: MODEL_LABEL[id],
    stroke: MODEL_COLOR[id],
    strokeWidth: 2,
    dot: false,
    activeDot: { r: 4, strokeWidth: 0 },
    connectNulls: false,
  })

  const axisProps = {
    tick: { fill: 'rgba(134,239,172,0.5)', fontSize: 11 },
    axisLine: false,
    tickLine: false,
  }

  return (
    <div className="ca-tab-content">
      {/* Validation accuracy */}
      <SectionHead title="Validation Accuracy over Epochs" sub="All 4 models — YOLO trained for 10 epochs" />
      <div className="ca-chart-wrap">
        <ResponsiveContainer width="100%" height={270}>
          <LineChart data={accData} margin={{ top: 8, right: 20, left: 0, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(74,222,128,0.07)" />
            <XAxis dataKey="epoch" label={{ value: 'Epoch', position: 'insideBottom', offset: -2, fill: 'rgba(134,239,172,0.4)', fontSize: 11 }} {...axisProps} />
            <YAxis tickFormatter={(v) => `${v}%`} domain={[45, 100]} {...axisProps} />
            <Tooltip content={<DarkTooltip />} />
            <Legend wrapperStyle={{ paddingTop: 8 }} formatter={(v) => <span style={{ color: 'rgba(134,239,172,0.7)', fontSize: 12 }}>{v}</span>} />
            {ORDER.map((id) => <Line key={id} {...lineProps(id)} />)}
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Train vs Val accuracy for each non-YOLO model */}
      <SectionHead title="Train Accuracy over Epochs" sub="Compared to validation — shows overfitting gap (YOLO excluded)" />
      <div className="ca-chart-wrap">
        <ResponsiveContainer width="100%" height={270}>
          <LineChart data={trainAccData} margin={{ top: 8, right: 20, left: 0, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(74,222,128,0.07)" />
            <XAxis dataKey="epoch" label={{ value: 'Epoch', position: 'insideBottom', offset: -2, fill: 'rgba(134,239,172,0.4)', fontSize: 11 }} {...axisProps} />
            <YAxis tickFormatter={(v) => `${v}%`} domain={[45, 100]} {...axisProps} />
            <Tooltip content={<DarkTooltip />} />
            <Legend wrapperStyle={{ paddingTop: 8 }} formatter={(v) => <span style={{ color: 'rgba(134,239,172,0.7)', fontSize: 12 }}>{v}</span>} />
            {ORDER.filter((id) => id !== 'yolo').map((id) => <Line key={id} {...lineProps(id)} />)}
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Validation loss */}
      <SectionHead title="Validation Loss over Epochs" />
      <div className="ca-chart-wrap">
        <ResponsiveContainer width="100%" height={250}>
          <LineChart data={lossData} margin={{ top: 8, right: 20, left: 0, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(74,222,128,0.07)" />
            <XAxis dataKey="epoch" label={{ value: 'Epoch', position: 'insideBottom', offset: -2, fill: 'rgba(134,239,172,0.4)', fontSize: 11 }} {...axisProps} />
            <YAxis domain={[0.2, 1.5]} {...axisProps} />
            <Tooltip content={<DarkTooltip suffix="" />} />
            <Legend wrapperStyle={{ paddingTop: 8 }} formatter={(v) => <span style={{ color: 'rgba(134,239,172,0.7)', fontSize: 12 }}>{v}</span>} />
            {ORDER.map((id) => <Line key={id} {...lineProps(id)} />)}
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// TAB 3 — Metrics Guide: plain-English definitions
// ─────────────────────────────────────────────────────────────────────────────
const METRIC_DEFS = [
  {
    name: 'Accuracy',
    icon: '🎯',
    color: '#22c55e',
    formula: 'Correct Predictions ÷ Total Predictions',
    definition:
      'The percentage of images the model classified correctly out of all images it was tested on. If a model sees 200 images and gets 186 right, its accuracy is 93%.',
    when: 'Best used when all classes are equally represented in the dataset — which is the case here (balanced 200 images per class in validation).',
    caveat: 'Can be misleading on imbalanced datasets. A model that always predicts the majority class can still show high accuracy.',
  },
  {
    name: 'Precision',
    icon: '🔎',
    color: '#38bdf8',
    formula: 'True Positives ÷ (True Positives + False Positives)',
    definition:
      'Of all the times the model said "this is Spider Mite" (for example), how many were actually Spider Mite? High precision means the model rarely raises a false alarm.',
    when: 'Important when the cost of a false positive is high — e.g. incorrectly treating a healthy plant with pesticide.',
    caveat: 'A model can achieve 100% precision by only making predictions when it is extremely confident, but it would miss many real cases (low recall).',
  },
  {
    name: 'Recall  (Sensitivity)',
    icon: '🕵️',
    color: '#a78bfa',
    formula: 'True Positives ÷ (True Positives + False Negatives)',
    definition:
      'Of all the images that were actually Spider Mite, how many did the model correctly find? High recall means the model misses very few real cases.',
    when: 'Critical when missing a disease is costly — e.g. an infected plant that goes undetected can spread the disease to others.',
    caveat: 'A model can get 100% recall by predicting every class for every image, but precision would then collapse.',
  },
  {
    name: 'F1 Score',
    icon: '⚖️',
    color: '#fb923c',
    formula: '2 × (Precision × Recall) ÷ (Precision + Recall)',
    definition:
      'The harmonic mean of Precision and Recall. It gives a single number that balances both concerns — catching real cases (recall) without too many false alarms (precision). A score of 1.0 is perfect; 0.0 is worst.',
    when: 'The go-to summary metric when you care equally about precision and recall, especially on multi-class problems like this one.',
    caveat: 'Treats precision and recall as equally important. If your use case weights one higher, a weighted F-beta score would be more appropriate.',
  },
  {
    name: 'Loss  (Cross-Entropy)',
    icon: '📉',
    color: '#f472b6',
    formula: '−Σ [ y · log(p) ]',
    definition:
      'Measures not just whether the prediction was right, but how confident the model was. If the model said "90% Spider Mite" and it was correct, loss is low. If it said "51% Spider Mite" and squeaked by, loss is higher. Lower is better.',
    when: 'Monitored during training to detect overfitting (train loss keeps falling but val loss rises) or under-fitting (both stay high).',
    caveat: 'Loss can decrease while accuracy stays flat, or accuracy can improve while loss barely changes — always watch both.',
  },
  {
    name: 'Train vs Validation',
    icon: '🔀',
    color: '#facc15',
    formula: 'Train split: 700 imgs/class  ·  Val split: 200 imgs/class',
    definition:
      'Training metrics are computed on the images the model learned from. Validation metrics are computed on held-out images the model has never seen. Validation is the honest score.',
    when: 'A large gap between train accuracy and val accuracy signals overfitting — the model memorised the training data instead of learning general patterns.',
    caveat: 'All reported best metrics in this app use validation data, not training data.',
  },
]

function MetricsGuideTab() {
  return (
    <div className="ca-tab-content">
      <SectionHead
        title="Metrics Glossary"
        sub="Plain-English definitions of every measure used in the comparative analysis"
      />
      <div className="mg-grid">
        {METRIC_DEFS.map((m) => (
          <div key={m.name} className="mg-card" style={{ '--mg-color': m.color }}>
            <div className="mg-card-header">
              <span className="mg-icon">{m.icon}</span>
              <h4 className="mg-name">{m.name}</h4>
            </div>
            <div className="mg-formula">{m.formula}</div>
            <p className="mg-def">{m.definition}</p>
            <div className="mg-tags">
              <div className="mg-tag mg-tag--when">
                <span className="mg-tag-label">When it matters</span>
                <p>{m.when}</p>
              </div>
              <div className="mg-tag mg-tag--caveat">
                <span className="mg-tag-label">Watch out</span>
                <p>{m.caveat}</p>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// Main CompareAnalysis component
// ─────────────────────────────────────────────────────────────────────────────
export default function CompareAnalysis({ onClose }) {
  const [tab, setTab] = useState('overview')
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    fetch('/metrics')
      .then((r) => r.json())
      .then(setData)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  const TABS = [
    { id: 'overview', label: 'Overview',       icon: '📊' },
    { id: 'curves',   label: 'Training Curves', icon: '📈' },
    { id: 'glossary', label: 'Metrics Guide',   icon: '📖' },
  ]

  return (
    <div className="ca-overlay" onClick={onClose}>
      <div className="ca-panel" onClick={(e) => e.stopPropagation()}>
        {/* Header */}
        <div className="ca-header">
          <div>
            <h2 className="ca-title">📊 Comparative Model Analysis</h2>
            <p className="ca-subtitle">
              Accuracy · Precision · Recall · F1 · Loss — 4 models, all metrics
            </p>
          </div>
          <button className="ca-close" onClick={onClose} aria-label="Close">✕</button>
        </div>

        <Tabs tabs={TABS} active={tab} onChange={setTab} />

        <div className="ca-body">
          {loading && (
            <div className="ca-loading">
              <div className="spinner spinner--lg" />
              <p>Loading training metrics…</p>
            </div>
          )}

          {error && (
            <div className="ca-error">
              Could not load metrics: {error}. Make sure the API is running.
            </div>
          )}

          {data && tab === 'overview' && <OverviewTab summary={data.summary} />}
          {data && tab === 'curves'   && <CurvesTab   curves={data.curves}   />}
          {tab === 'glossary'         && <MetricsGuideTab />}
        </div>
      </div>
    </div>
  )
}
