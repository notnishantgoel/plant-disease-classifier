import './ModelSelector.css'

const MODEL_ICONS = {
  efficientnet: '⚡',
  convnext: '🔷',
  vit_moe_v3: '🧠',
  yolo: '🎯',
}

const MODEL_TAGS = {
  efficientnet: 'Best Accuracy',
  yolo: 'Fastest',
}

export default function ModelSelector({ models, selected, onSelect }) {
  return (
    <section className="model-selector" id="model-selector">
      <div className="selector-header">
        <h2 className="section-label">Select Model</h2>
        <span className="model-count">{models.length} models available</span>
      </div>
      <div className="model-grid">
        {models.map((m, i) => (
          <button
            key={m.id}
            id={`model-${m.id}`}
            className={`model-card ${selected?.id === m.id ? 'model-card--active' : ''}`}
            onClick={() => onSelect(m)}
            style={{ animationDelay: `${i * 0.06}s` }}
          >
            {MODEL_TAGS[m.id] && (
              <span className="model-tag">{MODEL_TAGS[m.id]}</span>
            )}
            <span className="model-icon">{MODEL_ICONS[m.id] ?? '🤖'}</span>
            <div className="model-info">
              <span className="model-short">{m.shortName}</span>
              <span className="model-acc">{m.accuracy}</span>
            </div>
            <div className="model-size">{m.size}</div>
            {selected?.id === m.id && <span className="model-check">✓</span>}
          </button>
        ))}
      </div>
      {selected && (
        <p className="model-desc">{selected.description}</p>
      )}
    </section>
  )
}
