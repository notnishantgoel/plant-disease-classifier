import './ResultCard.css'

const DISEASE_INFO = {
  Blight_fungus: {
    displayName: 'Blight (Fungal)',
    emoji: '🍂',
    description: 'Fungal disease causing rapid death of plant tissue. Appears as water-soaked lesions that turn brown/black.',
    treatment: 'Apply copper-based fungicides. Remove and destroy infected tissue. Improve drainage and air circulation.',
  },
  Mosiac_Virus: {
    displayName: 'Mosaic Virus',
    emoji: '🟡',
    description: 'Viral infection causing mottled yellow-green patterns on leaves, distortion, and stunted growth.',
    treatment: 'No direct cure. Remove infected plants, control aphid vectors, and use virus-resistant cultivars.',
  },
  Spider_Mite: {
    displayName: 'Spider Mite',
    emoji: '🕷️',
    description: 'Tiny arachnid pests feeding on plant cells, causing yellowing, stippling, and fine webbing on leaves.',
    treatment: 'Apply miticides or neem oil. Increase humidity. Introduce predatory mites for biological control.',
  },
  Thrip_pest: {
    displayName: 'Thrip Pest',
    emoji: '🪲',
    description: 'Tiny insects piercing plant cells and extracting contents, causing silvery streaks and leaf deformation.',
    treatment: 'Use insecticidal soap or spinosad. Apply blue sticky traps. Remove heavily infested plant parts.',
  },
}

export default function ResultCard({ result }) {
  if (!result) return null

  const { predicted_class, confidence, all_probabilities } = result
  const info = DISEASE_INFO[predicted_class] ?? {
    displayName: predicted_class,
    emoji: '🌿',
    description: '',
    treatment: '',
  }

  const sorted = Object.entries(all_probabilities).sort((a, b) => b[1] - a[1])
  const confLevel = confidence >= 80 ? 'high' : confidence >= 50 ? 'medium' : 'low'

  return (
    <div className="result-card" id="diagnosis-result">
      {/* Header */}
      <div className="result-header">
        <div className="result-emoji-wrapper">
          <span className="result-emoji">{info.emoji}</span>
        </div>
        <div className="result-header-info">
          <h3 className="result-class">{info.displayName}</h3>
          <div className="result-meta">
            <span className={`result-conf result-conf--${confLevel}`}>
              {confidence.toFixed(1)}% confidence
            </span>
          </div>
        </div>
      </div>

      {/* Description */}
      {info.description && (
        <div className="result-section">
          <p className="result-section-label">Description</p>
          <p className="result-text">{info.description}</p>
        </div>
      )}

      {/* Treatment */}
      {info.treatment && (
        <div className="result-section result-section--treatment">
          <p className="result-section-label">💊 Recommended Treatment</p>
          <p className="result-text">{info.treatment}</p>
        </div>
      )}

      {/* Probabilities */}
      <div className="result-section">
        <p className="result-section-label">All Class Probabilities</p>
        <div className="prob-list">
          {sorted.map(([cls, prob]) => {
            const d = DISEASE_INFO[cls]
            const isTop = cls === predicted_class
            return (
              <div key={cls} className={`prob-row ${isTop ? 'prob-row--top' : ''}`}>
                <span className="prob-label">{d?.displayName ?? cls}</span>
                <div className="prob-track">
                  <div
                    className={`prob-bar ${isTop ? 'prob-bar--top' : ''}`}
                    style={{ width: `${prob}%` }}
                  />
                </div>
                <span className={`prob-pct ${isTop ? 'prob-pct--top' : ''}`}>{prob.toFixed(1)}%</span>
              </div>
            )
          })}
        </div>
      </div>
    </div>
  )
}
