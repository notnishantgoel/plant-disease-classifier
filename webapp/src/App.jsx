import { useState, useEffect, useCallback } from 'react'
import ModelSelector from './components/ModelSelector'
import ImageUploader from './components/ImageUploader'
import ResultCard from './components/ResultCard'
import CompareAnalysis from './components/CompareAnalysis'
import './App.css'

const FALLBACK_MODELS = [
  { id: 'efficientnet', name: 'EfficientNet v2-S',        shortName: 'EfficientNet', accuracy: '94.86%', size: '78 MB',  description: "Google's compound-scaled mobile CNN optimised for accuracy-per-FLOP. Highest accuracy among all models." },
  { id: 'convnext',     name: 'ConvNeXt-Tiny',             shortName: 'ConvNeXt',     accuracy: '89.50%', size: '106 MB', description: 'Pure-CNN redesigned with transformer best-practices — depthwise convolutions, LayerNorm, and GELU activations.' },
  { id: 'vit_moe_v3',   name: 'Vision Transformer + MoE',  shortName: 'ViT + MoE',    accuracy: '91.125%', size: '37 MB',  description: 'Custom Vision Transformer with Mixture-of-Experts routing, learnable positional embeddings, 4 stacked transformer blocks.' },
  { id: 'yolo',         name: 'YOLOv11 Nano',              shortName: 'YOLO v11',     accuracy: '86.50%', size: '~3 MB',  description: 'Ultralytics YOLOv11 nano classification head — fastest inference, smallest model, ideal for edge deployment.' },
]

export default function App() {
  const [models, setModels] = useState(FALLBACK_MODELS)
  const [selected, setSelected] = useState(FALLBACK_MODELS[0])
  const [preview, setPreview] = useState(null)
  const [imageFile, setImageFile] = useState(null)
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [apiStatus, setApiStatus] = useState('checking')
  const [showCompare, setShowCompare] = useState(false)

  // Fetch available models from API on mount
  useEffect(() => {
    fetch('/models')
      .then((r) => r.json())
      .then((data) => {
        if (data.models?.length) {
          setModels(data.models)
          setSelected(data.models[0])
        }
        setApiStatus('ok')
      })
      .catch(() => setApiStatus('error'))
  }, [])

  const handleImageSelect = useCallback((file, url) => {
    setImageFile(file)
    setPreview(url)
    setResult(null)
    setError(null)
  }, [])

  const handleModelChange = useCallback((model) => {
    setSelected(model)
    setResult(null)
    setError(null)
  }, [])

  const handlePredict = async () => {
    if (!imageFile || !selected) return
    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const form = new FormData()
      form.append('file', imageFile)
      form.append('model_id', selected.id)

      const res = await fetch('/predict', { method: 'POST', body: form })
      if (!res.ok) {
        const err = await res.json().catch(() => ({}))
        throw new Error(err.detail ?? `Server error ${res.status}`)
      }
      setResult(await res.json())
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="app">
      {/* Header */}
      <header className="app-header">
        <div className="header-logo">🌿</div>
        <div>
          <h1 className="header-title">PlantGuard AI</h1>
          <p className="header-sub">Plant Disease Classification System</p>
        </div>

        {/* Compare button */}
        <button
          className="compare-btn"
          onClick={() => setShowCompare(true)}
          title="Compare all models"
        >
          📊 Compare Models
        </button>

        <div className={`api-badge api-badge--${apiStatus}`}>
          <span style={{ fontSize: '8px' }}>●</span>
          {apiStatus === 'ok' ? 'API Connected' : apiStatus === 'error' ? 'API Offline' : 'Connecting…'}
        </div>
      </header>

      <main className="app-main">
        {/* Model selector */}
        <ModelSelector models={models} selected={selected} onSelect={handleModelChange} />

        {/* Workspace */}
        <div className="workspace">
          {/* Left: image upload */}
          <div className="panel panel--upload">
            <h2 className="panel-title">Upload Plant Image</h2>
            <ImageUploader onImageSelect={handleImageSelect} preview={preview} />

            <button
              id="predict-button"
              className="predict-btn"
              onClick={handlePredict}
              disabled={!imageFile || loading}
            >
              {loading ? (
                <>
                  <span className="spinner" /> Analyzing with {selected?.shortName}…
                </>
              ) : (
                <>🔬 Analyze Plant</>
              )}
            </button>

            {error && (
              <div className="error-box">
                <strong>⚠ Error:</strong> {error}
              </div>
            )}
          </div>

          {/* Right: results */}
          <div className="panel panel--result">
            <h2 className="panel-title">Diagnosis</h2>
            {result ? (
              <ResultCard result={result} />
            ) : (
              <div className="result-placeholder">
                {loading ? (
                  <div className="result-loading">
                    <div className="spinner spinner--lg" />
                    <p>Running inference…</p>
                  </div>
                ) : (
                  <>
                    <span className="placeholder-icon">🔬</span>
                    <p>Upload an image and click <strong>Analyze Plant</strong> to see the diagnosis here.</p>
                  </>
                )}
              </div>
            )}
          </div>
        </div>

        {/* Footer */}
        <footer className="app-footer">
          <p>
            Trained on 4,000 balanced plant disease images · 4 classes ·
            All models trained from scratch (no pre-trained weights)
          </p>
        </footer>
      </main>

      {/* Comparative analysis modal */}
      {showCompare && <CompareAnalysis onClose={() => setShowCompare(false)} />}
    </div>
  )
}
