import { useRef, useState } from 'react'
import './ImageUploader.css'

export default function ImageUploader({ onImageSelect, preview }) {
  const fileRef = useRef(null)
  const cameraRef = useRef(null)
  const [dragging, setDragging] = useState(false)

  const handleFile = (file) => {
    if (!file || !file.type.startsWith('image/')) return
    const url = URL.createObjectURL(file)
    onImageSelect(file, url)
  }

  const onDrop = (e) => {
    e.preventDefault()
    setDragging(false)
    handleFile(e.dataTransfer.files[0])
  }

  return (
    <div className="uploader">
      {preview ? (
        <div className="uploader__preview">
          <img src={preview} alt="Selected plant" className="uploader__img" />
          <div className="uploader__overlay">
            <button
              className="uploader__change"
              onClick={() => fileRef.current.click()}
            >
              ✦ Change image
            </button>
          </div>
        </div>
      ) : (
        <div
          className={`uploader__drop ${dragging ? 'uploader__drop--over' : ''}`}
          onClick={() => fileRef.current.click()}
          onDragOver={(e) => { e.preventDefault(); setDragging(true) }}
          onDragLeave={() => setDragging(false)}
          onDrop={onDrop}
        >
          <div className="uploader__ring">
            <span className="uploader__icon">🌿</span>
          </div>
          <p className="uploader__title">Drop an image here</p>
          <p className="uploader__sub">or click to browse files</p>
          <div className="uploader__formats">JPG, PNG, WEBP</div>
        </div>
      )}

      <div className="uploader__actions">
        <button id="browse-button" className="btn btn--glass" onClick={() => fileRef.current.click()}>
          <span className="btn-icon">📁</span> Browse
        </button>
        <button id="camera-button" className="btn btn--glass" onClick={() => cameraRef.current.click()}>
          <span className="btn-icon">📷</span> Camera
        </button>
      </div>

      {/* hidden file inputs */}
      <input
        ref={fileRef}
        type="file"
        accept="image/*"
        style={{ display: 'none' }}
        onChange={(e) => handleFile(e.target.files[0])}
      />
      <input
        ref={cameraRef}
        type="file"
        accept="image/*"
        capture="environment"
        style={{ display: 'none' }}
        onChange={(e) => handleFile(e.target.files[0])}
      />
    </div>
  )
}
