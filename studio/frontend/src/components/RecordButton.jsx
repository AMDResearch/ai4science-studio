import { useState, useRef } from 'react'

export function RecordButton() {
  const [recording, setRecording] = useState(false)
  const [converting, setConverting] = useState(false)
  const recorderRef = useRef(null)
  const chunksRef = useRef([])

  async function startRecording() {
    try {
      const stream = await navigator.mediaDevices.getDisplayMedia({
        video: { frameRate: 30, width: 1920, height: 1080 },
        audio: false,
      })
      const rec = new MediaRecorder(stream, { mimeType: 'video/webm;codecs=vp9' })
      chunksRef.current = []
      rec.ondataavailable = (e) => { if (e.data.size > 0) chunksRef.current.push(e.data) }
      rec.onstop = async () => {
        stream.getTracks().forEach(t => t.stop())
        setConverting(true)
        const blob = new Blob(chunksRef.current, { type: 'video/webm' })
        try {
          const fd = new FormData()
          fd.append('file', blob, 'recording.webm')
          const r = await fetch('/api/video/convert', { method: 'POST', body: fd })
          if (r.ok) {
            const { download_url } = await r.json()
            window.location.href = download_url
          }
        } catch (e) {
          // fallback: download webm directly
          const url = URL.createObjectURL(blob)
          const a = document.createElement('a')
          a.href = url; a.download = 'studio_demo.webm'; a.click()
          URL.revokeObjectURL(url)
        }
        setConverting(false)
      }
      rec.start(1000)
      recorderRef.current = rec
      setRecording(true)
    } catch (e) {
      console.error('Screen capture failed:', e)
    }
  }

  function stopRecording() {
    recorderRef.current?.stop()
    setRecording(false)
  }

  if (converting) return (
    <button className="btn btn-ghost" disabled style={{ flexShrink: 0, fontSize: '.75rem' }}>
      Converting...
    </button>
  )

  return (
    <button
      className={recording ? 'btn btn-primary' : 'btn btn-ghost'}
      style={{ flexShrink: 0, fontSize: '.75rem', padding: '.3rem .8rem' }}
      onClick={recording ? stopRecording : startRecording}
    >
      {recording ? '⏹ Stop' : '⏺ Record'}
    </button>
  )
}
