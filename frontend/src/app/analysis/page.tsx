// frontend/src/app/analysis/page.tsx
'use client'
import React, { useState, useEffect } from 'react'
import { Upload, Play, Loader2, Video, BarChart2, ShieldAlert } from 'lucide-react'

interface ProcessingSession {
  id: number
  status: 'pending' | 'processing' | 'completed' | 'failed' | 'stopped' | 'cancelled'
  total_frames?: number | null
  processed_frames?: number | null
  output_video_path?: string | null
  thumbnail_url?: string
  total_violations?: number | null
}

export default function VideoAnalysis() {
  const [mode, setMode] = useState<'file' | 'url'>('file')
  const [url, setUrl] = useState('')
  const [file, setFile] = useState<File | null>(null)
  const [uploading, setUploading] = useState(false)
  const [session, setSession] = useState<ProcessingSession | null>(null)
  const [streamReady, setStreamReady] = useState(false)

  // Polling for session status
  useEffect(() => {
    // Check for existing session in localStorage on mount
    const savedSessionId = localStorage.getItem('active_traffic_session')
    if (savedSessionId && !session) {
      fetch(`http://localhost:8000/api/sessions/${savedSessionId}`)
        .then(res => res.json())
        .then((data: ProcessingSession) => {
          setStreamReady(false)
          setSession(data)
        })
        .catch(err => console.error('Failed to restore session', err))
    }

    let interval: ReturnType<typeof setInterval> | undefined
    if (session && (session.status === 'pending' || session.status === 'processing')) {
      // Save active session to localStorage
      localStorage.setItem('active_traffic_session', session.id.toString())
      
      interval = setInterval(async () => {
        try {
          const res = await fetch(`http://localhost:8000/api/sessions/${session.id}`)
          const data: ProcessingSession = await res.json()
          setSession(data)
          if (data.status === 'completed' || data.status === 'failed') {
            clearInterval(interval)
            // Optionally clear localStorage when done, or keep it to show last result
          }
        } catch (err) {
          console.error('Failed to poll session status', err)
        }
      }, 2000)
    }
    return () => clearInterval(interval)
  }, [session])

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0])
    }
  }

  const handleUpload = async () => {
    if (mode === 'file' && !file) return
    if (mode === 'url' && !url) return
    
    setUploading(true)
    try {
      let res;
      if (mode === 'file') {
        const formData = new FormData()
        formData.append('file', file!)
        res = await fetch('http://localhost:8000/api/video/upload', {
          method: 'POST',
          body: formData,
        })
      } else {
        res = await fetch('http://localhost:8000/api/video/url', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ url, stream_mode: true, save_output: false }),
        })
      }
      
      if (!res.ok) throw new Error('Action failed')
      const data: ProcessingSession = await res.json()
      setStreamReady(false)
      setSession(data)
      localStorage.setItem('active_traffic_session', data.id.toString())
    } catch (err) {
      console.error(err)
      alert(`Failed to initialize ${mode} analysis`)
    } finally {
      setUploading(false)
    }
  }

  const handleStop = async () => {
    if (!session) return
    try {
      const res = await fetch(`http://localhost:8000/api/sessions/${session.id}/stop`, {
        method: 'POST',
      })
      if (!res.ok) throw new Error('Stop failed')
      const data: ProcessingSession = await res.json()
      setSession(data)
    } catch (err) {
      console.error(err)
      alert('Failed to terminate session')
    }
  }

  const progress = session?.total_frames 
    ? Math.min(100, Math.round(((session.processed_frames ?? 0) / session.total_frames) * 100)) 
    : 0

  return (
    <div className="p-8 space-y-8 animate-in fade-in duration-700">
      <div className="flex items-center justify-between">
        <div>
           <h1 className="text-3xl font-black text-white uppercase tracking-tighter italic">Video Analysis <span className="text-primary-container">Protocol</span></h1>
           <p className="text-xs font-bold text-primary-container/40 uppercase tracking-[0.3em] mt-1">Deep Neural Frame Processing</p>
        </div>
        <div className="px-4 py-2 bg-primary-container/10 border border-primary-container/20 rounded-xl flex items-center gap-3">
           <div className="w-2 h-2 rounded-full bg-primary-container animate-pulse"></div>
           <span className="text-[10px] font-black text-primary-container uppercase tracking-widest">Neural Core Active</span>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Upload Section */}
        <div className="lg:col-span-1 space-y-6">
           <div className="glass-panel p-8 rounded-[40px] border-primary-container/10 relative overflow-hidden group">
              <div className="absolute inset-0 bg-primary-container/5 opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none"></div>
              
              <div className="flex items-center justify-between mb-6">
                 <h3 className="text-xs font-black text-primary-container uppercase tracking-[0.2em] flex items-center gap-2">
                    <Upload size={14} /> Data Ingestion
                 </h3>
                 <div className="flex bg-black/40 rounded-full p-1 border border-white/5">
                    <button 
                       onClick={() => setMode('file')}
                       className={`px-3 py-1 rounded-full text-[9px] font-black uppercase tracking-widest transition-all ${mode === 'file' ? 'bg-primary-container text-black' : 'text-primary-container/40 hover:text-primary-container'}`}
                    >File</button>
                    <button 
                       onClick={() => setMode('url')}
                       className={`px-3 py-1 rounded-full text-[9px] font-black uppercase tracking-widest transition-all ${mode === 'url' ? 'bg-primary-container text-black' : 'text-primary-container/40 hover:text-primary-container'}`}
                    >URL</button>
                 </div>
              </div>
              
              <div className="space-y-4">
                 {mode === 'file' ? (
                    <label className="block border-2 border-dashed border-primary-container/20 rounded-[32px] p-10 text-center cursor-pointer hover:border-primary-container/40 hover:bg-primary-container/5 transition-all group/label">
                       <input type="file" className="hidden" accept="video/*" onChange={handleFileChange} />
                       <Video size={40} className="mx-auto text-primary-container/20 group-hover/label:text-primary-container/60 transition-colors mb-4" />
                       <p className="text-[10px] font-bold text-primary-container/40 uppercase tracking-widest leading-relaxed truncate">
                          {file ? file.name : 'Select Video Source'}
                       </p>
                    </label>
                 ) : (
                    <div className="space-y-4">
                       <div className="relative">
                          <input 
                             type="text" 
                             placeholder="YouTube or Direct Video URL" 
                             value={url}
                             onChange={(e) => setUrl(e.target.value)}
                             className="w-full bg-black/40 border border-primary-container/20 rounded-2xl py-4 px-5 text-xs text-white placeholder:text-primary-container/20 focus:border-primary-container/60 focus:outline-none transition-all"
                          />
                       </div>
                       <p className="text-[9px] font-bold text-primary-container/30 uppercase tracking-[0.2em] px-2 leading-relaxed">
                          Enter a direct .mp4 link or a YouTube URL for neural stream ingestion.
                       </p>
                    </div>
                 )}

                 <button 
                    disabled={uploading || (mode === 'file' && !file) || (mode === 'url' && !url) || !!(session && (session.status === 'pending' || session.status === 'processing'))}
                    onClick={handleUpload}
                    className="w-full py-5 rounded-2xl bg-primary-container text-black font-black text-xs uppercase tracking-[0.2em] hover:bg-white disabled:opacity-30 disabled:cursor-not-allowed transition-all shadow-[0_0_30px_rgba(0,240,255,0.2)] flex items-center justify-center gap-2"
                 >
                    {uploading ? <Loader2 className="animate-spin" size={16} /> : <Play size={16} />}
                    {session && (session.status === 'pending' || session.status === 'processing') ? 'Processing Engine Active' : 'Initialize Analysis'}
                 </button>
                 
                 {session && (session.status === 'completed' || session.status === 'failed') && (
                    <button 
                       onClick={() => {
                          setSession(null)
                          localStorage.removeItem('active_traffic_session')
                       }}
                       className="w-full py-3 text-[10px] font-bold text-primary-container/40 uppercase tracking-widest hover:text-primary-container transition-colors"
                    >
                       Start New Session
                    </button>
                 )}
              </div>
           </div>

           {session && (
              <div className="glass-panel p-8 rounded-[40px] border-primary-container/10">
                 <h3 className="text-xs font-black text-primary-container uppercase tracking-[0.2em] mb-6 flex items-center gap-2">
                    <BarChart2 size={14} /> Session Telemetry
                 </h3>
                 <div className="space-y-4">
                    <div className="flex justify-between items-center py-2 border-b border-primary-container/5">
                       <span className="text-[9px] font-bold text-primary-container/30 uppercase tracking-widest">Status</span>
                       <span className={`text-[10px] font-black uppercase tracking-widest ${
                          session.status === 'completed' ? 'text-success' : 
                          session.status === 'failed' ? 'text-error' : 'text-primary-container animate-pulse'
                       }`}>
                          {session.status}
                       </span>
                    </div>
                    <div className="flex justify-between items-center py-2 border-b border-primary-container/5">
                       <span className="text-[9px] font-bold text-primary-container/30 uppercase tracking-widest">Progress</span>
                       <span className="text-xs font-mono text-white">{progress}%</span>
                    </div>
                    <div className="flex justify-between items-center py-2">
                       <span className="text-[9px] font-bold text-primary-container/30 uppercase tracking-widest">Frames</span>
                       <span className="text-xs font-mono text-white">{session.processed_frames} / {session.total_frames || '?'}</span>
                    </div>

                    {(session.status === 'processing' || session.status === 'pending') && (
                       <button 
                          onClick={handleStop}
                          className="w-full mt-4 py-3 rounded-xl bg-error/10 border border-error/20 text-error font-black text-[10px] uppercase tracking-widest hover:bg-error hover:text-white transition-all flex items-center justify-center gap-2"
                       >
                          < ShieldAlert size={14} /> Terminate Process
                       </button>
                    )}
                 </div>
              </div>
           )}
        </div>

        {/* Processing View / Live Feed */}
        <div className="lg:col-span-2 space-y-6">
           <div className="glass-panel p-4 rounded-[48px] border-primary-container/20 bg-black/40 aspect-video relative overflow-hidden flex items-center justify-center">
              {session?.status === 'completed' && session.output_video_path ? (
                 <video 
                    src={`http://localhost:8000/api/assets/${session.output_video_path}`} 
                    controls
                    className="w-full h-full object-contain rounded-[32px] shadow-[0_0_50px_rgba(0,0,0,0.5)]"
                    poster={session.thumbnail_url}
                 />
              ) : session?.status === 'processing' || session?.status === 'pending' ? (
                 <div className="relative w-full h-full flex items-center justify-center">
                    <img 
                       key={session.id}
                       src={`http://localhost:8000/api/stream/live/${session.id}`} 
                       alt="Neural Feed"
                       className={`w-full h-full object-contain rounded-[32px] transition-opacity duration-300 ${streamReady ? 'opacity-100' : 'opacity-0'}`}
                       onError={(e) => {
                          e.currentTarget.style.display = 'none'
                       }}
                       onLoad={(e) => {
                          e.currentTarget.style.display = 'block'
                          setStreamReady(true)
                       }}
                    />
                    {!streamReady && <div className="absolute inset-0 flex flex-col items-center justify-center bg-black/40 backdrop-blur-sm rounded-[32px]">
                       <Loader2 className="text-primary-container animate-spin mb-4" size={48} />
                       <p className="text-xs font-black text-primary-container uppercase tracking-[0.4em] animate-pulse">Waiting For First Frame</p>
                       <p className="text-[10px] text-white/40 mt-2 font-mono">{progress}% Complete</p>
                    </div>}
                    {streamReady && (
                       <div className="absolute top-6 left-6 px-3 py-2 rounded-full bg-black/60 border border-primary-container/20 backdrop-blur-md flex items-center gap-2">
                          <span className="w-2 h-2 rounded-full bg-primary-container animate-pulse"></span>
                          <span className="text-[9px] font-black text-primary-container uppercase tracking-widest">Live Processing</span>
                       </div>
                    )}
                 </div>
              ) : (
                 <div className="text-center space-y-4">
                    <div className="w-20 h-20 rounded-full bg-primary-container/5 border border-primary-container/10 flex items-center justify-center mx-auto">
                       <Video size={32} className="text-primary-container/20" />
                    </div>
                    <p className="text-[10px] font-bold text-primary-container/20 uppercase tracking-[0.4em]">Awaiting Uplink</p>
                 </div>
              )}
              
              {session && (session.status === 'processing' || session.status === 'pending') && (
                 <div className="absolute bottom-10 left-10 right-10">
                    <div className="h-1.5 w-full bg-white/5 rounded-full overflow-hidden border border-white/10 backdrop-blur-md">
                       <div 
                          className="h-full bg-primary-container shadow-[0_0_15px_#00f0ff] transition-all duration-500" 
                          style={{ width: `${progress}%` }}
                       ></div>
                    </div>
                 </div>
              )}
           </div>

           <div className="grid grid-cols-2 gap-6">
              <div className="glass-panel p-6 rounded-[32px] border-primary-container/10">
                 <p className="text-[9px] font-bold text-primary-container/30 uppercase tracking-widest mb-1 text-center">Threats Detected</p>
                 <p className="text-2xl font-black text-white text-center italic uppercase tracking-tighter">
                    {session?.status === 'completed' ? session.total_violations || '00' : '00'}
                 </p>
              </div>
              <div className="glass-panel p-6 rounded-[32px] border-primary-container/10">
                 <p className="text-[9px] font-bold text-primary-container/30 uppercase tracking-widest mb-1 text-center">Engine Latency</p>
                 <p className="text-2xl font-black text-white text-center italic uppercase tracking-tighter">
                    {session?.status === 'processing' ? '08ms' : '--'}
                 </p>
              </div>
           </div>
        </div>
      </div>
    </div>
  )
}
