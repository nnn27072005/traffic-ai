'use client'
import React, { useState, useEffect } from 'react'
import { Maximize2, Pause, Play, RefreshCw, AlertCircle } from 'lucide-react'

interface VideoPlayerProps {
  sessionId?: number
  isActive?: boolean
}

export const VideoPlayer: React.FC<VideoPlayerProps> = ({ sessionId, isActive = true }) => {
  const [error, setError] = useState(false)
  const [isLoading, setIsLoading] = useState(true)
  const streamUrl = sessionId 
    ? `http://localhost:8000/api/stream/live/${sessionId}`
    : 'http://localhost:8000/api/stream/live/default' // Fallback or placeholder

  const handleRetry = () => {
    setError(false)
    setIsLoading(true)
  }

  return (
    <div className="glass relative group overflow-hidden" style={{ aspectRatio: '16/9', background: '#000' }}>
      {isActive && !error ? (
        <img 
          src={streamUrl} 
          alt="Traffic Stream" 
          className="w-full h-full object-cover"
          onLoad={() => setIsLoading(false)}
          onError={() => setError(true)}
        />
      ) : (
        <div className="absolute inset-0 flex flex-col items-center justify-center text-center p-6">
          <AlertCircle size={48} className="text-danger mb-4 opacity-50" />
          <h3 className="text-lg font-semibold mb-2">Stream Offline</h3>
          <p className="text-sm text-text-muted mb-4 max-w-xs">
            Unable to connect to the live processing feed. Make sure the backend session is active.
          </p>
          <button 
            onClick={handleRetry}
            className="btn-primary"
            style={{ padding: '8px 16px', fontSize: '0.875rem' }}
          >
            <RefreshCw size={16} /> Retry Connection
          </button>
        </div>
      )}

      {/* Loading Overlay */}
      {isLoading && isActive && !error && (
        <div className="absolute inset-0 bg-black flex items-center justify-center">
          <div className="flex flex-col items-center gap-3">
            <div className="w-8 h-8 border-4 border-primary border-t-transparent rounded-full animate-spin"></div>
            <span className="text-sm font-medium">Initializing Stream...</span>
          </div>
        </div>
      )}

      {/* Control Overlay */}
      <div className="absolute bottom-0 left-0 right-0 p-4 bg-gradient-to-t from-black/80 to-transparent opacity-0 group-hover:opacity-100 transition-opacity">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <button className="text-white hover:text-primary transition-colors">
              <Play size={20} fill="currentColor" />
            </button>
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 rounded-full bg-danger animate-pulse"></div>
              <span className="text-xs font-bold uppercase tracking-wider">Live Feed</span>
            </div>
          </div>
          <button className="text-white hover:text-primary transition-colors">
            <Maximize2 size={20} />
          </button>
        </div>
      </div>
    </div>
  )
}
