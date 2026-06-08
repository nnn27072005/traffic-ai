// frontend/src/app/logout/page.tsx
'use client'
import React, { useEffect } from 'react'
import { useRouter } from 'next/navigation'
import { ShieldAlert, RefreshCw, Power } from 'lucide-react'

export default function Logout() {
  const router = useRouter()

  useEffect(() => {
    // Clear any session data here if needed
    console.log("Session terminated")
  }, [])

  return (
    <div className="min-h-[80vh] flex items-center justify-center">
      <div className="glass-panel p-12 rounded-[48px] max-w-md w-full text-center space-y-8 relative overflow-hidden">
        <div className="scan-line"></div>
        <div className="hud-crosshair-tl"></div>
        <div className="hud-crosshair-tr"></div>
        <div className="hud-crosshair-bl"></div>
        <div className="hud-crosshair-br"></div>
        
        <div className="w-20 h-20 rounded-full bg-error/10 flex items-center justify-center mx-auto text-error animate-pulse">
           <Power size={40} />
        </div>
        
        <div className="space-y-2">
          <h1 className="text-3xl font-black tracking-tighter text-error">Session Terminated</h1>
          <p className="text-text-muted text-sm font-medium">
             Your secure connection to TRAFFIC_AI_CORE has been closed.
          </p>
        </div>

        <div className="pt-4 flex flex-col gap-4">
           <button 
             onClick={() => router.push('/')}
             className="w-full py-4 rounded-2xl bg-primary-container text-black font-black text-xs uppercase tracking-[0.2em] hover:scale-[1.02] transition-all shadow-[0_0_20px_var(--primary-glow)]"
           >
              Re-establish Connection
           </button>
           <div className="text-[10px] font-bold text-text-muted flex items-center justify-center gap-2">
              <RefreshCw size={12} className="animate-spin" /> Auto-redirect in 10s
           </div>
        </div>
      </div>
    </div>
  )
}
