// frontend/src/app/profile/page.tsx
'use client'
import React, { useEffect } from 'react'
import { useAuth } from '@/context/AuthContext'
import { User, Mail, Shield, Zap, Globe, LogOut, ChevronLeft, Calendar, BadgeCheck } from 'lucide-react'
import Link from 'next/link'
import { useRouter } from 'next/navigation'

export default function Profile() {
  const { user, logout, loading } = useAuth()
  const router = useRouter()

  useEffect(() => {
    if (!loading && !user) {
      router.push('/login')
    }
  }, [loading, user, router])

  if (loading || !user) {
    return null
  }

  const handleLogout = () => {
    logout()
    router.push('/login')
  }
  const deployedAt = user.created_at ? new Date(user.created_at).toLocaleDateString() : 'N/A'

  return (
    <div className="min-h-screen bg-background relative overflow-hidden flex flex-col">
      {/* Background Grid & HUD Elements */}
      <div className="absolute inset-0 grid-bg opacity-10 pointer-events-none"></div>
      <div className="absolute top-20 left-20 w-64 h-64 bg-primary-container/5 rounded-full blur-[100px] pointer-events-none"></div>

      {/* Header / Navigation */}
      <header className="relative z-20 border-b border-primary-container/10 bg-surface/50 backdrop-blur-md px-8 py-4">
        <div className="max-w-7xl mx-auto flex items-center justify-between">
          <Link href="/" className="flex items-center gap-2 text-[10px] font-bold text-primary-container/60 hover:text-primary-container uppercase tracking-widest transition-colors group">
            <ChevronLeft size={14} className="group-hover:-translate-x-1 transition-transform" /> Mission Control
          </Link>
          <div className="flex items-center gap-4">
             <div className="text-right">
                <p className="text-[10px] font-black text-primary-container uppercase tracking-widest">{user.username}</p>
                <p className="text-[8px] font-bold text-primary-container/40 uppercase tracking-widest">{user.role}</p>
             </div>
             <div className="w-10 h-10 rounded-xl bg-primary-container/10 border border-primary-container/20 flex items-center justify-center overflow-hidden">
                {user.avatar_url ? (
                   <img src={user.avatar_url} alt="Profile" className="w-full h-full object-cover" />
                ) : (
                   <User size={20} className="text-primary-container" />
                )}
             </div>
          </div>
        </div>
      </header>

      <main className="flex-1 relative z-10 p-8">
        <div className="max-w-4xl mx-auto space-y-8">
           
           {/* Profile Header Card */}
           <div className="glass-panel p-10 rounded-[48px] border-primary-container/20 relative overflow-hidden">
              <div className="scan-line"></div>
              <div className="hud-crosshair-tl"></div>
              <div className="hud-crosshair-tr"></div>
              <div className="hud-crosshair-bl"></div>
              <div className="hud-crosshair-br"></div>

              <div className="flex flex-col md:flex-row items-center gap-10">
                 <div className="relative">
                    <div className="w-40 h-40 rounded-[40px] bg-surface border-2 border-primary-container/30 p-1 relative z-10 shadow-[0_0_50px_rgba(0,240,255,0.15)]">
                       <div className="w-full h-full rounded-[36px] overflow-hidden bg-black/40 flex items-center justify-center">
                          {user.avatar_url ? (
                             <img src={user.avatar_url} alt="Avatar" className="w-full h-full object-cover scale-110" />
                          ) : (
                             <User size={64} className="text-primary-container/20" />
                          )}
                       </div>
                    </div>
                    <div className="absolute -bottom-2 -right-2 bg-primary-container text-black p-2 rounded-2xl z-20 shadow-lg">
                       <BadgeCheck size={24} />
                    </div>
                    <div className="absolute inset-0 bg-primary-container/20 rounded-full blur-3xl opacity-20 animate-pulse"></div>
                 </div>

                 <div className="text-center md:text-left flex-1">
                    <div className="flex flex-wrap items-center justify-center md:justify-start gap-4 mb-2">
                       <h2 className="text-4xl font-black tracking-tighter text-white uppercase italic">{user.full_name || user.username}</h2>
                       <span className="px-3 py-1 bg-primary-container/10 border border-primary-container/30 rounded-full text-[9px] font-bold text-primary-container uppercase tracking-widest">
                          {user.role} Level 01
                       </span>
                    </div>
                    <p className="text-primary-container/60 font-mono text-sm tracking-tight mb-6">UID: {user.id.toString().padStart(8, '0')}</p>
                    
                    <div className="grid grid-cols-2 md:grid-cols-3 gap-6">
                       <div className="space-y-1">
                          <p className="text-[9px] font-bold text-primary-container/30 uppercase tracking-widest">Status</p>
                          <p className="text-xs font-bold text-success flex items-center gap-2">
                             <span className="w-2 h-2 rounded-full bg-success animate-ping"></span> Active Uplink
                          </p>
                       </div>
                       <div className="space-y-1">
                          <p className="text-[9px] font-bold text-primary-container/30 uppercase tracking-widest">Deployed</p>
                          <p className="text-xs font-bold text-white flex items-center gap-2 italic">
                             <Calendar size={12} className="text-primary-container/60" /> {deployedAt}
                          </p>
                       </div>
                       <div className="space-y-1">
                          <p className="text-[9px] font-bold text-primary-container/30 uppercase tracking-widest">Region</p>
                          <p className="text-xs font-bold text-white flex items-center gap-2 italic">
                             <Globe size={12} className="text-primary-container/60" /> Global Core
                          </p>
                       </div>
                    </div>
                 </div>
              </div>
           </div>

           {/* Stats / Details Grid */}
           <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
              <div className="glass-panel p-8 rounded-[40px] border-primary-container/10">
                 <h3 className="text-xs font-black text-primary-container uppercase tracking-[0.2em] mb-6 flex items-center gap-2">
                    <Zap size={14} /> Identity Details
                 </h3>
                 <div className="space-y-4">
                    <div className="flex items-center justify-between py-3 border-b border-primary-container/5">
                       <span className="text-[10px] font-bold text-primary-container/40 uppercase tracking-widest">Login Identity</span>
                       <span className="text-sm font-mono text-white">{user.username}</span>
                    </div>
                    <div className="flex items-center justify-between py-3 border-b border-primary-container/5">
                       <span className="text-[10px] font-bold text-primary-container/40 uppercase tracking-widest">Uplink Email</span>
                       <span className="text-sm font-mono text-white">{user.email || 'N/A'}</span>
                    </div>
                    <div className="flex items-center justify-between py-3 border-b border-primary-container/5">
                       <span className="text-[10px] font-bold text-primary-container/40 uppercase tracking-widest">Auth Protocol</span>
                       <span className="text-sm font-mono text-primary-container uppercase">{user.google_id ? 'Google OAuth' : 'Standard Cipher'}</span>
                    </div>
                 </div>
              </div>

              <div className="glass-panel p-8 rounded-[40px] border-primary-container/10 flex flex-col justify-between">
                 <div>
                    <h3 className="text-xs font-black text-primary-container uppercase tracking-[0.2em] mb-6 flex items-center gap-2">
                       <Shield size={14} /> Security Actions
                    </h3>
                    <p className="text-[10px] text-text-muted leading-relaxed mb-8 uppercase tracking-wider font-medium">
                       Manage your operational session and security credentials. Termination will disconnect all active uplinks.
                    </p>
                 </div>

                 <button 
                    onClick={handleLogout}
                    className="w-full py-5 rounded-2xl bg-error/10 border border-error/20 text-error font-black text-xs uppercase tracking-[0.2em] hover:bg-error hover:text-black transition-all flex items-center justify-center gap-2 group"
                 >
                    Terminate Session
                    <LogOut size={16} className="group-hover:translate-x-1 transition-transform" />
                 </button>
              </div>
           </div>

        </div>
      </main>

      {/* Footer Info */}
      <footer className="p-8 text-center opacity-30">
         <div className="flex items-center justify-center gap-6">
            <div className="flex items-center gap-2 text-[8px] font-bold uppercase tracking-widest text-primary-container">
               <Shield size={10} /> Encrypted Node
            </div>
            <div className="flex items-center gap-2 text-[8px] font-bold uppercase tracking-widest text-primary-container">
               <Zap size={10} /> Neural Core V2
            </div>
         </div>
      </footer>
    </div>
  )
}
