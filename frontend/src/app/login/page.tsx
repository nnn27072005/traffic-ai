// frontend/src/app/login/page.tsx
'use client'
import React, { useState, useEffect } from 'react'
import { useSearchParams } from 'next/navigation'
import { useAuth } from '@/context/AuthContext'
import { Shield, Lock, User, Globe, ArrowRight, Zap, Info } from 'lucide-react'
import Link from 'next/link'
import { GoogleLogin, GoogleOAuthProvider } from '@react-oauth/google'

export default function Login() {
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState('')
  const { login } = useAuth()
  const searchParams = useSearchParams()
  const [showSuccess, setShowSuccess] = useState(false)

  useEffect(() => {
    if (searchParams.get('registered')) {
      setShowSuccess(true)
    }
  }, [searchParams])

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setError('')
    try {
      const formData = new FormData()
      formData.append('username', username)
      formData.append('password', password)

      const res = await fetch('http://localhost:8000/api/auth/login', {
        method: 'POST',
        body: formData,
      })

      if (!res.ok) {
        const data = await res.json()
        let message = 'Invalid credentials'
        if (typeof data.detail === 'string') {
          message = data.detail
        } else if (Array.isArray(data.detail)) {
          message = data.detail.map((e: any) => e.msg).join(', ')
        }
        throw new Error(message)
      }
      
      const data = await res.json()
      login(data.access_token, data.user)
    } catch (err: any) {
      setError(err.message)
    }
  }

  const handleGoogleSuccess = async (response: any) => {
     try {
       const res = await fetch('http://localhost:8000/api/auth/google', {
         method: 'POST',
         headers: { 'Content-Type': 'application/json' },
         body: JSON.stringify({ token: response.credential }),
       })
       if (!res.ok) {
         const data = await res.json()
         let message = 'Google authentication failed'
         if (typeof data.detail === 'string') {
           message = data.detail
         } else if (Array.isArray(data.detail)) {
           message = data.detail.map((e: any) => e.msg).join(', ')
         }
         throw new Error(message)
       }
       const data = await res.json()
       login(data.access_token, data.user)
     } catch (err: any) {
       setError(err.message)
     }
  }

  return (
    <GoogleOAuthProvider clientId={process.env.NEXT_PUBLIC_GOOGLE_CLIENT_ID || ''}>
      <div className="min-h-screen flex items-center justify-center bg-background relative overflow-hidden p-6">
        {/* Background Grid & HUD Elements */}
        <div className="absolute inset-0 grid-bg opacity-20"></div>
        <div className="absolute top-20 left-20 w-64 h-64 bg-primary-container/5 rounded-full blur-[100px]"></div>
        <div className="absolute bottom-20 right-20 w-96 h-96 bg-primary-container/5 rounded-full blur-[120px]"></div>

        <div className="max-w-md w-full relative z-10">
          <div className="glass-panel p-10 rounded-[48px] border-primary-container/20 shadow-[0_20px_50px_rgba(0,0,0,0.5)]">
            <div className="scan-line"></div>
            <div className="hud-crosshair-tl"></div>
            <div className="hud-crosshair-tr"></div>
            <div className="hud-crosshair-bl"></div>
            <div className="hud-crosshair-br"></div>

            <div className="text-center mb-10">
              <div className="w-16 h-16 rounded-3xl bg-surface flex items-center justify-center mx-auto mb-6 border border-primary-container/30 shadow-[0_0_20px_rgba(0,240,255,0.15)] group hover:scale-110 transition-all cursor-pointer">
                <Shield size={32} className="text-primary-container" />
              </div>
              <h1 className="text-3xl font-black tracking-tighter text-primary-container uppercase">Security Clearance</h1>
              <p className="text-text-muted text-[10px] font-bold uppercase tracking-[0.3em] mt-2">Authorization Required for Uplink</p>
            </div>

            <form onSubmit={handleSubmit} className="space-y-5">
              <div className="space-y-1">
                 <label className="text-[9px] font-bold text-primary-container/60 uppercase tracking-widest ml-1">Identity UID</label>
                 <div className="relative group">
                    <User size={16} className="absolute left-4 top-1/2 -translate-y-1/2 text-primary-container/40 group-focus-within:text-primary-container transition-colors" />
                    <input 
                      type="text" 
                      required
                      value={username}
                      onChange={(e) => setUsername(e.target.value)}
                      placeholder="OPERATOR_ID" 
                      className="w-full bg-black/40 border border-primary-container/20 rounded-2xl py-4 pl-12 pr-4 text-sm font-mono text-primary-container focus:outline-none focus:border-primary-container/60 focus:ring-4 focus:ring-primary-container/5 transition-all placeholder:text-primary-container/20"
                    />
                 </div>
              </div>

              <div className="space-y-1">
                 <label className="text-[9px] font-bold text-primary-container/60 uppercase tracking-widest ml-1">Access Cipher</label>
                 <div className="relative group">
                    <Lock size={16} className="absolute left-4 top-1/2 -translate-y-1/2 text-primary-container/40 group-focus-within:text-primary-container transition-colors" />
                    <input 
                      type="password" 
                      required
                      value={password}
                      onChange={(e) => setPassword(e.target.value)}
                      placeholder="••••••••" 
                      className="w-full bg-black/40 border border-primary-container/20 rounded-2xl py-4 pl-12 pr-4 text-sm font-mono text-primary-container focus:outline-none focus:border-primary-container/60 focus:ring-4 focus:ring-primary-container/5 transition-all placeholder:text-primary-container/20"
                    />
                 </div>
              </div>

              {error && (
                <div className="bg-error/10 border border-error/20 rounded-xl p-3 flex items-center gap-3 text-error animate-shake">
                   <Info size={14} />
                   <span className="text-[10px] font-bold uppercase tracking-wider">{error}</span>
                </div>
              )}

              {showSuccess && (
                <div className="bg-success/10 border border-success/20 rounded-xl p-3 flex items-center gap-3 text-success animate-in fade-in slide-in-from-top-2">
                   <Zap size={14} />
                   <span className="text-[10px] font-bold uppercase tracking-wider">Asset Provisioned. Proceed with Uplink.</span>
                </div>
              )}

              <button 
                type="submit"
                className="w-full py-5 rounded-2xl bg-primary-container text-black font-black text-xs uppercase tracking-[0.2em] hover:scale-[1.02] active:scale-[0.98] transition-all shadow-[0_0_30px_rgba(0,240,255,0.2)] flex items-center justify-center gap-2 group"
              >
                Establish Uplink
                <ArrowRight size={16} className="group-hover:translate-x-1 transition-transform" />
              </button>
            </form>

            <div className="mt-8 relative py-4 flex items-center justify-center">
               <div className="absolute inset-0 flex items-center"><div className="w-full border-t border-primary-container/10"></div></div>
               <span className="relative z-10 bg-surface px-4 text-[9px] font-bold text-primary-container/30 uppercase tracking-[0.3em]">Alternate Protocol</span>
            </div>

            <div className="mt-4 flex justify-center">
               <div className="w-full scale-110">
                  <GoogleLogin 
                    onSuccess={handleGoogleSuccess} 
                    onError={() => setError('Google Sign-In Failed')}
                    theme="filled_black"
                    shape="pill"
                    text="continue_with"
                    width="350"
                  />
               </div>
            </div>

            <p className="text-center mt-10 text-[10px] text-text-muted font-medium">
               New Asset Deployment? <Link href="/register" className="text-primary-container hover:underline underline-offset-4 decoration-2">Register Identity</Link>
            </p>
          </div>
          
          <div className="mt-8 flex justify-center gap-6 opacity-30">
             <div className="flex items-center gap-2 text-[9px] font-bold uppercase tracking-widest text-primary-container">
                <Globe size={12} /> Encrypted Node
             </div>
             <div className="flex items-center gap-2 text-[9px] font-bold uppercase tracking-widest text-primary-container">
                <Zap size={12} /> Neural Core V2
             </div>
          </div>
        </div>
      </div>
    </GoogleOAuthProvider>
  )
}
