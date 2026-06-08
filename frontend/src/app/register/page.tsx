// frontend/src/app/register/page.tsx
'use client'
import React, { useState } from 'react'
import { useRouter } from 'next/navigation'
import { Shield, Lock, User, Mail, ArrowRight, Info, ChevronLeft } from 'lucide-react'
import Link from 'next/link'
import { GoogleLogin, GoogleOAuthProvider } from '@react-oauth/google'
import { useAuth } from '@/context/AuthContext'

export default function Register() {
  const [username, setUsername] = useState('')
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)
  const router = useRouter()

  const { login } = useAuth()

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setError('')
    setLoading(true)
    try {
      const res = await fetch('http://localhost:8000/api/auth/register', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username, email, password }),
      })

      if (!res.ok) {
        const data = await res.json()
        let message = 'Registration failed'
        if (typeof data.detail === 'string') {
          message = data.detail
        } else if (Array.isArray(data.detail)) {
          message = data.detail.map((e: any) => e.msg).join(', ')
        }
        throw new Error(message)
      }
      
      router.push('/login?registered=true')
    } catch (err: any) {
      setError(err.message)
    } finally {
      setLoading(false)
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
        let message = 'Google registration failed'
        if (typeof data.detail === 'string') {
          message = data.detail
        } else if (Array.isArray(data.detail)) {
          message = data.detail.map((e: any) => e.msg).join(', ')
        }
        throw new Error(message)
      }
      const data = await res.json()
      router.push('/login?registered=true')
    } catch (err: any) {
      setError(err.message)
    }
 }

  return (
    <GoogleOAuthProvider clientId={process.env.NEXT_PUBLIC_GOOGLE_CLIENT_ID || ''}>
    <div className="min-h-screen flex items-center justify-center bg-background relative overflow-hidden p-6">
      <div className="absolute inset-0 grid-bg opacity-20"></div>
      
      <div className="max-w-md w-full relative z-10">
        <div className="glass-panel p-10 rounded-[48px] border-primary-container/20">
          <div className="scan-line"></div>
          <div className="hud-crosshair-tl"></div>
          <div className="hud-crosshair-tr"></div>
          <div className="hud-crosshair-bl"></div>
          <div className="hud-crosshair-br"></div>

          <Link href="/login" className="flex items-center gap-2 text-[10px] font-bold text-primary-container/40 hover:text-primary-container uppercase tracking-widest mb-8 transition-colors group">
             <ChevronLeft size={14} className="group-hover:-translate-x-1 transition-transform" /> Back to Clearance
          </Link>

          <div className="text-center mb-10">
            <h1 className="text-3xl font-black tracking-tighter text-primary-container uppercase">New Asset Registration</h1>
            <p className="text-text-muted text-[10px] font-bold uppercase tracking-[0.3em] mt-2">Provisioning Operational Identity</p>
          </div>

          <form onSubmit={handleSubmit} className="space-y-5">
            <div className="space-y-1">
               <label className="text-[9px] font-bold text-primary-container/60 uppercase tracking-widest ml-1">Assigned Username</label>
               <div className="relative group">
                  <User size={16} className="absolute left-4 top-1/2 -translate-y-1/2 text-primary-container/40 group-focus-within:text-primary-container transition-colors" />
                  <input 
                    type="text" 
                    required
                    value={username}
                    onChange={(e) => setUsername(e.target.value)}
                    placeholder="OPERATOR_ALIAS" 
                    className="w-full bg-black/40 border border-primary-container/20 rounded-2xl py-4 pl-12 pr-4 text-sm font-mono text-primary-container focus:outline-none focus:border-primary-container/60 focus:ring-4 focus:ring-primary-container/5 transition-all placeholder:text-primary-container/20"
                  />
               </div>
            </div>

            <div className="space-y-1">
               <label className="text-[9px] font-bold text-primary-container/60 uppercase tracking-widest ml-1">Uplink Email</label>
               <div className="relative group">
                  <Mail size={16} className="absolute left-4 top-1/2 -translate-y-1/2 text-primary-container/40 group-focus-within:text-primary-container transition-colors" />
                  <input 
                    type="email" 
                    required
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    placeholder="OPERATOR@DOMAIN.CORE" 
                    className="w-full bg-black/40 border border-primary-container/20 rounded-2xl py-4 pl-12 pr-4 text-sm font-mono text-primary-container focus:outline-none focus:border-primary-container/60 focus:ring-4 focus:ring-primary-container/5 transition-all placeholder:text-primary-container/20"
                  />
               </div>
            </div>

            <div className="space-y-1">
               <label className="text-[9px] font-bold text-primary-container/60 uppercase tracking-widest ml-1">Security Cipher</label>
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

            <button 
              type="submit"
              disabled={loading}
              className="w-full py-5 rounded-2xl bg-primary-container text-black font-black text-xs uppercase tracking-[0.2em] hover:scale-[1.02] active:scale-[0.98] transition-all shadow-[0_0_30px_rgba(0,240,255,0.2)] flex items-center justify-center gap-2 group disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {loading ? 'Initializing...' : 'Register Asset'}
              {!loading && <ArrowRight size={16} className="group-hover:translate-x-1 transition-transform" />}
            </button>
            <div className="mt-8 relative py-4 flex items-center justify-center">
               <div className="absolute inset-0 flex items-center"><div className="w-full border-t border-primary-container/10"></div></div>
               <span className="relative z-10 bg-surface px-4 text-[9px] font-bold text-primary-container/30 uppercase tracking-[0.3em]">Alternate Protocol</span>
            </div>

            <div className="mt-4 flex justify-center">
               <div className="w-full scale-110">
                  <GoogleLogin 
                    onSuccess={handleGoogleSuccess} 
                    onError={() => setError('Google Sign-Up Failed')}
                    theme="filled_black"
                    shape="pill"
                    text="signup_with"
                    width="350"
                  />
               </div>
            </div>
          </form>

          <p className="text-center mt-10 text-[10px] text-text-muted font-medium">
             Already Provisioned? <Link href="/login" className="text-primary-container hover:underline underline-offset-4 decoration-2">Access Command</Link>
          </p>
        </div>
      </div>
    </div>
    </GoogleOAuthProvider>
  )
}
