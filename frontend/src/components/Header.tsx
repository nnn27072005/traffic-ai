// frontend/src/components/Header.tsx
'use client'
import React from 'react'
import { User, Bell, Search, Zap, Shield } from 'lucide-react'
import { useAuth } from '@/context/AuthContext'
import Link from 'next/link'
import { usePathname } from 'next/navigation'

export const Header = () => {
  const { user } = useAuth()
  const pathname = usePathname()

  if (pathname === '/login' || pathname === '/register') return null

  return (
    <header className="relative z-30 h-16 border-b border-primary-container/10 bg-black/20 backdrop-blur-md flex items-center justify-between px-8">
      {/* Search / Context info */}
      <div className="flex items-center gap-6">
        <div className="relative group hidden sm:block">
           <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-primary-container/40 group-focus-within:text-primary-container transition-colors" />
           <input 
             type="text" 
             placeholder="SEARCH_COORDINATES..." 
             className="bg-primary-container/5 border border-primary-container/10 rounded-xl py-2 pl-9 pr-4 text-[10px] font-mono text-primary-container focus:outline-none focus:border-primary-container/30 focus:ring-4 focus:ring-primary-container/5 transition-all placeholder:text-primary-container/20 w-64"
           />
        </div>
        <div className="flex items-center gap-4 text-[9px] font-bold uppercase tracking-widest text-primary-container/30">
           <div className="flex items-center gap-1.5"><Zap size={10} /> Latency: 12ms</div>
           <div className="flex items-center gap-1.5"><Shield size={10} /> Secure Uplink</div>
        </div>
      </div>

      {/* Right side actions */}
      <div className="flex items-center gap-4">
        <button className="p-2.5 rounded-xl bg-primary-container/5 border border-primary-container/10 text-primary-container/60 hover:text-primary-container hover:bg-primary-container/10 transition-all relative">
           <Bell size={18} />
           <span className="absolute top-2 right-2 w-1.5 h-1.5 bg-primary-container rounded-full shadow-[0_0_8px_#00f0ff] animate-pulse"></span>
        </button>

        <Link href="/profile" className="flex items-center gap-3 p-1.5 pl-4 rounded-2xl bg-primary-container/10 border border-primary-container/20 hover:border-primary-container/40 transition-all group">
           <div className="text-right">
              <p className="text-[10px] font-black text-primary-container uppercase tracking-tighter group-hover:tracking-widest transition-all">
                {user?.full_name || user?.username || 'GUEST_OPERATOR'}
              </p>
              <p className="text-[7px] font-bold text-primary-container/40 uppercase tracking-[0.2em]">View Profile</p>
           </div>
           <div className="w-9 h-9 rounded-xl bg-surface border border-primary-container/30 flex items-center justify-center overflow-hidden group-hover:scale-110 transition-all">
              {user?.avatar_url ? (
                 <img src={user.avatar_url} alt="Profile" className="w-full h-full object-cover" />
              ) : (
                 <User size={18} className="text-primary-container" />
              )}
           </div>
        </Link>
      </div>
    </header>
  )
}
