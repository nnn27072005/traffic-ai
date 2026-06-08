// frontend/src/components/Sidebar.tsx
'use client'
import React from 'react'
import Link from 'next/link'
import { usePathname } from 'next/navigation'
import { 
  LayoutDashboard, History, BarChart3, Settings, 
  ShieldAlert, Activity, Cpu, Bell, Database, User, LogOut, HelpCircle, Video, ListTree
} from 'lucide-react'
import { useAuth } from '@/context/AuthContext'

function NavLink({ href, icon: Icon, label, active = false }: any) {
  return (
    <Link 
      href={href} 
      className={`flex items-center px-6 py-3 space-x-3 transition-all duration-200 group ${
        active 
          ? 'bg-primary-container/20 text-primary-container border-r-4 border-primary-container shadow-[inset_-10px_0_20px_rgba(0,240,255,0.1)]' 
          : 'text-primary-container/40 hover:bg-primary-container/10 hover:text-primary-container'
      }`}
    >
      <Icon size={18} className={active ? 'drop-shadow-[0_0_5px_rgba(0,240,255,0.5)]' : ''} />
      <span className="font-headline text-[13px] font-bold uppercase tracking-widest">{label}</span>
    </Link>
  )
}

export const Sidebar = () => {
  const pathname = usePathname()
  const { user, logout } = useAuth()

  if (pathname === '/login' || pathname === '/register') return null

  return (
    <nav className="bg-black/80 backdrop-blur-2xl text-primary-container font-mono text-sm tracking-tighter border-r border-primary-container/20 shadow-[4px_0_20px_rgba(0,240,255,0.05)] fixed left-0 top-0 h-screen w-64 flex flex-col z-40 hidden md:flex">
      <Link href="/profile" className="p-6 border-b border-primary-container/20 flex items-center space-x-4 hover:bg-primary-container/5 transition-colors group">
        <div className="w-10 h-10 rounded-full bg-surface flex items-center justify-center border border-primary-container/30 overflow-hidden group-hover:border-primary-container/60 transition-colors">
          {user?.avatar_url ? (
             <img src={user.avatar_url} alt="Profile" className="w-full h-full object-cover" />
          ) : (
             <User size={20} className="text-primary-container" />
          )}
        </div>
        <div>
          <div className="font-headline text-xs font-bold text-primary-container uppercase truncate max-w-[120px]">
            {user?.full_name || user?.username || 'GUEST_OPERATOR'}
          </div>
          <div className="font-headline text-[9px] text-primary-container/70 flex items-center tracking-widest uppercase">
            <span className="w-1.5 h-1.5 rounded-full bg-primary-container mr-2 shadow-[0_0_4px_#00f0ff] animate-pulse"></span>
            Active
          </div>
        </div>
      </Link>
      
      <div className="flex-1 py-4 flex flex-col space-y-1">
        <NavLink href="/" icon={LayoutDashboard} label="Dashboard" active={pathname === '/'} />
        <NavLink href="/analysis" icon={Video} label="Video Analysis" active={pathname === '/analysis'} />
        <NavLink href="/analytics" icon={BarChart3} label="Analytics" active={pathname === '/analytics'} />
        <NavLink href="/history" icon={ListTree} label="Incident Logs" active={pathname === '/history'} />
        <NavLink href="/settings" icon={Settings} label="Configurations" active={pathname === '/settings'} />
      </div>

      <div className="mt-auto border-t border-primary-container/20 py-4 flex flex-col space-y-1">
        <div className="px-6 py-2">
           <div className="text-[9px] font-bold text-primary-container/30 uppercase tracking-[0.2em] mb-4">Core Systems</div>
           <div className="space-y-3">
              <SubsystemStatus label="Inference" status="98.2%" />
              <SubsystemStatus label="Asset Uplink" status="Stable" />
           </div>
        </div>
        <div className="h-4" />
        <NavLink href="/support" icon={HelpCircle} label="Support" active={pathname === '/support'} />
        <div onClick={logout} className="cursor-pointer">
           <NavLink href="#" icon={LogOut} label="Logout" />
        </div>
      </div>
    </nav>
  )
}

function SubsystemStatus({ label, status }: any) {
  return (
    <div className="flex items-center justify-between opacity-60 hover:opacity-100 transition-opacity cursor-crosshair">
       <span className="text-[10px] font-bold uppercase tracking-wider text-primary-container/50">{label}</span>
       <span className="text-[10px] font-mono text-primary-container">{status}</span>
    </div>
  )
}
