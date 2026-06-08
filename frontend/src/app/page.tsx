// frontend/src/app/page.tsx
'use client'
import { useState, useEffect } from 'react'
import { 
  Activity, ShieldAlert, Car, Map, TrendingUp, AlertCircle, 
  ChevronRight, ArrowUpRight, Zap, Globe, Cpu, Search, Bell, Settings, User, Router, Gauge, Info, ZoomIn, Maximize, List
} from 'lucide-react'
import Link from 'next/link'
import { VideoPlayer } from '@/components/VideoPlayer'
import { useAuth } from '@/context/AuthContext'

export default function Dashboard() {
  const { user } = useAuth()
  const [stats, setStats] = useState({
    total_vehicles: 0,
    active_violations: 0,
    peak_flow: 0,
    system_latency: 0,
    active_nodes: 142
  })
  const [activeSession, setActiveSession] = useState<number | null>(1)
  const [recentViolations, setRecentViolations] = useState([])

  useEffect(() => {
    const fetchData = async () => {
      try {
        const statsRes = await fetch('http://localhost:8000/api/stats/summary')
        const statsData = await statsRes.json()
        setStats(prev => ({
          ...prev,
          total_vehicles: statsData.total_vehicles || 0,
          active_violations: statsData.total_violations || 0,
          system_latency: 124.5, // fps simulation
        }))

        const violationsRes = await fetch('http://localhost:8000/api/violations/?limit=5')
        const vData = await violationsRes.json()
        setRecentViolations(vData)
      } catch (e) {
        console.error("Dashboard data fetch failed", e)
      }
    }
    fetchData()
    const interval = setInterval(fetchData, 3000)
    return () => clearInterval(interval)
  }, [])

  return (
    <div className="flex-1 flex flex-col relative z-10 animate-fade-in">
      {/* TopAppBar */}
      <header className="bg-black/60 backdrop-blur-xl font-headline tracking-wider uppercase border-b border-primary-container/20 shadow-[0_4px_20px_rgba(0,240,255,0.1)] flex justify-between items-center w-full px-6 h-16 shrink-0">
        <div className="flex items-center">
          <h1 className="text-2xl font-black text-primary-container drop-shadow-[0_0_8px_rgba(0,240,255,0.6)]">TRAFFIC_AI_CORE</h1>
        </div>
        
        <div className="flex-1 max-w-md mx-8 hidden md:block">
          <div className="relative w-full h-8 flex items-center glass-panel rounded border-b border-primary-container">
            <input 
              className="w-full bg-transparent border-none text-primary-container font-mono text-sm focus:ring-0 placeholder:text-primary-container/30 px-3" 
              placeholder="QUERY SYSTEM..." 
              type="text"
            />
            <Search size={16} className="text-primary-container/60 absolute right-2" />
          </div>
        </div>

        <div className="flex items-center space-x-4">
          <Link href="/support" className="text-primary-container/60 hover:text-primary-container hover:bg-primary-container/10 p-2 rounded transition-all duration-300">
            <Bell size={20} />
          </Link>
          <Link href="/settings" className="text-primary-container/60 hover:text-primary-container hover:bg-primary-container/10 p-2 rounded transition-all duration-300">
            <Settings size={20} />
          </Link>
          <div className="w-8 h-8 rounded-full bg-surface flex items-center justify-center border border-primary-container/30 overflow-hidden">
            {user?.avatar_url ? (
               <img src={user.avatar_url} alt="Profile" className="w-full h-full object-cover" />
            ) : (
               <User size={18} className="text-primary-container" />
            )}
          </div>
        </div>
      </header>

      {/* Dashboard Layout */}
      <main className="flex-1 overflow-y-auto p-6 md:p-8 flex flex-col space-y-6">
        
        {/* Top KPI Metrics Row */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 shrink-0">
          <KPIBlock 
            label="Active Nodes" 
            value={stats.active_nodes} 
            total={150} 
            icon={<Globe size={18} />} 
            progress={94} 
          />
          <KPIBlock 
            label="Neural Engine FPS" 
            value={stats.system_latency} 
            unit="fps" 
            icon={<Zap size={18} />} 
            active 
          />
          <KPIBlock 
            label="Daily Incidents" 
            value={stats.active_violations} 
            trend="+12%" 
            icon={<ShieldAlert size={18} />} 
            progress={45} 
            isError 
          />
        </div>

        {/* Main Workspace: Video Feed & Incident Log */}
        <div className="flex-1 grid grid-cols-1 lg:grid-cols-12 gap-6 min-h-0">
          
          {/* Live Surveillance Feed Widget */}
          <div className="lg:col-span-8 glass-panel rounded flex flex-col relative overflow-hidden h-[500px] lg:h-auto">
            <div className="p-3 border-b border-primary-container/20 flex justify-between items-center bg-black/40 shrink-0">
              <div className="flex items-center space-x-2">
                <span className="w-2 h-2 rounded-full bg-error animate-pulse shadow-[0_0_6px_#ffb4ab]"></span>
                <span className="font-headline text-xs font-bold text-primary-container uppercase tracking-wider">Live Feed: Node_Alpha_Junction</span>
              </div>
              <div className="flex space-x-2 text-text-muted">
                <ZoomIn size={18} className="cursor-pointer hover:text-primary-container transition-colors" />
                <Maximize size={18} className="cursor-pointer hover:text-primary-container transition-colors" />
              </div>
            </div>
            
            <div className="flex-1 relative bg-black group overflow-hidden">
              <VideoPlayer sessionId={activeSession || 1} />
              
              {/* HUD Elements overlaying the video */}
              <div className="absolute inset-0 pointer-events-none p-4">
                 {/* Crosshairs */}
                 <div className="hud-crosshair-tl"></div>
                 <div className="hud-crosshair-tr"></div>
                 <div className="hud-crosshair-bl"></div>
                 <div className="hud-crosshair-br"></div>
                 <div className="scan-line"></div>

                 {/* Simulated Bounding Box */}
                 <div className="absolute top-[30%] left-[45%] w-[120px] h-[80px] border border-primary-container shadow-[0_0_4px_#00f0ff] bg-primary-container/5 flex flex-col justify-between p-1">
                    <div className="text-[9px] font-mono text-primary-container bg-black/60 self-start px-1">OBJ: MTRCYC [0.98]</div>
                 </div>

                 {/* Alert Box */}
                 <div className="absolute top-[50%] left-[20%] w-[100px] h-[70px] border border-error shadow-[0_0_4px_#ffb4ab] bg-error/5 flex flex-col justify-between p-1">
                    <div className="text-[9px] font-mono text-error bg-black/60 self-start px-1 animate-pulse">ERR: SPD_LMT [0.95]</div>
                 </div>
              </div>

              {/* Telemetry overlay bottom */}
              <div className="absolute bottom-0 left-0 right-0 p-3 flex justify-between font-mono text-[10px] text-primary-container bg-gradient-to-t from-black/90 to-transparent">
                <div className="flex space-x-4">
                  <span>REC: 00:14:59:22</span>
                  <span>FMT: 4K_RAW</span>
                </div>
                <div className="flex space-x-4">
                  <span>LAT: 34.0522 N</span>
                  <span>LNG: 118.2437 W</span>
                </div>
              </div>
            </div>
          </div>

          {/* Incident Pulse Feed */}
          <div className="lg:col-span-4 glass-panel rounded flex flex-col h-[500px] lg:h-auto overflow-hidden">
            <div className="p-3 border-b border-primary-container/20 flex justify-between items-center bg-black/40 shrink-0">
              <span className="font-headline text-xs font-bold text-primary-container uppercase tracking-wider">Incident Pulse</span>
              <Activity size={18} className="text-text-muted cursor-pointer hover:text-primary-container" />
            </div>
            
            <div className="flex-1 overflow-y-auto p-2 space-y-2">
              {recentViolations.map((v: any, idx: number) => (
                <div key={v.id} className={`bg-black/40 border-l-2 ${v.violation_type === 'WithoutHelmet' ? 'border-secondary' : 'border-error'} p-3 hover:bg-surface transition-colors group relative cursor-pointer`}>
                  <div className="flex justify-between items-start mb-1 relative z-10">
                    <span className={`font-mono text-[11px] ${v.violation_type === 'WithoutHelmet' ? 'text-secondary' : 'text-error'}`}>
                      {v.violation_type.toUpperCase()}
                    </span>
                    <span className="font-mono text-[9px] text-text-muted">T-{idx}:0{idx+1}</span>
                  </div>
                  <div className="text-xs text-text-main mb-2 relative z-10">
                    Detection on Track #{v.track_id} in Sector Alpha.
                  </div>
                  <div className="flex space-x-2 relative z-10">
                    <span className="px-1.5 py-0.5 bg-white/5 text-text-muted font-mono text-[9px] rounded border border-white/10">ID:{v.id}</span>
                    <span className="px-1.5 py-0.5 bg-white/5 text-text-muted font-mono text-[9px] rounded border border-white/10">CAM_0{v.track_id % 4 + 1}</span>
                  </div>
                </div>
              ))}
              
              {recentViolations.length === 0 && (
                <div className="p-8 text-center text-text-muted text-xs opacity-50 font-mono">
                   -- WAITING FOR DATA UPLINK --
                </div>
              )}
            </div>

            <div className="p-3 border-t border-primary-container/20 shrink-0 bg-black/60 flex justify-center">
              <Link href="/history" className="text-[10px] font-bold text-primary-container/70 hover:text-primary-container uppercase tracking-widest flex items-center gap-2 transition-all">
                <span>View All System Logs</span>
                <ChevronRight size={14} />
              </Link>
            </div>
          </div>

        </div>
      </main>
    </div>
  )
}

function KPIBlock({ label, value, total, unit, trend, icon, progress, active, isError }: any) {
  return (
    <div className={`${active ? 'glass-panel-active' : 'glass-panel'} p-5 relative rounded overflow-hidden`}>
      <div className="scan-line"></div>
      <div className="hud-crosshair-tl"></div>
      <div className="hud-crosshair-tr"></div>
      {active && <><div className="hud-crosshair-bl"></div><div className="hud-crosshair-br"></div></>}
      
      <div className="flex justify-between items-start mb-3">
        <span className="font-headline text-[10px] font-bold text-text-muted uppercase tracking-widest">{label}</span>
        <div className={`opacity-50 ${isError ? 'text-error' : 'text-primary-container'}`}>{icon}</div>
      </div>
      
      <div className="flex items-end space-x-2">
        <span className={`text-4xl font-bold ${isError ? 'text-error' : 'text-primary-container'}`}>{value}</span>
        {total && <span className="font-mono text-sm text-text-muted opacity-60 mb-1">/ {total}</span>}
        {unit && <span className="font-mono text-sm text-text-muted opacity-60 mb-1">{unit}</span>}
        {trend && <span className={`font-mono text-xs mb-1 ${isError ? 'text-error/80' : 'text-success'}`}>{trend}</span>}
      </div>
      
      {progress !== undefined && (
        <div className="w-full bg-white/5 h-1 mt-4">
          <div className={`${isError ? 'bg-error' : 'bg-primary-container'} h-full shadow-[0_0_8px_currentColor] transition-all duration-1000`} style={{ width: `${progress}%` }}></div>
        </div>
      )}
      
      {active && (
        <div className="w-full flex space-x-[2px] mt-4 h-2">
           {[1, 2, 3, 4].map(i => (
             <div key={i} className="bg-primary-container h-full flex-1 shadow-[0_0_4px_#00f0ff]"></div>
           ))}
           <div className="bg-primary-container h-full flex-1 opacity-20"></div>
        </div>
      )}
    </div>
  )
}
