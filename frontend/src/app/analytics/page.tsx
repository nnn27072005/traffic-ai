// frontend/src/app/analytics/page.tsx
'use client'
import { useState, useEffect } from 'react'
import { 
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, 
  ResponsiveContainer, BarChart, Bar, Cell, PieChart, Pie 
} from 'recharts'
import { 
  BarChart3, TrendingUp, Users, AlertTriangle, Clock, 
  Map as MapIcon, Share2, Download, Filter, Target, Activity
} from 'lucide-react'

export default function Analytics() {
  const [data, setData] = useState([
    { time: '08:00', flow: 120, violations: 5, safety: 95 },
    { time: '09:00', flow: 150, violations: 8, safety: 92 },
    { time: '10:00', flow: 200, violations: 12, safety: 88 },
    { time: '11:00', flow: 180, violations: 7, safety: 94 },
    { time: '12:00', flow: 160, violations: 4, safety: 97 },
    { time: '13:00', flow: 140, violations: 6, safety: 95 },
    { time: '14:00', flow: 170, violations: 9, safety: 91 },
  ])

  const typeData = [
    { name: 'Helmet', value: 400, color: 'var(--primary)' },
    { name: 'Wrong Way', value: 300, color: 'var(--secondary)' },
    { name: 'Speeding', value: 300, color: 'var(--danger)' },
    { name: 'Red Light', value: 200, color: 'var(--warning)' },
  ]

  return (
    <div className="animate-fade-in space-y-8 pb-12">
      {/* Hero Visualization */}
      <div className="relative h-[280px] rounded-[32px] overflow-hidden group">
        <img 
          src="/traffic_heatmap_viz.png" 
          alt="Traffic Heatmap" 
          className="absolute inset-0 w-full h-full object-cover opacity-60 transition-transform duration-1000 group-hover:scale-105"
        />
        <div className="absolute inset-0 bg-gradient-to-t from-background via-background/20 to-transparent" />
        <div className="absolute bottom-0 left-0 p-10">
          <div className="flex items-center gap-3 mb-3">
             <span className="badge badge-success">
                <Activity size={12} className="animate-pulse" />
                Live Telemetry Active
             </span>
             <span className="text-[10px] font-black uppercase tracking-widest text-text-muted">Sector: Downtown Grid-04</span>
          </div>
          <h1 className="text-4xl font-black tracking-tighter">Predictive <span className="text-primary">Flow Analytics</span></h1>
          <p className="text-text-muted mt-2 max-w-md text-sm font-medium">
            Advanced spatial mapping of vehicle movement patterns and safety violation hotspots.
          </p>
        </div>
        <div className="absolute top-10 right-10 flex gap-3">
           <button className="glass p-3 hover:text-primary transition-colors">
              <Share2 size={20} />
           </button>
           <button className="glass p-3 hover:text-primary transition-colors">
              <Download size={20} />
           </button>
        </div>
      </div>

      {/* High-Level Insights */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <InsightCard 
          icon={TrendingUp} 
          label="Flow Intensity Index" 
          value="162.8" 
          trend="+12.5%" 
          color="text-primary" 
          desc="Vehicles per hour across all zones"
        />
        <InsightCard 
          icon={Target} 
          label="Precision Accuracy" 
          value="98.4%" 
          trend="+0.2%" 
          color="text-secondary" 
          desc="AI Model detection confidence"
        />
        <InsightCard 
          icon={AlertTriangle} 
          label="Incident Velocity" 
          value="4.2" 
          trend="-2.1%" 
          color="text-accent" 
          desc="Average safety violations per hour"
        />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Main Volumetric Chart */}
        <div className="lg:col-span-2 glass p-8 space-y-8">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="text-xl font-bold flex items-center gap-3">
                <BarChart3 size={24} className="text-primary" />
                Temporal Flow Distribution
              </h2>
              <p className="text-xs text-text-muted mt-1 uppercase tracking-widest font-bold">Volume vs. Compliance (24H)</p>
            </div>
            <div className="flex gap-2">
               <button className="px-4 py-2 rounded-xl bg-primary/10 text-primary text-[10px] font-bold uppercase tracking-widest">Real-time</button>
               <button className="px-4 py-2 rounded-xl hover:bg-white/5 text-text-muted text-[10px] font-bold uppercase tracking-widest transition-colors">Historical</button>
            </div>
          </div>
          <div className="h-[340px]">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={data}>
                <defs>
                  <linearGradient id="colorFlow" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="var(--primary)" stopOpacity={0.3}/>
                    <stop offset="95%" stopColor="var(--primary)" stopOpacity={0}/>
                  </linearGradient>
                  <linearGradient id="colorSafety" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="var(--secondary)" stopOpacity={0.3}/>
                    <stop offset="95%" stopColor="var(--secondary)" stopOpacity={0}/>
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="rgba(255,255,255,0.05)" />
                <XAxis dataKey="time" axisLine={false} tickLine={false} tick={{fill: 'var(--text-muted)', fontSize: 10}} />
                <YAxis axisLine={false} tickLine={false} tick={{fill: 'var(--text-muted)', fontSize: 10}} />
                <Tooltip 
                  contentStyle={{ background: 'var(--sidebar-bg)', borderColor: 'var(--card-border)', borderRadius: '16px', boxShadow: '0 20px 40px rgba(0,0,0,0.5)' }}
                  itemStyle={{ fontSize: '12px', fontWeight: 700 }}
                  cursor={{ stroke: 'var(--primary)', strokeWidth: 1, strokeDasharray: '4 4' }}
                />
                <Area type="monotone" dataKey="flow" stroke="var(--primary)" strokeWidth={3} fillOpacity={1} fill="url(#colorFlow)" />
                <Area type="monotone" dataKey="violations" stroke="var(--danger)" strokeWidth={2} fill="transparent" />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Categorical Breakdown */}
        <div className="glass p-8 space-y-8">
           <div>
              <h2 className="text-xl font-bold flex items-center gap-3">
                <Filter size={24} className="text-secondary" />
                Type Classification
              </h2>
              <p className="text-xs text-text-muted mt-1 uppercase tracking-widest font-bold">Violation Source Analysis</p>
           </div>
           <div className="h-[280px]">
             <ResponsiveContainer width="100%" height="100%">
               <BarChart data={typeData} layout="vertical" margin={{ left: -20 }}>
                 <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="rgba(255,255,255,0.05)" />
                 <XAxis type="number" hide />
                 <YAxis dataKey="name" type="category" axisLine={false} tickLine={false} tick={{fill: 'white', fontSize: 11, fontWeight: 700}} width={80} />
                 <Tooltip 
                    cursor={{fill: 'rgba(255,255,255,0.05)'}}
                    contentStyle={{ background: 'var(--sidebar-bg)', borderColor: 'var(--card-border)', borderRadius: '16px' }}
                 />
                 <Bar dataKey="value" radius={[0, 8, 8, 0]} barSize={24}>
                   {typeData.map((entry, index) => (
                     <Cell key={`cell-${index}`} fill={entry.color} />
                   ))}
                 </Bar>
               </BarChart>
             </ResponsiveContainer>
           </div>
           <div className="space-y-4">
              <div className="text-[10px] font-black uppercase tracking-[0.2em] text-text-muted opacity-50">Top Detections</div>
              {typeData.map((item, idx) => (
                 <div key={idx} className="flex items-center justify-between">
                    <span className="text-xs font-bold">{item.name}</span>
                    <span className="text-xs font-mono font-bold text-primary">{item.value}</span>
                 </div>
              ))}
           </div>
        </div>
      </div>

      {/* Deep Intelligence Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          <StrategicCard label="Helmet Compliance" value="88.4%" trend="+1.2%" desc="Avg compliance in monitored zones" />
          <StrategicCard label="System Inference" value="142ms" trend="-10ms" desc="Average AI processing latency" />
          <StrategicCard label="False Positive" value="0.42%" trend="Stable" desc="Model validation error rate" />
          <StrategicCard label="Network Uptime" value="99.98%" trend="+0.01%" desc="Edge-node connectivity status" />
      </div>
    </div>
  )
}

function InsightCard({ icon: Icon, label, value, trend, color, desc }: any) {
  return (
    <div className="glass p-8 flex flex-col gap-6 group hover:border-primary/30">
      <div className="flex items-center justify-between">
        <div className={`w-14 h-14 rounded-2xl bg-sidebar-bg flex items-center justify-center ${color} glow-${color.split('-')[1]}`}>
          <Icon size={28} />
        </div>
        <div className="text-right">
           <div className={`text-xs font-bold ${trend.startsWith('+') ? 'text-success' : 'text-accent'}`}>{trend}</div>
           <div className="text-[9px] font-black uppercase tracking-widest text-text-muted">vs Prev Period</div>
        </div>
      </div>
      <div>
        <div className="text-[10px] font-black uppercase tracking-[0.2em] text-text-muted mb-2">{label}</div>
        <div className="text-3xl font-black tracking-tighter mb-1 font-mono">{value}</div>
        <p className="text-[10px] font-medium text-text-muted leading-relaxed">{desc}</p>
      </div>
    </div>
  )
}

function StrategicCard({ label, value, trend, desc }: any) {
  return (
    <div className="glass p-6 group cursor-help">
       <div className="text-[9px] font-black uppercase tracking-[0.2em] text-text-muted mb-4 group-hover:text-primary transition-colors">{label}</div>
       <div className="flex items-baseline gap-2 mb-2">
          <div className="text-2xl font-black font-mono tracking-tighter">{value}</div>
          <div className={`text-[10px] font-bold ${trend.startsWith('+') ? 'text-success' : 'text-accent'}`}>{trend}</div>
       </div>
       <p className="text-[9px] font-medium text-text-muted opacity-80 leading-relaxed">{desc}</p>
    </div>
  )
}
