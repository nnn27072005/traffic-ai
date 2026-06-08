// frontend/src/app/support/page.tsx
'use client'
import React from 'react'
import { HelpCircle, Mail, MessageSquare, Book, Shield, ExternalLink } from 'lucide-react'

export default function Support() {
  return (
    <div className="animate-fade-in space-y-8 pb-12">
      <div className="relative h-[200px] rounded-[32px] overflow-hidden glass-panel flex items-center px-12">
        <div className="scan-line"></div>
        <div className="hud-crosshair-tl"></div>
        <div className="hud-crosshair-tr"></div>
        <div>
          <h1 className="text-4xl font-black tracking-tighter text-primary-container">System Support <span className="text-text-main">& Assistance</span></h1>
          <p className="text-text-muted mt-2 max-w-lg text-sm font-medium">
            Access technical documentation, contact engineering, or browse the operational knowledge base.
          </p>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <SupportCard 
          icon={Book} 
          title="Documentation" 
          desc="Complete system manual, API references, and deployment guides."
          link="Browse Docs"
        />
        <SupportCard 
          icon={MessageSquare} 
          title="Live Support" 
          desc="Chat with a technical specialist for immediate system assistance."
          link="Start Chat"
        />
        <SupportCard 
          icon={Mail} 
          title="Email Desk" 
          desc="Submit technical tickets or request hardware maintenance."
          link="Open Ticket"
        />
      </div>

      <div className="glass-panel p-8 rounded-[32px]">
        <h2 className="text-xl font-bold mb-6 flex items-center gap-3">
          <Shield size={24} className="text-primary-container" />
          System Health & Safety
        </h2>
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
           <div className="space-y-4">
              <p className="text-sm text-text-muted leading-relaxed">
                The TRAFFIC_AI_CORE is designed for 99.9% uptime. If you encounter any anomalies in the Neural Engine or Asset Storage subsystems, please run a diagnostic scan from the Configurations page.
              </p>
              <button className="btn-primary">Run Global Diagnostic</button>
           </div>
           <div className="space-y-3">
              <FaqItem q="How do I add a new camera node?" a="Go to Configurations > Camera Streaming and click 'Add Node'." />
              <FaqItem q="What is the data retention policy?" a="Standard retention is 30 days, configurable in Storage settings." />
           </div>
        </div>
      </div>
    </div>
  )
}

function SupportCard({ icon: Icon, title, desc, link }: any) {
  return (
    <div className="glass-panel p-8 rounded-[32px] group hover:border-primary-container/40 transition-all">
       <div className="w-12 h-12 rounded-2xl bg-surface flex items-center justify-center text-primary-container mb-6 group-hover:scale-110 transition-transform">
          <Icon size={24} />
       </div>
       <h3 className="text-lg font-bold mb-2">{title}</h3>
       <p className="text-xs text-text-muted mb-6 leading-relaxed">{desc}</p>
       <button className="text-[10px] font-bold text-primary-container uppercase tracking-widest flex items-center gap-2">
          {link} <ExternalLink size={12} />
       </button>
    </div>
  )
}

function FaqItem({ q, a }: any) {
  return (
    <div className="p-4 rounded-2xl bg-white/5 border border-white/5">
       <div className="text-sm font-bold mb-1">{q}</div>
       <div className="text-xs text-text-muted">{a}</div>
    </div>
  )
}
