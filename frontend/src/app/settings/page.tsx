// frontend/src/app/settings/page.tsx
'use client'
import { useState } from 'react'
import { Save, Camera, Database, Cpu, Shield, Bell, HardDrive, Sliders } from 'lucide-react'

export default function Settings() {
  const [config, setConfig] = useState({
    confidence_threshold: 0.25,
    line_ratio: 0.5,
    storage_backend: 'minio',
    device: 'cpu',
    enable_notifications: true,
    camera_url: 'rtsp://admin:password@192.168.1.100:554/stream1'
  })

  const handleSave = () => {
    // In a real app, this would POST to /api/settings
    alert('Settings saved successfully (Simulation)')
  }

  return (
    <div className="animate-fade-in max-w-4xl space-y-8">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">System Settings</h1>
          <p className="text-text-muted mt-1">Configure AI models, storage backends, and interface preferences</p>
        </div>
        <button onClick={handleSave} className="btn-primary flex items-center gap-2 px-6 py-2.5">
          <Save size={18} />
          Save Changes
        </button>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* AI Analysis Section */}
        <section className="glass p-6 space-y-6">
          <div className="flex items-center gap-3 border-b border-sidebar-border pb-4">
            <Cpu className="text-primary" size={20} />
            <h2 className="text-lg font-bold">Inference Engine</h2>
          </div>
          
          <div className="space-y-4">
            <div>
              <label className="text-xs font-bold uppercase tracking-wider text-text-muted mb-2 block">
                Confidence Threshold ({(config.confidence_threshold * 100).toFixed(0)}%)
              </label>
              <input 
                type="range" min="0.1" max="0.9" step="0.05"
                value={config.confidence_threshold}
                onChange={(e) => setConfig({...config, confidence_threshold: parseFloat(e.target.value)})}
                className="w-full accent-primary"
              />
            </div>

            <div>
              <label className="text-xs font-bold uppercase tracking-wider text-text-muted mb-2 block">
                Detection Device
              </label>
              <select 
                value={config.device}
                onChange={(e) => setConfig({...config, device: e.target.value})}
                className="w-full glass bg-sidebar-bg border-none p-3 text-sm rounded-xl outline-none focus:ring-1 focus:ring-primary"
              >
                <option value="cpu">CPU (Standard)</option>
                <option value="cuda">NVIDIA GPU (CUDA 12.x)</option>
                <option value="mps">Apple Silicon (MPS)</option>
              </select>
            </div>
          </div>
        </section>

        {/* Storage Section */}
        <section className="glass p-6 space-y-6">
          <div className="flex items-center gap-3 border-b border-sidebar-border pb-4">
            <HardDrive className="text-secondary" size={20} />
            <h2 className="text-lg font-bold">Evidence Storage</h2>
          </div>
          
          <div className="space-y-4">
            <div>
              <label className="text-xs font-bold uppercase tracking-wider text-text-muted mb-2 block">
                Backend Provider
              </label>
              <div className="grid grid-cols-2 gap-2">
                <button 
                  onClick={() => setConfig({...config, storage_backend: 'local'})}
                  className={`p-3 rounded-xl border text-sm font-medium transition-all ${config.storage_backend === 'local' ? 'bg-primary/10 border-primary text-primary' : 'border-card-border text-text-muted hover:border-white/20'}`}
                >
                  Local System
                </button>
                <button 
                  onClick={() => setConfig({...config, storage_backend: 'minio'})}
                  className={`p-3 rounded-xl border text-sm font-medium transition-all ${config.storage_backend === 'minio' ? 'bg-primary/10 border-primary text-primary' : 'border-card-border text-text-muted hover:border-white/20'}`}
                >
                  MinIO (S3)
                </button>
              </div>
            </div>

            <div className="flex items-center justify-between p-3 glass rounded-xl bg-sidebar-bg/50">
              <div>
                <div className="text-sm font-bold">Auto-Cleanup</div>
                <div className="text-[10px] text-text-muted">Purge assets older than 30 days</div>
              </div>
              <input type="checkbox" defaultChecked className="w-5 h-5 accent-primary" />
            </div>
          </div>
        </section>

        {/* Camera Section */}
        <section className="glass p-6 space-y-6 md:col-span-2">
          <div className="flex items-center gap-3 border-b border-sidebar-border pb-4">
            <Camera className="text-primary" size={20} />
            <h2 className="text-lg font-bold">Primary Camera Source</h2>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 items-end">
            <div className="md:col-span-2">
              <label className="text-xs font-bold uppercase tracking-wider text-text-muted mb-2 block">
                RTSP / HTTP Stream URL
              </label>
              <input 
                type="text" 
                value={config.camera_url}
                onChange={(e) => setConfig({...config, camera_url: e.target.value})}
                className="w-full glass bg-sidebar-bg border-none p-3 text-sm rounded-xl outline-none focus:ring-1 focus:ring-primary font-monospace"
              />
            </div>
            <button className="glass py-3 text-sm font-bold hover:bg-white/5 transition-colors border border-card-border">
              Test Connection
            </button>
          </div>
        </section>

        {/* Notifications Section */}
        <section className="glass p-6 space-y-4">
          <div className="flex items-center gap-3 border-b border-sidebar-border pb-4">
            <Bell className="text-warning" size={20} />
            <h2 className="text-lg font-bold">Incident Alerts</h2>
          </div>
          
          <div className="space-y-3">
            {[
              { label: 'High Severity Alerts', desc: 'Notify on critical safety violations' },
              { label: 'System Health Status', desc: 'Alert if camera goes offline' },
              { label: 'Daily Analytics Report', desc: 'Morning summary of traffic flow' }
            ].map((item, idx) => (
              <div key={idx} className="flex items-center justify-between py-2">
                <div>
                  <div className="text-sm font-medium">{item.label}</div>
                  <div className="text-[10px] text-text-muted">{item.desc}</div>
                </div>
                <div className="relative inline-flex items-center cursor-pointer">
                  <input type="checkbox" className="sr-only peer" defaultChecked={idx < 2} />
                  <div className="w-11 h-6 bg-sidebar-bg rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-primary"></div>
                </div>
              </div>
            ))}
          </div>
        </section>

        {/* Security Section */}
        <section className="glass p-6 space-y-4">
          <div className="flex items-center gap-3 border-b border-sidebar-border pb-4">
            <Shield className="text-success" size={20} />
            <h2 className="text-lg font-bold">Access Control</h2>
          </div>
          <div className="space-y-4">
            <div className="p-4 rounded-xl bg-sidebar-bg/50 border border-card-border border-dashed text-center">
              <Shield size={24} className="mx-auto mb-2 text-text-muted opacity-50" />
              <div className="text-xs font-bold uppercase mb-1">RBAC Enabled</div>
              <div className="text-[10px] text-text-muted">Administrator: admin@traffic.local</div>
            </div>
            <button className="w-full text-xs font-bold py-2 hover:text-primary transition-colors uppercase tracking-widest">
              Manage API Keys
            </button>
          </div>
        </section>
      </div>
    </div>
  )
}
