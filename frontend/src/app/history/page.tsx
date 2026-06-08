// frontend/src/app/history/page.tsx
'use client'
import { useState, useEffect } from 'react'
import { Calendar, Search, Filter, ShieldAlert, Image as ImageIcon, ChevronRight, ChevronLeft } from 'lucide-react'

interface Violation {
  id: number
  track_id: number
  timestamp: string
  violation_type: string
  confidence: number
  image_path?: string | null
  asset_url?: string | null
}

export default function History() {
  const [violations, setViolations] = useState<Violation[]>([])
  const [loading, setLoading] = useState(true)
  const [searchTerm, setSearchTerm] = useState('')
  const [page, setPage] = useState(1)
  const limit = 12

  useEffect(() => {
    fetch(`http://localhost:8000/api/violations/?limit=${limit}&offset=${(page - 1) * limit}`)
      .then(res => res.json())
      .then(data => {
        setViolations(data)
        setLoading(false)
      })
      .catch(err => {
        console.error("Failed to fetch violations:", err)
        setLoading(false)
      })
  }, [page])

  const filteredViolations = violations.filter((v) => 
    v.track_id?.toString().includes(searchTerm) || 
    v.violation_type?.toLowerCase().includes(searchTerm.toLowerCase())
  )

  const getEvidenceUrl = (violation: Violation) => {
    if (violation.asset_url) return violation.asset_url
    if (violation.image_path) return `http://localhost:8000/api/assets/${violation.image_path}`
    return null
  }

  return (
    <div className="animate-fade-in space-y-8">
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-6">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Violation History</h1>
          <p className="text-text-muted mt-1">Audit log and evidence repository for all detected incidents</p>
        </div>

        <div className="flex items-center gap-3">
          <div className="glass px-4 py-2 flex items-center gap-3 focus-within:ring-2 focus-within:ring-primary/50 transition-all">
            <Search size={18} className="text-text-muted" />
            <input 
              type="text" 
              placeholder="Search ID or Type..." 
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="bg-transparent border-none text-white outline-none text-sm w-48 md:w-64" 
            />
          </div>
          <button className="glass p-2.5 text-text-muted hover:text-white transition-colors">
            <Filter size={20} />
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
        {loading ? (
          Array(8).fill(0).map((_, i) => (
            <div key={i} className="glass aspect-video animate-pulse rounded-2xl" />
          ))
        ) : filteredViolations.map((v) => {
          const evidenceUrl = getEvidenceUrl(v)
          return (
          <div key={v.id} className="glass group overflow-hidden hover:scale-[1.02] transition-all duration-300">
            <div className="relative aspect-video bg-sidebar-bg overflow-hidden">
              {evidenceUrl ? (
                <img 
                  src={evidenceUrl} 
                  alt="Violation Evidence"
                  className="w-full h-full object-cover group-hover:scale-110 transition-transform duration-500"
                />
              ) : (
                <div className="w-full h-full flex flex-col items-center justify-center text-text-muted gap-2">
                  <ImageIcon size={32} strokeWidth={1} />
                  <span className="text-[10px] uppercase tracking-widest font-bold">No Image Evidence</span>
                </div>
              )}
              <div className="absolute top-3 left-3">
                <span className={`badge ${v.violation_type === 'WithoutHelmet' ? 'badge-danger' : 'badge-warning'}`}>
                  {v.violation_type === 'WithoutHelmet' ? 'No Helmet' : v.violation_type}
                </span>
              </div>
              <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-transparent to-transparent opacity-0 group-hover:opacity-100 transition-opacity flex items-end p-4">
                <button className="w-full py-2 bg-white/10 hover:bg-white/20 backdrop-blur-md rounded-xl text-xs font-bold uppercase tracking-wider transition-colors">
                  View Full Evidence
                </button>
              </div>
            </div>
            
            <div className="p-4 space-y-3">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <div className="w-8 h-8 rounded-lg bg-sidebar-bg flex items-center justify-center text-xs font-bold text-primary">
                    #{v.id}
                  </div>
                  <div>
                    <div className="text-xs font-bold uppercase tracking-tight">Track ID: {v.track_id}</div>
                    <div className="text-[10px] text-text-muted flex items-center gap-1">
                      <Calendar size={10} /> {new Date(v.timestamp).toLocaleString([], { dateStyle: 'short', timeStyle: 'short' })}
                    </div>
                  </div>
                </div>
                <div className="text-right">
                  <div className="text-xs font-bold text-primary">{(v.confidence * 100).toFixed(0)}%</div>
                  <div className="text-[9px] text-text-muted uppercase font-bold">Conf.</div>
                </div>
              </div>
            </div>
          </div>
          )
        })}
      </div>

      {!loading && filteredViolations.length === 0 && (
        <div className="glass py-20 flex flex-col items-center justify-center gap-4 text-center">
          <div className="w-16 h-16 rounded-full bg-sidebar-bg flex items-center justify-center text-text-muted">
            <ShieldAlert size={32} />
          </div>
          <div>
            <h3 className="text-xl font-bold">No records found</h3>
            <p className="text-text-muted max-w-xs mx-auto">Try adjusting your search filters or check back later for new detections.</p>
          </div>
        </div>
      )}

      {/* Pagination */}
      <div className="flex items-center justify-center gap-4 pt-8">
        <button 
          onClick={() => setPage(p => Math.max(1, p - 1))}
          disabled={page === 1}
          className="glass p-2 disabled:opacity-30 disabled:cursor-not-allowed hover:bg-white/5"
        >
          <ChevronLeft size={20} />
        </button>
        <span className="text-sm font-bold uppercase tracking-widest text-text-muted">Page {page}</span>
        <button 
          onClick={() => setPage(p => p + 1)}
          disabled={violations.length < limit}
          className="glass p-2 disabled:opacity-30 disabled:cursor-not-allowed hover:bg-white/5"
        >
          <ChevronRight size={20} />
        </button>
      </div>
    </div>
  )
}
