'use client'
import React from 'react'
import { Eye, Clock, AlertTriangle, CheckCircle2 } from 'lucide-react'

interface Violation {
  id: number
  timestamp: string
  violation_type: string
  confidence: number
  severity: 'low' | 'medium' | 'high' | 'critical'
  reviewed: boolean
  track_id: number
  image_path?: string
  asset_url?: string
}

interface ViolationTableProps {
  violations: Violation[]
}

export const ViolationTable: React.FC<ViolationTableProps> = ({ violations }) => {
  const getEvidenceUrl = (violation: Violation) => {
    if (violation.asset_url) return violation.asset_url
    if (violation.image_path) return `http://localhost:8000/api/assets/${violation.image_path}`
    return null
  }

  const getSeverityBadge = (severity: string) => {
    switch (severity) {
      case 'critical': return <span className="badge" style={{ background: 'rgba(244, 63, 94, 0.2)', color: '#f43f5e', border: '1px solid #f43f5e' }}>CRITICAL</span>
      case 'high':     return <span className="badge badge-danger">HIGH</span>
      case 'medium':   return <span className="badge badge-warning">MEDIUM</span>
      default:         return <span className="badge badge-success">LOW</span>
    }
  }

  return (
    <div className="overflow-x-auto">
      <table className="data-table">
        <thead>
          <tr>
            <th>Violation</th>
            <th>Type</th>
            <th>Confidence</th>
            <th>Severity</th>
            <th>Status</th>
            <th>Time</th>
            <th>Actions</th>
          </tr>
        </thead>
        <tbody>
          {violations.map((v) => {
            const evidenceUrl = getEvidenceUrl(v)
            return (
            <tr key={v.id} className="animate-slide-up">
              <td>
                <div className="flex items-center gap-3">
                  <div className="w-12 h-12 rounded-lg bg-sidebar-bg flex-shrink-0 overflow-hidden border border-card-border">
                    {evidenceUrl ? (
                      <img 
                        src={evidenceUrl} 
                        alt="Crop" 
                        className="w-full h-full object-cover"
                      />
                    ) : (
                      <div className="w-full h-full flex items-center justify-center text-text-muted">
                        <AlertTriangle size={16} />
                      </div>
                    )}
                  </div>
                  <div>
                    <div className="font-semibold">Track #{v.track_id}</div>
                    <div className="text-xs text-text-muted">ID: {v.id}</div>
                  </div>
                </div>
              </td>
              <td>
                <span className="text-sm font-medium">{v.violation_type}</span>
              </td>
              <td>
                <div className="flex items-center gap-2">
                  <div className="w-16 h-1.5 bg-sidebar-bg rounded-full overflow-hidden">
                    <div 
                      className="h-full bg-primary" 
                      style={{ width: `${v.confidence * 100}%` }}
                    ></div>
                  </div>
                  <span className="text-xs font-mono">{(v.confidence * 100).toFixed(0)}%</span>
                </div>
              </td>
              <td>{getSeverityBadge(v.severity)}</td>
              <td>
                {v.reviewed ? (
                  <div className="flex items-center gap-1.5 text-secondary text-xs font-medium">
                    <CheckCircle2 size={14} /> Reviewed
                  </div>
                ) : (
                  <div className="flex items-center gap-1.5 text-warning text-xs font-medium">
                    <Clock size={14} /> Pending
                  </div>
                )}
              </td>
              <td className="text-sm text-text-muted font-mono">
                {new Date(v.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })}
              </td>
              <td>
                <button className="p-2 hover:bg-primary/10 rounded-lg text-text-muted hover:text-primary transition-colors">
                  <Eye size={18} />
                </button>
              </td>
            </tr>
            )
          })}
          {violations.length === 0 && (
            <tr>
              <td colSpan={7} className="text-center py-12 text-text-muted">
                <div className="flex flex-col items-center gap-2">
                  <CheckCircle2 size={32} className="opacity-20" />
                  <p>No violations detected in the current period.</p>
                </div>
              </td>
            </tr>
          )}
        </tbody>
      </table>
    </div>
  )
}
