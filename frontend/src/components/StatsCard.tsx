'use client'
import React from 'react'
import { LucideIcon } from 'lucide-react'

interface StatsCardProps {
  label: string
  value: string | number
  icon: LucideIcon
  trend?: string
  trendColor?: string
  color?: string
  delay?: string
}

export const StatsCard: React.FC<StatsCardProps> = ({ 
  label, value, icon: Icon, trend, trendColor = 'text-secondary', color = 'var(--primary)', delay = '' 
}) => {
  return (
    <div className={`glass stats-card animate-slide-up ${delay}`}>
      <div className="flex justify-between items-start mb-4">
        <span className="stats-label">{label}</span>
        <div className="p-2 rounded-lg" style={{ background: `${color}15`, color }}>
          <Icon size={20} />
        </div>
      </div>
      <div className="stats-value">{value}</div>
      {trend && (
        <div className={`text-xs font-medium mt-2 flex items-center gap-1 ${trendColor}`}>
          {trend}
        </div>
      )}
    </div>
  )
}
