'use client'
import React from 'react'
import { ResponsiveContainer, AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip } from 'recharts'

interface DataPoint {
  timestamp: string
  vehicle_count: number
  count_in: number
  count_out: number
}

interface AnalyticsChartsProps {
  data: any[]
}

export const AnalyticsCharts: React.FC<AnalyticsChartsProps> = ({ data }) => {
  // Format data for Recharts
  const chartData = data.map(p => ({
    time: new Date(p.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    Inflow: p.count_in,
    Outflow: p.count_out,
    Total: p.vehicle_count
  })).reverse()

  return (
    <div style={{ width: '100%', height: 300 }}>
      <ResponsiveContainer>
        <AreaChart data={chartData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
          <defs>
            <linearGradient id="colorIn" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="var(--primary)" stopOpacity={0.3}/>
              <stop offset="95%" stopColor="var(--primary)" stopOpacity={0}/>
            </linearGradient>
            <linearGradient id="colorOut" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="var(--secondary)" stopOpacity={0.3}/>
              <stop offset="95%" stopColor="var(--secondary)" stopOpacity={0}/>
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="rgba(255,255,255,0.05)" />
          <XAxis 
            dataKey="time" 
            axisLine={false} 
            tickLine={false} 
            tick={{ fill: 'var(--text-muted)', fontSize: 10 }}
            minTickGap={30}
          />
          <YAxis 
            axisLine={false} 
            tickLine={false} 
            tick={{ fill: 'var(--text-muted)', fontSize: 10 }} 
          />
          <Tooltip 
            contentStyle={{ 
              backgroundColor: 'var(--sidebar-bg)', 
              borderColor: 'var(--card-border)',
              borderRadius: '8px',
              fontSize: '12px'
            }}
            itemStyle={{ fontWeight: 600 }}
          />
          <Area 
            type="monotone" 
            dataKey="Inflow" 
            stroke="var(--primary)" 
            strokeWidth={2}
            fillOpacity={1} 
            fill="url(#colorIn)" 
          />
          <Area 
            type="monotone" 
            dataKey="Outflow" 
            stroke="var(--secondary)" 
            strokeWidth={2}
            fillOpacity={1} 
            fill="url(#colorOut)" 
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  )
}
