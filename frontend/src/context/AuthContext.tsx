// frontend/src/context/AuthContext.tsx
'use client'
import React, { createContext, useContext, useState, useEffect } from 'react'
import { useRouter, usePathname } from 'next/navigation'
import { setCookie, deleteCookie, getCookie } from 'cookies-next'

interface User {
  id: number
  username: string
  full_name?: string | null
  email: string | null
  role: string
  avatar_url: string | null
  google_id?: string | null
  created_at?: string
}

interface AuthContextType {
  user: User | null
  token: string | null
  login: (token: string, user: User) => void
  logout: () => void
  isAuthenticated: boolean
  loading: boolean
}

const AuthContext = createContext<AuthContextType | undefined>(undefined)

export const AuthProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [user, setUser] = useState<User | null>(null)
  const [token, setToken] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)
  const router = useRouter()
  const pathname = usePathname()

  useEffect(() => {
    const savedToken = getCookie('auth_token')
    const savedUser = localStorage.getItem('auth_user')

    if (savedToken && savedUser) {
      setToken(savedToken as string)
      setUser(JSON.parse(savedUser))
    }
    setLoading(false)
  }, [])

  // Auth Guard Logic
  useEffect(() => {
    if (!loading) {
      const publicPaths = ['/login', '/register', '/logout']
      if (!token && !publicPaths.includes(pathname)) {
        router.push('/login')
      }
    }
  }, [loading, token, pathname, router])

  const login = (newToken: string, newUser: User) => {
    setToken(newToken)
    setUser(newUser)
    setCookie('auth_token', newToken, { maxAge: 60 * 60 * 24 * 7 }) // 7 days
    localStorage.setItem('auth_user', JSON.stringify(newUser))
    router.push('/')
  }

  const logout = () => {
    setToken(null)
    setUser(null)
    deleteCookie('auth_token')
    localStorage.removeItem('auth_user')
    router.push('/login')
  }

  return (
    <AuthContext.Provider value={{ user, token, login, logout, isAuthenticated: !!token, loading }}>
      {children}
    </AuthContext.Provider>
  )
}

export const useAuth = () => {
  const context = useContext(AuthContext)
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider')
  }
  return context
}
