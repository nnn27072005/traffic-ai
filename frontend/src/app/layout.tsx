// frontend/src/app/layout.tsx
import './globals.css'
import type { Metadata } from 'next'
import { Sidebar } from '@/components/Sidebar'
import { Header } from '@/components/Header'
import { AuthProvider } from '@/context/AuthContext'
import { Space_Grotesk, Inter } from 'next/font/google'

const spaceGrotesk = Space_Grotesk({ 
  subsets: ['latin'],
  variable: '--font-space-grotesk',
})

const inter = Inter({
  subsets: ['latin'],
  variable: '--font-inter',
})

export const metadata: Metadata = {
  title: 'Traffic AI | Advanced Monitoring',
  description: 'AI-Powered Traffic Safety & Analytics System',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en" className="dark">
      <body className={`${inter.variable} ${spaceGrotesk.variable} font-body bg-background text-text-main antialiased selection:bg-primary-container selection:text-black overflow-hidden`}>
        <AuthProvider>
          <div className="flex h-screen w-full relative overflow-hidden">
            {/* Background Grid */}
            <div className="absolute inset-0 grid-bg pointer-events-none opacity-40"></div>
            
            <Sidebar />
            
            <main className="flex-1 flex flex-col min-w-0 md:ml-64 relative overflow-hidden">
              <Header />
              <div className="flex-1 overflow-auto">
                {children}
              </div>
            </main>
          </div>
        </AuthProvider>
      </body>
    </html>
  )
}
