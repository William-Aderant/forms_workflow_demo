'use client'

import { ThemeProvider, createTheme } from '@mui/material/styles'
import CssBaseline from '@mui/material/CssBaseline'
import { useMemo, useEffect, useState } from 'react'

export default function ThemeProviderWrapper({
  children,
}: {
  children: React.ReactNode
}) {
  const [mounted, setMounted] = useState(false)
  
  // Ensure we're on the client before creating theme
  useEffect(() => {
    setMounted(true)
  }, [])
  
  // Create theme inside component to ensure it's only called on client
  const theme = useMemo(() => {
    // Only create theme on client side
    if (typeof window === 'undefined') {
      // Return a minimal theme object for SSR (shouldn't be used)
      return {} as any
    }
    return createTheme({
      palette: {
        mode: 'light',
        primary: {
          main: '#1976d2',
        },
        secondary: {
          main: '#dc004e',
        },
      },
      typography: {
        fontFamily: [
          '-apple-system',
          'BlinkMacSystemFont',
          '"Segoe UI"',
          'Roboto',
          '"Helvetica Neue"',
          'Arial',
          'sans-serif',
        ].join(','),
      },
    })
  }, [])
  
  // Don't render until mounted (client-side only)
  if (!mounted) {
    return <>{children}</>
  }

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      {children}
    </ThemeProvider>
  )
}

