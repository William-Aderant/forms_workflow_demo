'use client'

import React, { useState, useEffect } from 'react'
import {
  Container,
  Box,
  Typography,
  Button,
  Grid,
  Alert,
  CircularProgress,
  Tabs,
  Tab,
  Paper,
} from '@mui/material'
import { PlayArrow, Refresh, Compare } from '@mui/icons-material'
import api, { FormInfo, RateLimitStatus } from '@/lib/api'
import RateLimitStatusComponent from '@/components/RateLimitStatus'
import FormList from '@/components/FormList'
import TechStackComparison from '@/components/TechStackComparison'
import { useRouter } from 'next/navigation'

interface TabPanelProps {
  children?: React.ReactNode
  index: number
  value: number
}

function TabPanel(props: TabPanelProps) {
  const { children, value, index, ...other } = props
  return (
    <div role="tabpanel" hidden={value !== index} {...other}>
      {value === index && <Box>{children}</Box>}
    </div>
  )
}

export default function HomePage() {
  const [forms, setForms] = useState<FormInfo[]>([])
  const [rateLimitStatus, setRateLimitStatus] = useState<RateLimitStatus | null>(null)
  const [loading, setLoading] = useState(false)
  const [scraping, setScraping] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [progress, setProgress] = useState<{ current?: number; total?: number; message?: string } | null>(null)
  const [currentForm, setCurrentForm] = useState<string | null>(null)
  const [activeTab, setActiveTab] = useState(0)
  const router = useRouter()

  const fetchForms = async () => {
    try {
      setLoading(true)
      const fetchedForms = await api.listForms()
      setForms(fetchedForms)
    } catch (err: any) {
      setError(err.message || 'Failed to fetch forms')
    } finally {
      setLoading(false)
    }
  }

  const fetchRateLimitStatus = async () => {
    try {
      const status = await api.getRateLimitStatus()
      setRateLimitStatus(status)
    } catch (err) {
      console.error('Failed to fetch rate limit status:', err)
    }
  }

  const handleScrape = async () => {
    // #region agent log
    fetch('http://127.0.0.1:7252/ingest/564ebba2-2403-423b-931f-138186fce4fd',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'page.tsx:50',message:'handleScrape called',data:{timestamp:Date.now()},timestamp:Date.now(),sessionId:'debug-session',runId:'run1',hypothesisId:'A'})}).catch(()=>{});
    // #endregion
    try {
      setScraping(true)
      setError(null)
      setProgress(null)
      setCurrentForm(null)
      
      // #region agent log
      fetch('http://127.0.0.1:7252/ingest/564ebba2-2403-423b-931f-138186fce4fd',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'page.tsx:57',message:'About to call api.scrapeForms',data:{url:'https://selfhelp.courts.ca.gov/find-forms/all'},timestamp:Date.now(),sessionId:'debug-session',runId:'run1',hypothesisId:'A'})}).catch(()=>{});
      // #endregion
      const response = await api.scrapeForms({
        url: 'https://selfhelp.courts.ca.gov/find-forms/all',
      })
      // #region agent log
      fetch('http://127.0.0.1:7252/ingest/564ebba2-2403-423b-931f-138186fce4fd',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'page.tsx:61',message:'api.scrapeForms returned',data:{jobId:response?.job_id,hasJobId:!!response?.job_id},timestamp:Date.now(),sessionId:'debug-session',runId:'run1',hypothesisId:'A'})}).catch(()=>{});
      // #endregion
      
      if (response.job_id) {
        // Subscribe to real-time progress updates
        const unsubscribe = api.subscribeToProgress(
          response.job_id,
          (event, data) => {
            switch (event) {
              case 'start':
                setProgress({ message: data.message || 'Starting...' })
                break
              case 'progress':
                setProgress({
                  current: data.current,
                  total: data.total,
                  message: data.message || `Processing ${data.current}/${data.total}...`
                })
                // Refresh forms list to show new/updated forms
                fetchForms()
                fetchRateLimitStatus()
                break
              case 'form_start':
                setCurrentForm(data.form_name || data.form_id)
                setProgress({
                  current: data.index,
                  total: data.total,
                  message: `Starting: ${data.form_name}`
                })
                break
              case 'form_processing':
                setProgress(prev => ({
                  ...prev,
                  message: `Processing: ${data.form_name}`
                }))
                break
              case 'form_complete':
                setCurrentForm(null)
                setProgress({
                  current: data.processed,
                  total: data.total,
                  message: `Completed: ${data.form_name}`
                })
                // Refresh immediately when form completes
                fetchForms()
                fetchRateLimitStatus()
                break
              case 'form_error':
                setCurrentForm(null)
                setProgress(prev => ({
                  ...prev,
                  message: `Error: ${data.form_name} - ${data.error}`
                }))
                fetchForms()
                break
              case 'complete':
                setScraping(false)
                setProgress(null)
                setCurrentForm(null)
                fetchForms()
                fetchRateLimitStatus()
                break
              case 'error':
                setScraping(false)
                setError(data.message || 'Scraping failed')
                setProgress(null)
                setCurrentForm(null)
                break
            }
          },
          (err) => {
            console.error('SSE error:', err)
            setScraping(false)
            setError('Connection to server lost')
          }
        )
        
        // Store unsubscribe function (will be called when component unmounts or scraping completes)
        // For now, we'll let it close automatically on complete/error events
      } else {
        // Fallback to polling if no job_id
        setTimeout(() => {
          fetchForms()
          fetchRateLimitStatus()
        }, 2000)
        setScraping(false)
      }
    } catch (err: any) {
      // #region agent log
      fetch('http://127.0.0.1:7252/ingest/564ebba2-2403-423b-931f-138186fce4fd',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'page.tsx:145',message:'handleScrape catch block',data:{errorMessage:err?.message,errorName:err?.name,isNetworkError:err?.message?.includes('Network Error'),stack:err?.stack},timestamp:Date.now(),sessionId:'debug-session',runId:'run1',hypothesisId:'A,B,C'})}).catch(()=>{});
      // #endregion
      setError(err.message || 'Failed to start scraping')
      setScraping(false)
    }
  }

  const handleFormClick = (formId: string) => {
    router.push(`/forms/${formId}`)
  }

  useEffect(() => {
    fetchForms()
    fetchRateLimitStatus()

    // Poll for updates every 10 seconds (less frequent since we have real-time updates)
    // When scraping is active, SSE provides real-time updates
    const interval = setInterval(() => {
      if (!scraping) {
        fetchForms()
        fetchRateLimitStatus()
      }
    }, 10000)

    return () => clearInterval(interval)
  }, [scraping])

  return (
    <Container maxWidth="xl" sx={{ py: 4 }}>
      <Box mb={4}>
        <Typography variant="h4" component="h1" gutterBottom>
          Court Forms OCR Workflow
        </Typography>
        <Typography variant="body1" color="text.secondary" paragraph>
          Scrape and process court forms from California Courts website with OCR.
          Compare results across different technology configurations.
        </Typography>
      </Box>

      {/* Main Navigation Tabs */}
      <Paper sx={{ mb: 3 }}>
        <Tabs 
          value={activeTab} 
          onChange={(_, v) => setActiveTab(v)}
          variant="fullWidth"
        >
          <Tab 
            icon={<PlayArrow />} 
            label="Form Scraping" 
            iconPosition="start"
          />
          <Tab 
            icon={<Compare />} 
            label="Technology Stack Comparison" 
            iconPosition="start"
          />
        </Tabs>
      </Paper>

      {/* Tab Panel: Form Scraping */}
      <TabPanel value={activeTab} index={0}>
        {error && (
          <Alert severity="error" sx={{ mb: 3 }} onClose={() => setError(null)}>
            {error}
          </Alert>
        )}

        {progress && (
          <Alert severity="info" sx={{ mb: 3 }}>
            {progress.message}
            {progress.current !== undefined && progress.total !== undefined && (
              <Typography variant="body2" sx={{ mt: 1 }}>
                Progress: {progress.current} / {progress.total} forms
              </Typography>
            )}
            {currentForm && (
              <Typography variant="body2" sx={{ mt: 1, fontWeight: 'bold' }}>
                Current: {currentForm}
              </Typography>
            )}
          </Alert>
        )}

        <Grid container spacing={3}>
          <Grid item xs={12} md={4}>
            <Box mb={2}>
              <Button
                variant="contained"
                startIcon={scraping ? <CircularProgress size={20} /> : <PlayArrow />}
                onClick={handleScrape}
                disabled={scraping}
                fullWidth
                size="large"
              >
                {scraping ? 'Scraping...' : 'Scrape Forms'}
              </Button>
            </Box>
            <Button
              variant="outlined"
              startIcon={<Refresh />}
              onClick={() => {
                fetchForms()
                fetchRateLimitStatus()
              }}
              fullWidth
              disabled={loading}
            >
              Refresh
            </Button>
            {rateLimitStatus && (
              <Box mt={3}>
                <RateLimitStatusComponent status={rateLimitStatus} />
              </Box>
            )}
          </Grid>

          <Grid item xs={12} md={8}>
            <FormList
              forms={forms}
              onFormClick={handleFormClick}
              loading={loading}
            />
          </Grid>
        </Grid>
      </TabPanel>

      {/* Tab Panel: Technology Stack Comparison */}
      <TabPanel value={activeTab} index={1}>
        <TechStackComparison />
      </TabPanel>
    </Container>
  )
}
