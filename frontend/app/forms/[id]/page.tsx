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
  Breadcrumbs,
  Link,
} from '@mui/material'
import { ArrowBack, Home } from '@mui/icons-material'
import { useRouter, useParams } from 'next/navigation'
import api, { FormDetail, ImageMetadata } from '@/lib/api'
import OCRResults from '@/components/OCRResults'
import ImageGallery from '@/components/ImageGallery'

export default function FormDetailPage() {
  const params = useParams()
  const router = useRouter()
  const formId = params.id as string

  const [form, setForm] = useState<FormDetail | null>(null)
  const [images, setImages] = useState<ImageMetadata[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true)
        const [formData, imageData] = await Promise.all([
          api.getForm(formId),
          api.getFormImages(formId),
        ])
        setForm(formData)
        setImages(imageData)
      } catch (err: any) {
        setError(err.message || 'Failed to fetch form details')
      } finally {
        setLoading(false)
      }
    }

    if (formId) {
      fetchData()
    }
  }, [formId])

  if (loading) {
    return (
      <Container maxWidth="lg" sx={{ py: 4 }}>
        <Box display="flex" justifyContent="center" alignItems="center" minHeight={400}>
          <CircularProgress />
        </Box>
      </Container>
    )
  }

  if (error || !form) {
    return (
      <Container maxWidth="lg" sx={{ py: 4 }}>
        <Alert severity="error">{error || 'Form not found'}</Alert>
        <Box mt={2}>
          <Button
            variant="outlined"
            startIcon={<ArrowBack />}
            onClick={() => router.push('/')}
          >
            Back to Dashboard
          </Button>
        </Box>
      </Container>
    )
  }

  return (
    <Container maxWidth="lg" sx={{ py: 4 }}>
      <Breadcrumbs sx={{ mb: 3 }}>
        <Link
          color="inherit"
          href="/"
          onClick={(e) => {
            e.preventDefault()
            router.push('/')
          }}
          sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}
        >
          <Home fontSize="small" />
          Home
        </Link>
        <Typography color="text.primary">{form.name}</Typography>
      </Breadcrumbs>

      <Box mb={3} display="flex" alignItems="center" justifyContent="space-between">
        <Box>
          <Typography variant="h4" component="h1" gutterBottom>
            {form.name}
          </Typography>
          <Typography variant="body2" color="text.secondary">
            Status: {form.status} | OCR Confidence: {form.ocr_confidence?.toFixed(1) || 'N/A'}% |
            Text Length: {form.text_length?.toLocaleString() || 'N/A'} characters
          </Typography>
        </Box>
        <Button
          variant="outlined"
          startIcon={<ArrowBack />}
          onClick={() => router.push('/')}
        >
          Back
        </Button>
      </Box>

      {form.error_message && (
        <Alert severity="error" sx={{ mb: 3 }}>
          {form.error_message}
        </Alert>
      )}

      <Grid container spacing={3}>
        <Grid item xs={12}>
          <OCRResults markdownText={form.markdown_text} formName={form.name} />
        </Grid>
        {images.length > 0 && (
          <Grid item xs={12}>
            <ImageGallery images={images} formName={form.name} />
          </Grid>
        )}
      </Grid>
    </Container>
  )
}




