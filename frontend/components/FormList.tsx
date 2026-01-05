'use client'

import React from 'react'
import {
  Box,
  Card,
  CardContent,
  Typography,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Chip,
  Link,
  CircularProgress,
} from '@mui/material'
import { CheckCircle, Error, HourglassEmpty, Visibility } from '@mui/icons-material'
import { FormInfo } from '@/lib/api'

interface FormListProps {
  forms: FormInfo[]
  onFormClick?: (formId: string) => void
  loading?: boolean
}

export default function FormList({ forms, onFormClick, loading }: FormListProps) {
  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'completed':
        return <CheckCircle color="success" fontSize="small" />
      case 'error':
        return <Error color="error" fontSize="small" />
      case 'processing':
        return <CircularProgress size={16} />
      default:
        return <HourglassEmpty color="disabled" fontSize="small" />
    }
  }

  const getStatusColor = (status: string): 'success' | 'error' | 'warning' | 'default' => {
    switch (status) {
      case 'completed':
        return 'success'
      case 'error':
        return 'error'
      case 'processing':
        return 'warning'
      default:
        return 'default'
    }
  }

  if (loading) {
    return (
      <Card>
        <CardContent>
          <Box display="flex" justifyContent="center" alignItems="center" minHeight={200}>
            <CircularProgress />
          </Box>
        </CardContent>
      </Card>
    )
  }

  if (forms.length === 0) {
    return (
      <Card>
        <CardContent>
          <Typography variant="body1" color="text.secondary" align="center">
            No forms found. Start scraping to see forms here.
          </Typography>
        </CardContent>
      </Card>
    )
  }

  return (
    <Card>
      <CardContent>
        <Typography variant="h6" component="h2" gutterBottom>
          Scraped Forms ({forms.length})
        </Typography>
        <TableContainer component={Paper} variant="outlined">
          <Table>
            <TableHead>
              <TableRow>
                <TableCell>Status</TableCell>
                <TableCell>Form Name</TableCell>
                <TableCell align="right">OCR Confidence</TableCell>
                <TableCell align="right">Text Length</TableCell>
                <TableCell>Actions</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {forms.map((form) => (
                <TableRow key={form.id} hover>
                  <TableCell>
                    <Chip
                      icon={getStatusIcon(form.status)}
                      label={form.status}
                      color={getStatusColor(form.status)}
                      size="small"
                    />
                  </TableCell>
                  <TableCell>
                    <Typography variant="body2" fontWeight="medium">
                      {form.name}
                    </Typography>
                    {form.error_message && (
                      <Typography variant="caption" color="error" display="block">
                        {form.error_message}
                      </Typography>
                    )}
                  </TableCell>
                  <TableCell align="right">
                    {form.ocr_confidence != null && typeof form.ocr_confidence === 'number' ? (
                      <Typography variant="body2">
                        {form.ocr_confidence.toFixed(1)}%
                      </Typography>
                    ) : (
                      <Typography variant="body2" color="text.secondary">
                        -
                      </Typography>
                    )}
                  </TableCell>
                  <TableCell align="right">
                    {form.text_length != null && typeof form.text_length === 'number' ? (
                      <Typography variant="body2">
                        {form.text_length.toLocaleString()}
                      </Typography>
                    ) : (
                      <Typography variant="body2" color="text.secondary">
                        -
                      </Typography>
                    )}
                  </TableCell>
                  <TableCell>
                    {form.status === 'completed' && onFormClick && (
                      <Link
                        component="button"
                        variant="body2"
                        onClick={() => onFormClick(form.id)}
                        sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}
                      >
                        <Visibility fontSize="small" />
                        View
                      </Link>
                    )}
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </TableContainer>
      </CardContent>
    </Card>
  )
}

