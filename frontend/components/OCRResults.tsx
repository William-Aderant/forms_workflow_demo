'use client'

import React, { useState } from 'react'
import {
  Box,
  Card,
  CardContent,
  Typography,
  Button,
  TextField,
  IconButton,
  Tooltip,
} from '@mui/material'
import { ContentCopy, Check } from '@mui/icons-material'

interface OCRResultsProps {
  markdownText?: string
  formName?: string
}

export default function OCRResults({ markdownText, formName }: OCRResultsProps) {
  const [copied, setCopied] = useState(false)

  const handleCopy = async () => {
    if (markdownText) {
      await navigator.clipboard.writeText(markdownText)
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    }
  }

  if (!markdownText) {
    return (
      <Card>
        <CardContent>
          <Typography variant="body1" color="text.secondary" align="center">
            No OCR results available
          </Typography>
        </CardContent>
      </Card>
    )
  }

  return (
    <Card>
      <CardContent>
        <Box display="flex" alignItems="center" justifyContent="space-between" mb={2}>
          <Typography variant="h6" component="h2">
            OCR Results {formName && `- ${formName}`}
          </Typography>
          <Tooltip title={copied ? 'Copied!' : 'Copy to clipboard'}>
            <IconButton onClick={handleCopy} size="small">
              {copied ? <Check color="success" /> : <ContentCopy />}
            </IconButton>
          </Tooltip>
        </Box>

        <TextField
          fullWidth
          multiline
          rows={20}
          value={markdownText}
          variant="outlined"
          InputProps={{
            readOnly: true,
            sx: {
              fontFamily: 'monospace',
              fontSize: '0.875rem',
            },
          }}
          sx={{
            '& .MuiOutlinedInput-root': {
              backgroundColor: 'background.paper',
            },
          }}
        />

        <Box mt={2}>
          <Typography variant="caption" color="text.secondary">
            {markdownText.length} characters
          </Typography>
        </Box>
      </CardContent>
    </Card>
  )
}




