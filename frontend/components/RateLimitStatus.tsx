'use client'

import React from 'react'
import { Box, Card, CardContent, Typography, LinearProgress, Chip } from '@mui/material'
import { Warning, CheckCircle } from '@mui/icons-material'
import { RateLimitStatus as RateLimitStatusType } from '@/lib/api'

interface RateLimitStatusProps {
  status: RateLimitStatusType
}

export default function RateLimitStatus({ status }: RateLimitStatusProps) {
  const { total_calls, max_calls, remaining_calls, calls_by_service } = status
  const percentage = (total_calls / max_calls) * 100
  const isWarning = percentage >= 80
  const isCritical = percentage >= 95

  return (
    <Card>
      <CardContent>
        <Box display="flex" alignItems="center" justifyContent="space-between" mb={2}>
          <Typography variant="h6" component="h2">
            Rate Limit Status
          </Typography>
          {isCritical ? (
            <Chip
              icon={<Warning />}
              label="Critical"
              color="error"
              size="small"
            />
          ) : isWarning ? (
            <Chip
              icon={<Warning />}
              label="Warning"
              color="warning"
              size="small"
            />
          ) : (
            <Chip
              icon={<CheckCircle />}
              label="OK"
              color="success"
              size="small"
            />
          )}
        </Box>

        <Box mb={2}>
          <Box display="flex" justifyContent="space-between" mb={1}>
            <Typography variant="body2" color="text.secondary">
              Calls Used
            </Typography>
            <Typography variant="body2" fontWeight="bold">
              {total_calls} / {max_calls}
            </Typography>
          </Box>
          <LinearProgress
            variant="determinate"
            value={percentage}
            color={isCritical ? 'error' : isWarning ? 'warning' : 'primary'}
            sx={{ height: 8, borderRadius: 4 }}
          />
        </Box>

        <Box mb={2}>
          <Typography variant="body2" color="text.secondary" gutterBottom>
            Remaining Calls: <strong>{remaining_calls}</strong>
          </Typography>
        </Box>

        {Object.keys(calls_by_service).length > 0 && (
          <Box>
            <Typography variant="body2" color="text.secondary" gutterBottom>
              Calls by Service:
            </Typography>
            <Box display="flex" gap={1} flexWrap="wrap" mt={1}>
              {Object.entries(calls_by_service).map(([service, count]) => (
                <Chip
                  key={service}
                  label={`${service}: ${count}`}
                  size="small"
                  variant="outlined"
                />
              ))}
            </Box>
          </Box>
        )}
      </CardContent>
    </Card>
  )
}




