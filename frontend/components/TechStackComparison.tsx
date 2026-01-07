'use client'

import React, { useState, useCallback } from 'react'
import {
  Box,
  Card,
  CardContent,
  Typography,
  Button,
  CircularProgress,
  Alert,
  Paper,
  Tabs,
  Tab,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Chip,
  LinearProgress,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Grid,
  Divider,
  IconButton,
  Tooltip,
} from '@mui/material'
import {
  CloudUpload,
  Compare,
  ExpandMore,
  CheckCircle,
  Warning,
  Error as ErrorIcon,
  TrendingUp,
  Speed,
  Psychology,
  Image as ImageIcon,
  ZoomIn,
  ZoomOut,
} from '@mui/icons-material'
import { api, ComparisonResponse, ModeResult, FieldComparison } from '@/lib/api'

interface TabPanelProps {
  children?: React.ReactNode
  index: number
  value: number
}

function TabPanel(props: TabPanelProps) {
  const { children, value, index, ...other } = props
  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`tabpanel-${index}`}
      aria-labelledby={`tab-${index}`}
      {...other}
    >
      {value === index && <Box sx={{ pt: 2 }}>{children}</Box>}
    </div>
  )
}

const MODE_COLORS: Record<string, string> = {
  textract_only: '#ef5350',
  textract_heuristics: '#ff9800',
  textract_layoutlm: '#42a5f5',
  full_stack: '#66bb6a',
}

const MODE_NAMES: Record<string, string> = {
  textract_only: 'Textract Only',
  textract_heuristics: 'Textract + Heuristics',
  textract_layoutlm: 'Textract + LayoutLM',
  full_stack: 'Full Stack',
}

const MODE_ICONS: Record<string, React.ReactNode> = {
  textract_only: <CloudUpload />,
  textract_heuristics: <Psychology />,
  textract_layoutlm: <Speed />,
  full_stack: <CheckCircle />,
}

function ConfidenceBadge({ confidence }: { confidence: number }) {
  const color = confidence >= 0.7 ? 'success' : confidence >= 0.4 ? 'warning' : 'error'
  return (
    <Chip
      label={`${(confidence * 100).toFixed(1)}%`}
      size="small"
      color={color}
      variant="outlined"
    />
  )
}

function ModeCard({ mode, result }: { mode: string; result: ModeResult }) {
  const color = MODE_COLORS[mode] || '#grey'
  
  return (
    <Card sx={{ height: '100%', borderLeft: `4px solid ${color}` }}>
      <CardContent>
        <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
          <Box sx={{ color, mr: 1 }}>{MODE_ICONS[mode]}</Box>
          <Typography variant="h6" component="div">
            {MODE_NAMES[mode] || mode}
          </Typography>
        </Box>
        
        <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
          {result.mode_description}
        </Typography>
        
        <Box sx={{ mb: 2 }}>
          <Typography variant="caption" color="text.secondary">Components:</Typography>
          <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5, mt: 0.5 }}>
            {result.components_used.map((comp, i) => (
              <Chip key={i} label={comp} size="small" variant="outlined" />
            ))}
          </Box>
        </Box>
        
        <Divider sx={{ my: 2 }} />
        
        <Grid container spacing={2}>
          <Grid item xs={6}>
            <Typography variant="caption" color="text.secondary">Avg Confidence</Typography>
            <Typography variant="h5" sx={{ color }}>
              {(result.statistics.average_confidence * 100).toFixed(1)}%
            </Typography>
          </Grid>
          <Grid item xs={6}>
            <Typography variant="caption" color="text.secondary">High Conf Fields</Typography>
            <Typography variant="h5">
              {result.statistics.high_confidence_count}/{result.statistics.total_fields}
            </Typography>
          </Grid>
          <Grid item xs={6}>
            <Typography variant="caption" color="text.secondary">Unknown Fields</Typography>
            <Typography variant="h5" color={result.statistics.unknown_field_count > 0 ? 'warning.main' : 'success.main'}>
              {result.statistics.unknown_field_count}
            </Typography>
          </Grid>
          <Grid item xs={6}>
            <Typography variant="caption" color="text.secondary">Processing Time</Typography>
            <Typography variant="h5">
              {result.processing_time_ms}ms
            </Typography>
          </Grid>
        </Grid>
      </CardContent>
    </Card>
  )
}

function FieldComparisonTable({ comparisons }: { comparisons: FieldComparison[] }) {
  const modes = ['textract_only', 'textract_heuristics', 'textract_layoutlm', 'full_stack']
  
  return (
    <TableContainer component={Paper} sx={{ maxHeight: 600 }}>
      <Table stickyHeader size="small">
        <TableHead>
          <TableRow>
            <TableCell>Field</TableCell>
            <TableCell>Type</TableCell>
            <TableCell>Label Text</TableCell>
            {modes.map(mode => (
              <TableCell key={mode} align="center" sx={{ 
                backgroundColor: `${MODE_COLORS[mode]}22`,
                borderBottom: `3px solid ${MODE_COLORS[mode]}`
              }}>
                {MODE_NAMES[mode]}
              </TableCell>
            ))}
            <TableCell>Improved?</TableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {comparisons.map((field) => (
            <TableRow key={field.field_id} hover>
              <TableCell>
                <Typography variant="caption" sx={{ fontFamily: 'monospace' }}>
                  {field.field_id}
                </Typography>
              </TableCell>
              <TableCell>
                <Chip label={field.field_type} size="small" />
              </TableCell>
              <TableCell sx={{ maxWidth: 150, overflow: 'hidden', textOverflow: 'ellipsis' }}>
                <Tooltip title={field.supporting_text || '(none)'}>
                  <span>{field.supporting_text || '—'}</span>
                </Tooltip>
              </TableCell>
              {modes.map(mode => {
                const result = field.results_by_mode[mode]
                if (!result) return <TableCell key={mode} align="center">—</TableCell>
                return (
                  <TableCell key={mode} align="center">
                    <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 0.5 }}>
                      <Typography variant="caption" sx={{ 
                        fontWeight: result.semantic_label !== 'UNKNOWN_FIELD' ? 'bold' : 'normal',
                        color: result.semantic_label === 'UNKNOWN_FIELD' ? 'text.disabled' : 'text.primary',
                        fontSize: '0.7rem'
                      }}>
                        {result.semantic_label.replace(/_/g, ' ')}
                      </Typography>
                      <ConfidenceBadge confidence={result.confidence} />
                    </Box>
                  </TableCell>
                )
              })}
              <TableCell align="center">
                {field.analysis.confidence_improved ? (
                  <TrendingUp color="success" />
                ) : field.analysis.label_changed ? (
                  <Warning color="warning" />
                ) : (
                  <Typography color="text.disabled">—</Typography>
                )}
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </TableContainer>
  )
}

function ImprovementSummary({ summary }: { summary: ComparisonResponse['improvement_summary'] }) {
  if (!summary.overall_improvement) return null
  
  const overall = summary.overall_improvement
  
  // Extract values with backward compatibility for old field names
  const confGain = overall.effective_confidence_gain_percent ?? overall.raw_confidence_gain_percent ?? 0;
  const unknownReduced = overall.unknown_fields_reduced ?? overall.unknown_reduction ?? 0;
  const highConfGain = overall.high_confidence_gained ?? overall.high_confidence_gain ?? 0;
  const totalTime = overall.total_processing_time_ms ?? overall.total_time_ms ?? 0;
  
  return (
    <Card sx={{ bgcolor: 'success.dark', color: 'white' }}>
      <CardContent>
        <Typography variant="h6" gutterBottom>
          📈 Overall Improvement: Baseline → Full Stack
        </Typography>
        <Grid container spacing={3}>
          <Grid item xs={6} md={3}>
            <Typography variant="caption">Confidence Gain</Typography>
            <Typography variant="h4">
              +{confGain.toFixed(1)}%
            </Typography>
          </Grid>
          <Grid item xs={6} md={3}>
            <Typography variant="caption">Unknown Fields Reduced</Typography>
            <Typography variant="h4">
              {(unknownReduced > 0 ? '-' : '')}{unknownReduced}
            </Typography>
          </Grid>
          <Grid item xs={6} md={3}>
            <Typography variant="caption">High Confidence Gained</Typography>
            <Typography variant="h4">
              +{highConfGain}
            </Typography>
          </Grid>
          <Grid item xs={6} md={3}>
            <Typography variant="caption">Total Processing Time</Typography>
            <Typography variant="h4">
              {totalTime}ms
            </Typography>
          </Grid>
        </Grid>
      </CardContent>
    </Card>
  )
}

function ConfidenceProgressionChart({ data }: { data: Array<{ mode: string; average_confidence: number }> }) {
  return (
    <Box sx={{ p: 2 }}>
      <Typography variant="subtitle2" gutterBottom>Confidence Progression</Typography>
      {data.map((item, index) => (
        <Box key={item.mode} sx={{ mb: 2 }}>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.5 }}>
            <Typography variant="caption">{MODE_NAMES[item.mode] || item.mode}</Typography>
            <Typography variant="caption">{(item.average_confidence * 100).toFixed(1)}%</Typography>
          </Box>
          <LinearProgress 
            variant="determinate" 
            value={item.average_confidence * 100} 
            sx={{ 
              height: 10, 
              borderRadius: 5,
              backgroundColor: `${MODE_COLORS[item.mode]}33`,
              '& .MuiLinearProgress-bar': {
                backgroundColor: MODE_COLORS[item.mode],
              }
            }}
          />
        </Box>
      ))}
    </Box>
  )
}

function ImageComparisonGrid({ modeResults }: { modeResults: Record<string, ModeResult> }) {
  const [selectedImage, setSelectedImage] = useState<string | null>(null)
  const [zoom, setZoom] = useState(100)
  const modes = ['textract_only', 'textract_heuristics', 'textract_layoutlm', 'full_stack']
  
  const hasImages = modes.some(mode => modeResults[mode]?.annotated_image)
  
  if (!hasImages) {
    return (
      <Box sx={{ p: 4, textAlign: 'center' }}>
        <ImageIcon sx={{ fontSize: 64, color: 'text.disabled', mb: 2 }} />
        <Typography color="text.secondary">
          No annotated images available. Images are generated when a page image is available.
        </Typography>
      </Box>
    )
  }
  
  return (
    <Box>
      {/* Full-size image modal */}
      {selectedImage && (
        <Box
          sx={{
            position: 'fixed',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            bgcolor: 'rgba(0,0,0,0.9)',
            zIndex: 9999,
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            cursor: 'pointer',
          }}
          onClick={() => setSelectedImage(null)}
        >
          <Box sx={{ position: 'absolute', top: 16, right: 16, display: 'flex', gap: 1 }}>
            <IconButton 
              onClick={(e) => { e.stopPropagation(); setZoom(z => Math.max(50, z - 25)); }}
              sx={{ bgcolor: 'white' }}
            >
              <ZoomOut />
            </IconButton>
            <Typography sx={{ color: 'white', lineHeight: '40px' }}>{zoom}%</Typography>
            <IconButton 
              onClick={(e) => { e.stopPropagation(); setZoom(z => Math.min(200, z + 25)); }}
              sx={{ bgcolor: 'white' }}
            >
              <ZoomIn />
            </IconButton>
            <Button 
              variant="contained" 
              color="error" 
              onClick={() => setSelectedImage(null)}
              sx={{ ml: 2 }}
            >
              Close
            </Button>
          </Box>
          <Box
            component="img"
            src={`data:image/png;base64,${selectedImage}`}
            alt="Annotated document"
            sx={{
              maxWidth: `${zoom}%`,
              maxHeight: '90vh',
              objectFit: 'contain',
            }}
            onClick={(e) => e.stopPropagation()}
          />
        </Box>
      )}
      
      <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
        <ImageIcon /> Visual Comparison - Click any image to enlarge
      </Typography>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
        Each image shows the same document processed through different technology configurations. 
        Notice how field labels and confidence levels improve from left to right.
      </Typography>
      
      {/* 2x2 Grid of images */}
      <Grid container spacing={2}>
        {modes.map(mode => {
          const result = modeResults[mode]
          const hasImage = result?.annotated_image
          const color = MODE_COLORS[mode]
          
          return (
            <Grid item xs={12} md={6} key={mode}>
              <Card 
                sx={{ 
                  height: '100%',
                  borderTop: `4px solid ${color}`,
                  cursor: hasImage ? 'pointer' : 'default',
                  transition: 'transform 0.2s, box-shadow 0.2s',
                  '&:hover': hasImage ? {
                    transform: 'scale(1.02)',
                    boxShadow: 6,
                  } : {}
                }}
                onClick={() => hasImage && setSelectedImage(result.annotated_image!)}
              >
                <CardContent sx={{ p: 1, '&:last-child': { pb: 1 } }}>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
                    <Typography variant="subtitle2" sx={{ color }}>
                      {MODE_NAMES[mode]}
                    </Typography>
                    {result && (
                      <Typography variant="caption" color="text.secondary">
                        {result.statistics.average_confidence.toFixed(1)}% avg | 
                        {result.statistics.high_confidence_count} high | 
                        {result.statistics.unknown_field_count} unknown
                      </Typography>
                    )}
                  </Box>
                  
                  {hasImage ? (
                    <Box
                      component="img"
                      src={`data:image/png;base64,${result.annotated_image}`}
                      alt={`${MODE_NAMES[mode]} annotated`}
                      sx={{
                        width: '100%',
                        height: 'auto',
                        maxHeight: 400,
                        objectFit: 'contain',
                        borderRadius: 1,
                        bgcolor: 'grey.100',
                      }}
                    />
                  ) : (
                    <Box 
                      sx={{ 
                        height: 200, 
                        display: 'flex', 
                        alignItems: 'center', 
                        justifyContent: 'center',
                        bgcolor: 'grey.100',
                        borderRadius: 1,
                      }}
                    >
                      <Typography color="text.disabled">No image available</Typography>
                    </Box>
                  )}
                </CardContent>
              </Card>
            </Grid>
          )
        })}
      </Grid>
      
      {/* Side-by-side progression view */}
      <Box sx={{ mt: 4 }}>
        <Typography variant="h6" gutterBottom>
          📊 Technology Progression (Horizontal Scroll)
        </Typography>
        <Box 
          sx={{ 
            display: 'flex', 
            gap: 2, 
            overflowX: 'auto', 
            pb: 2,
            '&::-webkit-scrollbar': { height: 8 },
            '&::-webkit-scrollbar-track': { bgcolor: 'grey.200', borderRadius: 4 },
            '&::-webkit-scrollbar-thumb': { bgcolor: 'grey.500', borderRadius: 4 },
          }}
        >
          {modes.map((mode, index) => {
            const result = modeResults[mode]
            const hasImage = result?.annotated_image
            const color = MODE_COLORS[mode]
            
            return (
              <React.Fragment key={mode}>
                <Box 
                  sx={{ 
                    minWidth: 300, 
                    flexShrink: 0,
                    borderTop: `3px solid ${color}`,
                    bgcolor: 'background.paper',
                    borderRadius: 1,
                    overflow: 'hidden',
                  }}
                >
                  <Box sx={{ p: 1, bgcolor: `${color}22` }}>
                    <Typography variant="caption" sx={{ fontWeight: 'bold', color }}>
                      Step {index + 1}: {MODE_NAMES[mode]}
                    </Typography>
                  </Box>
                  {hasImage ? (
                    <Box
                      component="img"
                      src={`data:image/png;base64,${result.annotated_image}`}
                      alt={`${MODE_NAMES[mode]}`}
                      sx={{
                        width: '100%',
                        height: 250,
                        objectFit: 'contain',
                        cursor: 'pointer',
                        bgcolor: 'grey.50',
                      }}
                      onClick={() => setSelectedImage(result.annotated_image!)}
                    />
                  ) : (
                    <Box sx={{ height: 250, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                      <Typography color="text.disabled" variant="caption">No image</Typography>
                    </Box>
                  )}
                </Box>
                {index < modes.length - 1 && (
                  <Box sx={{ display: 'flex', alignItems: 'center', color: 'text.secondary' }}>
                    <TrendingUp sx={{ fontSize: 32 }} />
                  </Box>
                )}
              </React.Fragment>
            )
          })}
        </Box>
      </Box>
    </Box>
  )
}

export default function TechStackComparison() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<ComparisonResponse | null>(null)
  const [activeTab, setActiveTab] = useState(0)

  const handleFileSelect = useCallback((event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (file) {
      if (!file.name.toLowerCase().endsWith('.pdf')) {
        setError('Please select a PDF file')
        return
      }
      setSelectedFile(file)
      setError(null)
      setResult(null)
    }
  }, [])

  const handleCompare = useCallback(async () => {
    if (!selectedFile) return

    setIsLoading(true)
    setError(null)

    try {
      const response = await api.comparePDF(selectedFile)
      
      if (!response.success) {
        throw new Error(response.error || 'Comparison failed')
      }
      
      setResult(response)
    } catch (err: any) {
      setError(err.message || 'Failed to process file')
    } finally {
      setIsLoading(false)
    }
  }, [selectedFile])

  const modes = ['textract_only', 'textract_heuristics', 'textract_layoutlm', 'full_stack']

  return (
    <Box sx={{ p: 3 }}>
      <Typography variant="h4" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
        <Compare /> Technology Stack Comparison
      </Typography>
      
      <Typography variant="body1" color="text.secondary" sx={{ mb: 3 }}>
        Upload a PDF to see how different technology configurations affect field detection and classification accuracy.
      </Typography>

      {/* Upload Section */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Box sx={{ display: 'flex', gap: 2, alignItems: 'center', flexWrap: 'wrap' }}>
            <Button
              variant="outlined"
              component="label"
              startIcon={<CloudUpload />}
            >
              Select PDF
              <input
                type="file"
                hidden
                accept=".pdf"
                onChange={handleFileSelect}
              />
            </Button>
            
            {selectedFile && (
              <Typography variant="body2" color="text.secondary">
                {selectedFile.name} ({(selectedFile.size / 1024).toFixed(1)} KB)
              </Typography>
            )}
            
            <Button
              variant="contained"
              color="primary"
              onClick={handleCompare}
              disabled={!selectedFile || isLoading}
              startIcon={isLoading ? <CircularProgress size={20} /> : <Compare />}
            >
              {isLoading ? 'Processing...' : 'Compare Technologies'}
            </Button>
          </Box>
        </CardContent>
      </Card>

      {/* Error */}
      {error && (
        <Alert severity="error" sx={{ mb: 3 }} onClose={() => setError(null)}>
          {error}
        </Alert>
      )}

      {/* Loading */}
      {isLoading && (
        <Card sx={{ mb: 3 }}>
          <CardContent>
            <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 2 }}>
              <CircularProgress size={60} />
              <Typography variant="h6">Processing through all technology configurations...</Typography>
              <Typography variant="body2" color="text.secondary">
                This may take a moment as we process through Textract, LayoutLM, and GLM
              </Typography>
            </Box>
          </CardContent>
        </Card>
      )}

      {/* Results */}
      {result && (
        <>
          {/* Improvement Summary */}
          {result.improvement_summary && (
            <Box sx={{ mb: 3 }}>
              <ImprovementSummary summary={result.improvement_summary} />
            </Box>
          )}

          {/* Mode Cards */}
          <Typography variant="h6" gutterBottom sx={{ mt: 3 }}>
            Results by Processing Mode
          </Typography>
          <Grid container spacing={2} sx={{ mb: 3 }}>
            {modes.map(mode => (
              result.mode_results[mode] && (
                <Grid item xs={12} sm={6} md={3} key={mode}>
                  <ModeCard mode={mode} result={result.mode_results[mode]} />
                </Grid>
              )
            ))}
          </Grid>

          {/* Tabs for detailed views */}
          <Paper sx={{ mt: 3 }}>
            <Tabs 
              value={activeTab} 
              onChange={(_, v) => setActiveTab(v)}
              variant="scrollable"
              scrollButtons="auto"
            >
              <Tab icon={<ImageIcon />} label="Image Comparison" iconPosition="start" />
              <Tab label="Field-by-Field Comparison" />
              <Tab label="Confidence Progression" />
              <Tab label="Raw Results" />
            </Tabs>

            <TabPanel value={activeTab} index={0}>
              <ImageComparisonGrid modeResults={result.mode_results} />
            </TabPanel>

            <TabPanel value={activeTab} index={1}>
              <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                Compare how each field is classified across different technology configurations.
              </Typography>
              <FieldComparisonTable comparisons={result.field_comparisons} />
            </TabPanel>

            <TabPanel value={activeTab} index={2}>
              <Grid container spacing={3}>
                <Grid item xs={12} md={6}>
                  <Paper elevation={2} sx={{ p: 2 }}>
                    <ConfidenceProgressionChart 
                      data={result.improvement_summary.confidence_progression || []} 
                    />
                  </Paper>
                </Grid>
                <Grid item xs={12} md={6}>
                  <Paper elevation={2} sx={{ p: 2 }}>
                    <Typography variant="subtitle2" gutterBottom>Unknown Field Reduction</Typography>
                    {(result.improvement_summary.unknown_reduction || []).map((item, index) => (
                      <Box key={item.mode} sx={{ mb: 2 }}>
                        <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.5 }}>
                          <Typography variant="caption">{MODE_NAMES[item.mode] || item.mode}</Typography>
                          <Typography variant="caption">
                            {item.unknown_count} unknown ({(item.unknown_rate * 100).toFixed(1)}%)
                          </Typography>
                        </Box>
                        <LinearProgress 
                          variant="determinate" 
                          value={(1 - item.unknown_rate) * 100} 
                          color="success"
                          sx={{ height: 10, borderRadius: 5 }}
                        />
                      </Box>
                    ))}
                  </Paper>
                </Grid>
              </Grid>
            </TabPanel>

            <TabPanel value={activeTab} index={3}>
              <Accordion>
                <AccordionSummary expandIcon={<ExpandMore />}>
                  <Typography>Raw JSON Response</Typography>
                </AccordionSummary>
                <AccordionDetails>
                  <Box 
                    component="pre" 
                    sx={{ 
                      p: 2, 
                      bgcolor: 'grey.900', 
                      color: 'grey.100',
                      borderRadius: 1,
                      overflow: 'auto',
                      maxHeight: 500,
                      fontSize: '0.75rem'
                    }}
                  >
                    {JSON.stringify(result, null, 2)}
                  </Box>
                </AccordionDetails>
              </Accordion>
            </TabPanel>
          </Paper>
        </>
      )}
    </Box>
  )
}

