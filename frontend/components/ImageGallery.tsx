'use client'

import React, { useState } from 'react'
import {
  Box,
  Card,
  CardContent,
  Typography,
  ImageList,
  ImageListItem,
  ImageListItemBar,
  IconButton,
  Dialog,
  DialogContent,
  DialogTitle,
  Chip,
  Tooltip,
} from '@mui/material'
import { Fullscreen, Download, Visibility, VisibilityOff } from '@mui/icons-material'
import { ImageMetadata } from '@/lib/api'
// Using regular img tag for local files served via API

interface ImageGalleryProps {
  images: ImageMetadata[]
  formName?: string
}

export default function ImageGallery({ images, formName }: ImageGalleryProps) {
  const [selectedImage, setSelectedImage] = useState<ImageMetadata | null>(null)
  const [showBoxes, setShowBoxes] = useState(true)

  const handleImageClick = (image: ImageMetadata) => {
    setSelectedImage(image)
  }

  const handleClose = () => {
    setSelectedImage(null)
  }

  const handleDownload = (imagePath: string, formName: string) => {
    // Create a temporary link to download the image
    const link = document.createElement('a')
    link.href = imagePath
    link.download = `${formName}_annotated.png`
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
  }

  if (images.length === 0) {
    return (
      <Card>
        <CardContent>
          <Typography variant="body1" color="text.secondary" align="center">
            No annotated images available
          </Typography>
        </CardContent>
      </Card>
    )
  }

  return (
    <>
      <Card>
        <CardContent>
          <Box display="flex" alignItems="center" justifyContent="space-between" mb={2}>
            <Typography variant="h6" component="h2">
              Annotated Images {formName && `- ${formName}`}
            </Typography>
            <Box display="flex" gap={1} alignItems="center">
              <Chip
                label={`${images.length} image${images.length !== 1 ? 's' : ''}`}
                size="small"
                variant="outlined"
              />
              <Tooltip title={showBoxes ? 'Hide bounding boxes' : 'Show bounding boxes'}>
                <IconButton
                  size="small"
                  onClick={() => setShowBoxes(!showBoxes)}
                >
                  {showBoxes ? <VisibilityOff /> : <Visibility />}
                </IconButton>
              </Tooltip>
            </Box>
          </Box>

          <ImageList cols={2} gap={16}>
            {images.map((image, index) => (
              <ImageListItem key={index}>
                <Box
                  sx={{
                    position: 'relative',
                    width: '100%',
                    height: 300,
                    cursor: 'pointer',
                    '&:hover': {
                      opacity: 0.9,
                    },
                  }}
                  onClick={() => handleImageClick(image)}
                >
                  <img
                    src={`${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}/api/images/${image.image_path.split('/').pop()}`}
                    alt={`Annotated form ${index + 1}`}
                    style={{ width: '100%', height: '100%', objectFit: 'contain' }}
                  />
                </Box>
                <ImageListItemBar
                  title={`Image ${index + 1}`}
                  subtitle={
                    image.form_fields && image.form_fields.length > 0
                      ? `${image.form_fields.length} form fields`
                      : `${image.boxes.length} bounding boxes`
                  }
                  actionIcon={
                    <>
                      <IconButton
                        sx={{ color: 'rgba(255, 255, 255, 0.54)' }}
                        onClick={(e) => {
                          e.stopPropagation()
                          handleImageClick(image)
                        }}
                      >
                        <Fullscreen />
                      </IconButton>
                      <IconButton
                        sx={{ color: 'rgba(255, 255, 255, 0.54)' }}
                        onClick={(e) => {
                          e.stopPropagation()
                          handleDownload(image.image_path, image.form_name)
                        }}
                      >
                        <Download />
                      </IconButton>
                    </>
                  }
                />
              </ImageListItem>
            ))}
          </ImageList>
        </CardContent>
      </Card>

      <Dialog
        open={selectedImage !== null}
        onClose={handleClose}
        maxWidth="lg"
        fullWidth
      >
        {selectedImage && (
          <>
            <DialogTitle>
              <Box display="flex" justifyContent="space-between" alignItems="center">
                <Typography variant="h6">
                  {selectedImage.form_name} - Annotated Image
                </Typography>
                <IconButton onClick={handleClose} size="small">
                  <Fullscreen />
                </IconButton>
              </Box>
            </DialogTitle>
            <DialogContent>
              <Box mb={2}>
                <img
                  src={`${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}/api/images/${selectedImage.image_path.split('/').pop()}`}
                  alt={selectedImage.form_name}
                  style={{ width: '100%', height: 'auto' }}
                />
              </Box>
              {selectedImage.form_fields && selectedImage.form_fields.length > 0 ? (
                <Box>
                  <Typography variant="subtitle2" gutterBottom>
                    Form Fields ({selectedImage.form_fields.length})
                  </Typography>
                  <Box display="flex" flexWrap="wrap" gap={1} mt={1}>
                    {selectedImage.form_fields.map((field, index) => (
                      <Chip
                        key={index}
                        label={`${field.field_type.toUpperCase()}: ${field.label_text || 'N/A'}`}
                        size="small"
                        variant="outlined"
                        color="primary"
                        title={`Confidence: ${field.confidence.toFixed(1)}%, Field Type Confidence: ${(field.field_confidence * 100).toFixed(1)}%`}
                      />
                    ))}
                  </Box>
                </Box>
              ) : (
                <Box>
                  <Typography variant="subtitle2" gutterBottom>
                    Bounding Boxes ({selectedImage.boxes.length})
                  </Typography>
                  <Box display="flex" flexWrap="wrap" gap={1} mt={1}>
                    {selectedImage.boxes.slice(0, 20).map((box, index) => (
                      <Chip
                        key={index}
                        label={`${box.block_type}: ${box.text.substring(0, 30)}`}
                        size="small"
                        variant="outlined"
                      />
                    ))}
                    {selectedImage.boxes.length > 20 && (
                      <Chip
                        label={`+${selectedImage.boxes.length - 20} more`}
                        size="small"
                        variant="outlined"
                      />
                    )}
                  </Box>
                </Box>
              )}
            </DialogContent>
          </>
        )}
      </Dialog>
    </>
  )
}
