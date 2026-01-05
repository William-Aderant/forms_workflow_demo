/**
 * API client for backend communication.
 */
import axios from 'axios'

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

const apiClient = axios.create({
  baseURL: API_URL,
  headers: {
    'Content-Type': 'application/json',
  },
})

// Types
export interface FormInfo {
  id: string
  name: string
  url: string
  status: 'pending' | 'processing' | 'completed' | 'error'
  ocr_confidence?: number
  text_length?: number
  created_at: string
  error_message?: string
}

export interface FormDetail extends FormInfo {
  markdown_text?: string
  blocks_count?: number
}

export interface ImageMetadata {
  image_path: string
  form_id: string
  form_name: string
  boxes: Array<{
    block_type: string
    text: string
    confidence: number
    bounding_box: {
      left: number
      top: number
      width: number
      height: number
    }
    pixel_coordinates?: {
      x1: number
      y1: number
      x2: number
      y2: number
    }
  }>
  created_at: string
}

export interface RateLimitStatus {
  total_calls: number
  max_calls: number
  remaining_calls: number
  calls_by_service: Record<string, number>
}

export interface ScrapeRequest {
  url?: string
}

export interface ScrapeResponse {
  message: string
  job_id?: string
  estimated_forms?: number
}

// API functions
export const api = {
  // Health check
  async healthCheck(): Promise<{ status: string; timestamp: string }> {
    const response = await apiClient.get('/api/health')
    return response.data
  },

  // Scrape forms
  async scrapeForms(request: ScrapeRequest = {}): Promise<ScrapeResponse> {
    const response = await apiClient.post('/api/forms/scrape', request)
    return response.data
  },

  // List all forms
  async listForms(): Promise<FormInfo[]> {
    const response = await apiClient.get('/api/forms')
    return response.data
  },

  // Get form details
  async getForm(formId: string): Promise<FormDetail> {
    const response = await apiClient.get(`/api/forms/${formId}`)
    return response.data
  },

  // Get form images
  async getFormImages(formId: string): Promise<ImageMetadata[]> {
    const response = await apiClient.get(`/api/forms/${formId}/images`)
    return response.data
  },

  // Get rate limit status
  async getRateLimitStatus(): Promise<RateLimitStatus> {
    const response = await apiClient.get('/api/rate-limit')
    return response.data
  },

  // Subscribe to scraping progress via Server-Sent Events
  subscribeToProgress(
    jobId: string,
    onMessage: (event: string, data: any) => void,
    onError?: (error: Event) => void
  ): () => void {
    const eventSource = new EventSource(`${API_URL}/api/forms/scrape/${jobId}/stream`)

    eventSource.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data)
        onMessage('message', data)
      } catch (e) {
        console.error('Failed to parse SSE message:', e)
      }
    }

    // Handle different event types
    eventSource.addEventListener('start', (event: any) => {
      try {
        const data = JSON.parse(event.data)
        onMessage('start', data)
      } catch (e) {
        console.error('Failed to parse SSE start event:', e)
      }
    })

    eventSource.addEventListener('progress', (event: any) => {
      try {
        const data = JSON.parse(event.data)
        onMessage('progress', data)
      } catch (e) {
        console.error('Failed to parse SSE progress event:', e)
      }
    })

    eventSource.addEventListener('form_start', (event: any) => {
      try {
        const data = JSON.parse(event.data)
        onMessage('form_start', data)
      } catch (e) {
        console.error('Failed to parse SSE form_start event:', e)
      }
    })

    eventSource.addEventListener('form_processing', (event: any) => {
      try {
        const data = JSON.parse(event.data)
        onMessage('form_processing', data)
      } catch (e) {
        console.error('Failed to parse SSE form_processing event:', e)
      }
    })

    eventSource.addEventListener('form_complete', (event: any) => {
      try {
        const data = JSON.parse(event.data)
        onMessage('form_complete', data)
      } catch (e) {
        console.error('Failed to parse SSE form_complete event:', e)
      }
    })

    eventSource.addEventListener('form_error', (event: any) => {
      try {
        const data = JSON.parse(event.data)
        onMessage('form_error', data)
      } catch (e) {
        console.error('Failed to parse SSE form_error event:', e)
      }
    })

    eventSource.addEventListener('complete', (event: any) => {
      try {
        const data = JSON.parse(event.data)
        onMessage('complete', data)
        eventSource.close()
      } catch (e) {
        console.error('Failed to parse SSE complete event:', e)
      }
    })

    eventSource.addEventListener('error', (event: any) => {
      try {
        const data = JSON.parse(event.data)
        onMessage('error', data)
        eventSource.close()
      } catch (e) {
        console.error('Failed to parse SSE error event:', e)
      }
    })

    if (onError) {
      eventSource.onerror = onError
    }

    // Return cleanup function
    return () => {
      eventSource.close()
    }
  },
}

export default api

