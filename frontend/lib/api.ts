/**
 * API client for backend communication.
 */
import axios from 'axios'

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8001'

// #region agent log
fetch('http://127.0.0.1:7252/ingest/564ebba2-2403-423b-931f-138186fce4fd',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'api.ts:6',message:'API_URL configured',data:{apiUrl:API_URL,envUrl:process.env.NEXT_PUBLIC_API_URL},timestamp:Date.now(),sessionId:'debug-session',runId:'run1',hypothesisId:'B'})}).catch(()=>{});
// #endregion

const apiClient = axios.create({
  baseURL: API_URL,
  headers: {
    'Content-Type': 'application/json',
  },
})

// #region agent log
apiClient.interceptors.request.use(config => {fetch('http://127.0.0.1:7252/ingest/564ebba2-2403-423b-931f-138186fce4fd',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'api.ts:15',message:'Axios request starting',data:{url:config.url,baseURL:config.baseURL,method:config.method,fullUrl:`${config.baseURL}${config.url}`},timestamp:Date.now(),sessionId:'debug-session',runId:'run1',hypothesisId:'B'})}).catch(()=>{}); return config;}, error => {fetch('http://127.0.0.1:7252/ingest/564ebba2-2403-423b-931f-138186fce4fd',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'api.ts:15',message:'Axios request error',data:{error:error?.message,stack:error?.stack},timestamp:Date.now(),sessionId:'debug-session',runId:'run1',hypothesisId:'B'})}).catch(()=>{}); return Promise.reject(error);});
apiClient.interceptors.response.use(response => {fetch('http://127.0.0.1:7252/ingest/564ebba2-2403-423b-931f-138186fce4fd',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'api.ts:15',message:'Axios response received',data:{status:response.status,url:response.config.url},timestamp:Date.now(),sessionId:'debug-session',runId:'run1',hypothesisId:'B'})}).catch(()=>{}); return response;}, error => {fetch('http://127.0.0.1:7252/ingest/564ebba2-2403-423b-931f-138186fce4fd',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'api.ts:15',message:'Axios response error',data:{message:error?.message,code:error?.code,responseStatus:error?.response?.status,responseData:error?.response?.data,url:error?.config?.url,baseURL:error?.config?.baseURL,fullUrl:error?.config?.baseURL?`${error.config.baseURL}${error.config.url}`:null},timestamp:Date.now(),sessionId:'debug-session',runId:'run1',hypothesisId:'A,B,C'})}).catch(()=>{}); return Promise.reject(error);});
// #endregion

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

export interface FormField {
  field_type: string
  label_text: string
  value_text?: string
  bounding_box: {
    left: number
    top: number
    width: number
    height: number
  }
  confidence: number
  field_confidence: number
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
  form_fields?: FormField[]
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
    // #region agent log
    fetch('http://127.0.0.1:7252/ingest/564ebba2-2403-423b-931f-138186fce4fd',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'api.ts:97',message:'scrapeForms called',data:{request,apiUrl:API_URL},timestamp:Date.now(),sessionId:'debug-session',runId:'run1',hypothesisId:'A,B'})}).catch(()=>{});
    // #endregion
    try {
      const response = await apiClient.post('/api/forms/scrape', request)
      // #region agent log
      fetch('http://127.0.0.1:7252/ingest/564ebba2-2403-423b-931f-138186fce4fd',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'api.ts:100',message:'scrapeForms success',data:{jobId:response.data?.job_id},timestamp:Date.now(),sessionId:'debug-session',runId:'run1',hypothesisId:'A,B'})}).catch(()=>{});
      // #endregion
      return response.data
    } catch (error: any) {
      // #region agent log
      fetch('http://127.0.0.1:7252/ingest/564ebba2-2403-423b-931f-138186fce4fd',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'api.ts:105',message:'scrapeForms error caught',data:{errorMessage:error?.message,errorCode:error?.code,isNetworkError:error?.message?.includes('Network Error'),responseStatus:error?.response?.status,responseData:error?.response?.data},timestamp:Date.now(),sessionId:'debug-session',runId:'run1',hypothesisId:'A,B,C'})}).catch(()=>{});
      // #endregion
      throw error
    }
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

