"""
Firecrawl service for scraping court forms website and extracting PDF links.
"""
import logging
import re
from typing import List, Optional, Dict, Any
from urllib.parse import urljoin, urlparse
# #region agent log
import json
import os
log_path = '/Users/william.holden/Documents/forms_workflow_demo/.cursor/debug.log'
os.makedirs(os.path.dirname(log_path), exist_ok=True)
with open(log_path, 'a') as f:
    f.write(json.dumps({"location":"firecrawl_service.py:11","message":"Attempting to inspect firecrawl module","data":{"step":"before_import"},"timestamp":int(__import__('time').time()*1000),"sessionId":"debug-session","runId":"run1","hypothesisId":"A"})+"\n")
# #endregion
try:
    from firecrawl import FirecrawlApp
    Firecrawl = FirecrawlApp  # Alias for compatibility
    # #region agent log
    log_path = '/Users/william.holden/Documents/forms_workflow_demo/.cursor/debug.log'
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, 'a') as f:
        f.write(json.dumps({"location":"firecrawl_service.py:17","message":"Successfully imported FirecrawlApp","data":{"type":str(type(Firecrawl))},"timestamp":int(__import__('time').time()*1000),"sessionId":"debug-session","runId":"run1","hypothesisId":"FIXED"})+"\n")
    # #endregion
except ImportError:
    # Fallback: try the old import pattern
    try:
        from firecrawl import Firecrawl
        # #region agent log
        log_path = '/Users/william.holden/Documents/forms_workflow_demo/.cursor/debug.log'
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, 'a') as f:
            f.write(json.dumps({"location":"firecrawl_service.py:25","message":"Using fallback Firecrawl import","data":{},"timestamp":int(__import__('time').time()*1000),"sessionId":"debug-session","runId":"run1","hypothesisId":"FALLBACK"})+"\n")
        # #endregion
    except ImportError as e:
        raise ImportError(f"Could not import Firecrawl or FirecrawlApp from firecrawl: {e}")

from app.config import Config
from app.utils.rate_limiter import RateLimiter
from app.utils.pdf_handler import PDFHandler

logger = logging.getLogger(__name__)


class FirecrawlService:
    """Service for scraping web pages and extracting PDF links using Firecrawl."""
    
    def __init__(self, rate_limiter: Optional[RateLimiter] = None):
        """
        Initialize Firecrawl service.
        
        Args:
            rate_limiter: Optional rate limiter instance
        """
        self.api_key = Config.FIRECRAWL_API_KEY
        self.rate_limiter = rate_limiter
        self.service_name = 'firecrawl'
        self.pdf_handler = PDFHandler()
        
        if not self.api_key:
            raise ValueError("FIRECRAWL_API_KEY is required")
        
        try:
            self.client = Firecrawl(api_key=self.api_key)
            # #region agent log
            with open(log_path, 'a') as f:
                f.write(json.dumps({"location":"firecrawl_service.py:64","message":"FirecrawlApp initialized","data":{"methods":str([m for m in dir(self.client) if not m.startswith('_')])},"timestamp":int(__import__('time').time()*1000),"sessionId":"debug-session","runId":"run1","hypothesisId":"API_METHODS"})+"\n")
            # #endregion
            logger.info("Initialized Firecrawl service")
        except Exception as e:
            logger.error(f"Failed to initialize Firecrawl client: {e}")
            raise
    
    def scrape_page(self, url: str) -> Optional[Dict[str, Any]]:
        """
        Scrape a web page using Firecrawl.
        
        Args:
            url: URL to scrape
        
        Returns:
            Dictionary with scraped content, or None if failed
        """
        # Check rate limits
        if self.rate_limiter:
            can_call, reason = self.rate_limiter.can_make_call(self.service_name)
            if not can_call:
                logger.warning(f"Rate limit exceeded: {reason}")
                return None
            
            # Wait if needed
            self.rate_limiter.wait_if_needed(self.service_name, min_delay_seconds=1.0)
        
        try:
            logger.info(f"Scraping page: {url}")
            
            # Call Firecrawl API - try different method names and signatures
            # #region agent log
            import inspect
            scrape_url_sig = None
            if hasattr(self.client, 'scrape_url'):
                try:
                    scrape_url_sig = str(inspect.signature(self.client.scrape_url))
                except:
                    pass
            with open(log_path, 'a') as f:
                f.write(json.dumps({"location":"firecrawl_service.py:103","message":"Attempting to call Firecrawl API","data":{"url":url,"has_scrape":hasattr(self.client, 'scrape'),"has_scrapeUrl":hasattr(self.client, 'scrapeUrl'),"has_scrape_url":hasattr(self.client, 'scrape_url'),"scrape_url_sig":scrape_url_sig},"timestamp":int(__import__('time').time()*1000),"sessionId":"debug-session","runId":"run1","hypothesisId":"API_CALL"})+"\n")
            # #endregion
            
            response = None
            if hasattr(self.client, 'scrape_url'):
                # FirecrawlApp.scrape_url() accepts url and optional pageOptions parameter
                # Build pageOptions if needed
                page_options = {}
                if Config.FIRECRAWL_ONLY_MAIN_CONTENT:
                    page_options['onlyMainContent'] = True
                if Config.FIRECRAWL_TIMEOUT:
                    page_options['timeout'] = Config.FIRECRAWL_TIMEOUT
                
                # #region agent log
                with open(log_path, 'a') as f:
                    f.write(json.dumps({"location":"firecrawl_service.py:110","message":"Calling scrape_url with pageOptions","data":{"url":url,"page_options":page_options},"timestamp":int(__import__('time').time()*1000),"sessionId":"debug-session","runId":"run1","hypothesisId":"API_CALL"})+"\n")
                # #endregion
                
                if page_options:
                    response = self.client.scrape_url(url, pageOptions=page_options)
                else:
                    response = self.client.scrape_url(url)
            elif hasattr(self.client, 'scrapeUrl'):
                response = self.client.scrapeUrl(url)
            elif hasattr(self.client, 'scrape'):
                response = self.client.scrape(url)
            else:
                raise AttributeError(f"FirecrawlApp has no scrape method. Available methods: {[m for m in dir(self.client) if not m.startswith('_')]}")
            
            # Record the call
            if self.rate_limiter:
                self.rate_limiter.record_call(self.service_name)
            
            # Extract content
            content = ""
            if response:
                if hasattr(response, 'markdown'):
                    content = response.markdown
                elif hasattr(response, 'get') and callable(response.get):
                    content = response.get('markdown') or response.get('content') or ''
                elif isinstance(response, dict):
                    content = response.get('markdown') or response.get('content') or ''
                elif isinstance(response, str):
                    content = response
            
            return {
                'url': url,
                'content': content,
                'success': bool(content)
            }
            
        except Exception as e:
            logger.error(f"Error scraping {url} with Firecrawl: {e}")
            if self.rate_limiter:
                self.rate_limiter.record_call(self.service_name)  # Still record failed call
            return None
    
    def extract_pdf_links(self, content: str, base_url: str) -> List[str]:
        """
        Extract PDF links from scraped content.
        
        Args:
            content: Scraped HTML/markdown content
            base_url: Base URL for resolving relative links
        
        Returns:
            List of PDF URLs
        """
        pdf_links = []
        
        # Pattern to match PDF links in markdown and HTML
        # Markdown: [text](url.pdf) or [text](url.pdf "title")
        # HTML: <a href="url.pdf"> or href="url.pdf"
        patterns = [
            r'\[([^\]]+)\]\(([^)]+\.pdf[^)]*)\)',  # Markdown links
            r'href=["\']([^"\']+\.pdf[^"\']*)["\']',  # HTML href
            r'https?://[^\s<>"]+\.pdf',  # Direct PDF URLs
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                if pattern.startswith('href='):
                    url = match.group(1)
                elif pattern.startswith('['):
                    url = match.group(2)
                else:
                    url = match.group(0)
                
                # Clean up URL (remove quotes, fragments, etc.)
                url = url.strip('"\'')
                url = url.split('#')[0]  # Remove fragment
                url = url.split('?')[0]  # Remove query params for now
                
                # Resolve relative URLs
                if url.startswith('/'):
                    url = urljoin(base_url, url)
                elif not url.startswith('http'):
                    url = urljoin(base_url, url)
                
                # Verify it's a PDF
                if url.lower().endswith('.pdf') or 'pdf' in url.lower():
                    if url not in pdf_links:
                        pdf_links.append(url)
        
        logger.info(f"Extracted {len(pdf_links)} PDF links")
        return pdf_links
    
    def scrape_and_extract_pdfs(self, url: str) -> List[Dict[str, Any]]:
        """
        Scrape court forms website and extract all PDF links.
        Handles the multi-step flow:
        1. Start at forms listing page (https://selfhelp.courts.ca.gov/find-forms/all)
        2. Extract all form links from listing pages (handle pagination)
        3. For each form, navigate to detail page and extract PDF download link
        4. Download PDFs in memory
        
        Args:
            url: Starting URL (should be https://selfhelp.courts.ca.gov/find-forms/all)
        
        Returns:
            List of dictionaries with PDF info (url, name, bytes)
        """
        # Use the correct starting URL
        if 'selfhelp.courts.ca.gov' not in url:
            url = 'https://selfhelp.courts.ca.gov/find-forms/all'
        
        logger.info(f"Starting forms scraping from: {url}")
        
        # Step 1: Scrape page 3 of the forms listing
        # The PDF download links are already on the listing page!
        all_pdf_urls = []
        all_form_names = {}
        
        for page_num in [3]:
            # Construct URL with page parameter
            if '?' in url:
                page_url = f"{url}&page={page_num}"
            else:
                page_url = f"{url}?page={page_num}"
            
            logger.info(f"Scraping forms listing page {page_num}: {page_url}")
            result = self.scrape_page(page_url)
            
            if not result or not result.get('success'):
                logger.warning(f"Failed to scrape forms listing page {page_num}: {page_url}")
                continue
            
            # #region agent log
            with open(log_path, 'a') as f:
                f.write(json.dumps({"location":"firecrawl_service.py:240","message":"Scraped forms listing page","data":{"page":page_num,"content_length":len(result['content'])},"timestamp":int(__import__('time').time()*1000),"sessionId":"debug-session","runId":"run1","hypothesisId":"LISTING_PAGE"})+"\n")
            # #endregion
            
            # Step 2: Extract PDF download links directly from the listing page
            # PDFs are listed as [Download](https://www.courts.ca.gov/documents/xxx.pdf) on the same page
            logger.info(f"Extracting PDF download links from page {page_num}")
            page_pdf_urls = self._extract_pdf_download_links(result['content'], page_url)
            all_pdf_urls.extend(page_pdf_urls)
            
            # Also extract form names from the listing page for better naming
            page_form_names = self._extract_form_names_from_listing(result['content'])
            all_form_names.update(page_form_names)
            logger.info(f"Extracted {len(page_form_names)} form names from page {page_num}")
        
        # Remove duplicates while preserving order
        seen_urls = set()
        pdf_urls = []
        for url in all_pdf_urls:
            if url not in seen_urls:
                pdf_urls.append(url)
                seen_urls.add(url)
        
        form_names = all_form_names
        logger.info(f"Extracted {len(form_names)} total form names from all pages")
        
        logger.info(f"Found {len(pdf_urls)} unique PDF URLs from page 3")
        
        # Log first few PDFs for debugging
        if pdf_urls:
            logger.info(f"Sample PDF URLs: {pdf_urls[:5]}")
        else:
            logger.warning("No PDF URLs found from page 3!")
        
        # Step 3: Return PDF URLs and names (don't download yet - will be done sequentially)
        pdf_info_list = []
        max_pdfs = min(50, len(pdf_urls))  # Limit to 50 PDFs due to rate limiting
        
        logger.info(f"Found {max_pdfs} PDFs to process (out of {len(pdf_urls)} found)")
        
        for pdf_url in pdf_urls[:max_pdfs]:
            # Extract form name from URL
            form_name = self._extract_form_name(pdf_url)
            
            # Try to match with form names from listing if available
            if form_names:
                # Extract form code from PDF URL (e.g., adopt200.pdf -> ADOPT-200)
                url_lower = pdf_url.lower()
                for form_code, name in form_names.items():
                    form_code_clean = form_code.lower().replace('-', '')
                    url_clean = url_lower.replace('.pdf', '').replace('_', '').replace('/', '')
                    if form_code_clean in url_clean:
                        form_name = f"{form_code} - {name}"
                        break
            
            pdf_info_list.append({
                'url': pdf_url,
                'name': form_name
            })
        
        logger.info(f"Prepared {len(pdf_info_list)} PDFs for sequential processing")
        return pdf_info_list
    
    def _extract_form_detail_links(self, content: str, base_url: str) -> List[str]:
        """
        Extract links to form detail pages from the forms listing page.
        Forms have links like "See form info" that go to detail pages.
        Based on the website structure, forms are listed with "See form info" links.
        
        Args:
            content: Scraped HTML/markdown content
            base_url: Base URL for resolving relative links
        
        Returns:
            List of form detail page URLs
        """
        form_links = []
        seen_urls = set()
        
        # #region agent log
        with open(log_path, 'a') as f:
            f.write(json.dumps({"location":"firecrawl_service.py:315","message":"Extracting form detail links","data":{"content_length":len(content),"content_preview":content[:500]},"timestamp":int(__import__('time').time()*1000),"sessionId":"debug-session","runId":"run1","hypothesisId":"FORM_LINKS"})+"\n")
        # #endregion
        
        # Pattern to match markdown links - extract URL from [text](url) format
        markdown_pattern = r'\[([^\]]+)\]\(([^)]+)\)'
        for match in re.finditer(markdown_pattern, content):
            link_text = match.group(1)
            url = match.group(2)
            
            # #region agent log
            with open(log_path, 'a') as f:
                f.write(json.dumps({"location":"firecrawl_service.py:325","message":"Found markdown link","data":{"link_text":link_text,"url":url},"timestamp":int(__import__('time').time()*1000),"sessionId":"debug-session","runId":"run1","hypothesisId":"FORM_LINKS"})+"\n")
            # #endregion
            
            link_text_lower = link_text.lower()
            
            # Skip navigation and non-form links
            skip_texts = ['skip to', 'next', 'previous', 'page', 'current page', 'go to page', 'go to next', 'search by', 'organize by']
            if any(skip in link_text_lower for skip in skip_texts):
                continue
            
            # Look for "See form info" links or form code patterns
            # Forms have links like "See form info" or contain form codes like ADOPT-200
            # Also look for any link that goes to a form detail page (not listing or pagination)
            is_form_link = (
                'see form' in link_text_lower or 
                'form info' in link_text_lower or
                re.search(r'[A-Z]{2,}-\d+', link_text) or  # Form code pattern like ADOPT-200
                ('download' in link_text_lower and '.pdf' in url.lower())
            )
            
            # Also check if URL looks like a form detail page (not listing page)
            is_form_url = (
                '/find-forms/' in url and 
                '/all' not in url and 
                '?page=' not in url and
                url.count('/') >= 4  # Form detail pages have more path segments
            )
            
            if is_form_link or is_form_url:
                # Clean up URL
                url = url.strip('"\'')
                url = url.split('#')[0]  # Remove fragment
                url = url.split('?')[0]  # Remove query params for now
                
                # Resolve relative URLs
                if url.startswith('/'):
                    url = urljoin('https://selfhelp.courts.ca.gov', url)
                elif not url.startswith('http'):
                    url = urljoin(base_url, url)
                
                # Only include valid form detail page URLs (not listing pages)
                if ('selfhelp.courts.ca.gov/find-forms' in url and 
                    url not in seen_urls and 
                    url != base_url and
                    '/all' not in url and
                    '?page=' not in url):  # Exclude listing pages and pagination
                    form_links.append(url)
                    seen_urls.add(url)
                    # #region agent log
                    with open(log_path, 'a') as f:
                        f.write(json.dumps({"location":"firecrawl_service.py:355","message":"Added form link","data":{"url":url,"link_text":link_text},"timestamp":int(__import__('time').time()*1000),"sessionId":"debug-session","runId":"run1","hypothesisId":"FORM_LINKS"})+"\n")
                    # #endregion
        
        # Also try HTML href patterns for form links
        href_pattern = r'href=["\']([^"\']*find-forms/[^"\']*)["\']'
        for match in re.finditer(href_pattern, content, re.IGNORECASE):
            url = match.group(1).strip('"\'')
            url = url.split('#')[0]
            
            if url.startswith('/'):
                url = urljoin('https://selfhelp.courts.ca.gov', url)
            elif not url.startswith('http'):
                url = urljoin(base_url, url)
            
            if ('selfhelp.courts.ca.gov/find-forms' in url and 
                url not in seen_urls and 
                url != base_url and
                '/all' not in url and
                '?page=' not in url):
                form_links.append(url)
                seen_urls.add(url)
        
        # Fallback: Extract ALL URLs that look like form detail pages
        # Form detail pages typically have paths like /find-forms/[form-code] or /find-forms/[category]/[form]
        if len(form_links) == 0:
            logger.warning("No form links found with standard patterns, trying fallback extraction")
            # Look for any URL pattern that matches form detail pages
            all_urls_pattern = r'https?://[^\s<>"\'\)]+selfhelp\.courts\.ca\.gov/find-forms/[^\s<>"\'\)]+'
            for match in re.finditer(all_urls_pattern, content, re.IGNORECASE):
                url = match.group(0).strip('"\'')
                url = url.split('#')[0]
                url = url.split('?')[0]
                
                # Exclude listing pages and pagination
                if ('/all' not in url and 
                    '?page=' not in url and
                    url not in seen_urls and
                    url != base_url):
                    form_links.append(url)
                    seen_urls.add(url)
                    # #region agent log
                    with open(log_path, 'a') as f:
                        f.write(json.dumps({"location":"firecrawl_service.py:410","message":"Added form link via fallback","data":{"url":url},"timestamp":int(__import__('time').time()*1000),"sessionId":"debug-session","runId":"run1","hypothesisId":"FORM_LINKS_FALLBACK"})+"\n")
                    # #endregion
        
        logger.info(f"Extracted {len(form_links)} form detail links")
        return form_links
    
    def _extract_pdf_download_links(self, content: str, base_url: str) -> List[str]:
        """
        Extract PDF download links from the listing page or form detail page.
        Looks for links to PDFs on courts.ca.gov/documents/ domain.
        
        Args:
            content: Scraped HTML/markdown content
            base_url: Base URL for resolving relative links
        
        Returns:
            List of PDF download URLs
        """
        pdf_links = []
        seen_urls = set()
        
        # #region agent log
        with open(log_path, 'a') as f:
            f.write(json.dumps({"location":"firecrawl_service.py:406","message":"Extracting PDF links","data":{"content_length":len(content)},"timestamp":int(__import__('time').time()*1000),"sessionId":"debug-session","runId":"run1","hypothesisId":"PDF_LINKS"})+"\n")
        # #endregion
        
        # Pattern 1: Markdown links - [Download](url.pdf) - this is the main pattern on listing page
        markdown_pattern = r'\[([^\]]*[Dd]ownload[^\]]*|[^\]]+)\]\(([^)]+\.pdf[^)]*)\)'
        for match in re.finditer(markdown_pattern, content, re.IGNORECASE):
            link_text = match.group(1)
            url = match.group(2)
            
            # #region agent log
            with open(log_path, 'a') as f:
                f.write(json.dumps({"location":"firecrawl_service.py:416","message":"Found markdown PDF link","data":{"link_text":link_text,"url":url},"timestamp":int(__import__('time').time()*1000),"sessionId":"debug-session","runId":"run1","hypothesisId":"PDF_LINKS"})+"\n")
            # #endregion
            
            url = url.strip('"\'')
            url = url.split('#')[0]
            url = url.split('?')[0]
            
            # Resolve relative URLs
            if url.startswith('/'):
                url = urljoin('https://www.courts.ca.gov', url)
            elif not url.startswith('http'):
                url = urljoin(base_url, url)
            
            # Verify it's a PDF and from courts.ca.gov
            if url.lower().endswith('.pdf') and 'courts.ca.gov' in url:
                if url not in seen_urls:
                    pdf_links.append(url)
                    seen_urls.add(url)
                    # #region agent log
                    with open(log_path, 'a') as f:
                        f.write(json.dumps({"location":"firecrawl_service.py:432","message":"Added PDF link","data":{"url":url},"timestamp":int(__import__('time').time()*1000),"sessionId":"debug-session","runId":"run1","hypothesisId":"PDF_LINKS"})+"\n")
                    # #endregion
        
        # Pattern 2: Direct PDF URLs in text (courts.ca.gov/documents/...pdf)
        direct_pattern = r'https?://[^"\'\s<>\)]*courts\.ca\.gov[^"\'\s<>\)]*\.pdf'
        for match in re.finditer(direct_pattern, content, re.IGNORECASE):
            url = match.group(0).strip('"\'')
            url = url.split('#')[0]
            url = url.split('?')[0]
            if url not in seen_urls and 'courts.ca.gov' in url:
                pdf_links.append(url)
                seen_urls.add(url)
        
        # Pattern 3: HTML href links
        href_pattern = r'href=["\']([^"\']*courts\.ca\.gov[^"\']*\.pdf[^"\']*)["\']'
        for match in re.finditer(href_pattern, content, re.IGNORECASE):
            url = match.group(1).strip('"\'')
            url = url.split('#')[0]
            url = url.split('?')[0]
            
            if url.startswith('/'):
                url = urljoin('https://www.courts.ca.gov', url)
            elif not url.startswith('http'):
                url = urljoin(base_url, url)
            
            if url.lower().endswith('.pdf') and 'courts.ca.gov' in url:
                if url not in seen_urls:
                    pdf_links.append(url)
                    seen_urls.add(url)
        
        logger.info(f"Extracted {len(pdf_links)} PDF download links")
        return pdf_links
    
    def _extract_form_names_from_listing(self, content: str) -> Dict[str, str]:
        """
        Extract form names from the listing page content.
        Forms are listed with their codes and names.
        
        Args:
            content: Scraped HTML/markdown content
        
        Returns:
            Dictionary mapping form codes to form names
        """
        form_names = {}
        
        # Pattern: Extract form code and name from listing
        # Format: [FORM-CODE \*\\n\\n\\nForm Name\\n\\n\\nEffective: ...](url)
        # Example: [ADOPT-200 \*\\n\\n\\nAdoption Request\\n\\n\\nEffective: ...]
        form_pattern = r'\[([A-Z]{2,}-\d+[^\]]*?)\\n\\n\\n([^\\n]+?)\\n\\n\\nEffective'
        for match in re.finditer(form_pattern, content):
            form_code = match.group(1).split()[0].strip()  # Get just the code part (remove asterisk)
            form_name = match.group(2).strip()
            if form_code and form_name:
                form_names[form_code] = form_name
        
        # #region agent log
        with open(log_path, 'a') as f:
            f.write(json.dumps({"location":"firecrawl_service.py:540","message":"Extracted form names","data":{"count":len(form_names),"sample":dict(list(form_names.items())[:3])},"timestamp":int(__import__('time').time()*1000),"sessionId":"debug-session","runId":"run1","hypothesisId":"FORM_NAMES"})+"\n")
        # #endregion
        
        return form_names
    
    def _extract_next_page_link(self, content: str, current_url: str) -> Optional[str]:
        """
        Extract the "next page" link from pagination.
        
        Args:
            content: Scraped HTML/markdown content
            current_url: Current page URL
        
        Returns:
            Next page URL or None
        """
        # Extract current page number from URL
        current_page = 1
        page_match = re.search(r'[?&]page=(\d+)', current_url)
        if page_match:
            current_page = int(page_match.group(1))
        
        # Look for "Next page" or "Page X" links in markdown
        # Pattern: [Next page Next ›](url) or [Page X](url)
        next_patterns = [
            r'\[Next[^\]]*\]\(([^)]+)\)',  # Markdown next link
            r'\[Page\s+\d+[^\]]*\]\(([^)]+)\)',  # Page number link
        ]
        
        for pattern in next_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                url = match.group(1).strip('"\'')
                url = url.split('#')[0]
                
                # Check if this is actually a next page (higher page number)
                page_match = re.search(r'[?&]page=(\d+)', url)
                if page_match:
                    next_page_num = int(page_match.group(1))
                    if next_page_num > current_page:
                        if url.startswith('/'):
                            url = urljoin('https://selfhelp.courts.ca.gov', url)
                        elif not url.startswith('http'):
                            url = urljoin(current_url, url)
                        
                        if url != current_url:
                            return url
        
        # Fallback: construct next page URL
        if 'page=' in current_url:
            next_page = current_page + 1
            next_url = re.sub(r'[?&]page=\d+', f'?page={next_page}', current_url)
            if '?' not in next_url:
                next_url = f"{current_url}?page={next_page}"
            return next_url
        else:
            # First page, construct page 2
            separator = '&' if '?' in current_url else '?'
            return f"{current_url}{separator}page=2"
    
    def _extract_form_name(self, url: str) -> str:
        """
        Extract form name from URL.
        
        Args:
            url: PDF URL
        
        Returns:
            Form name
        """
        parsed = urlparse(url)
        filename = parsed.path.split('/')[-1]
        
        # Remove .pdf extension and clean up
        name = filename.replace('.pdf', '').replace('.PDF', '')
        name = name.replace('_', ' ').replace('-', ' ')
        name = ' '.join(word.capitalize() for word in name.split())
        
        return name or "Unknown Form"

