"""
Configuration management for the Court Forms OCR Workflow application.
Loads AWS credentials and settings from environment variables.
"""
import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()


class Config:
    """Configuration class for AWS credentials and settings."""
    
    # AWS Credentials
    AWS_PROFILE: Optional[str] = os.getenv('AWS_PROFILE')
    AWS_ACCESS_KEY_ID: Optional[str] = os.getenv('AWS_ACCESS_KEY_ID')
    AWS_SECRET_ACCESS_KEY: Optional[str] = os.getenv('AWS_SECRET_ACCESS_KEY')
    AWS_SESSION_TOKEN: Optional[str] = os.getenv('AWS_SESSION_TOKEN')
    AWS_REGION: str = os.getenv('AWS_REGION', 'us-east-1')
    
    # Firecrawl Configuration
    FIRECRAWL_API_KEY: Optional[str] = os.getenv('FIRECRAWL_API_KEY')
    FIRECRAWL_BASE_URL: str = os.getenv('FIRECRAWL_BASE_URL', 'https://api.firecrawl.dev')
    
    # Rate Limiting
    MAX_TOTAL_CALLS: int = int(os.getenv('MAX_TOTAL_CALLS', '50'))
    ENABLE_RATE_LIMITING: bool = os.getenv('ENABLE_RATE_LIMITING', 'true').lower() == 'true'
    
    # Firecrawl Scraping Options
    FIRECRAWL_ONLY_MAIN_CONTENT: Optional[bool] = os.getenv('FIRECRAWL_ONLY_MAIN_CONTENT', 'true').lower() == 'true' if os.getenv('FIRECRAWL_ONLY_MAIN_CONTENT') else None
    FIRECRAWL_TIMEOUT: Optional[int] = int(os.getenv('FIRECRAWL_TIMEOUT', '30000')) if os.getenv('FIRECRAWL_TIMEOUT') else None
    
    # Image Processing
    # Set to 0 or negative for unlimited images
    SAMPLE_IMAGES_COUNT: int = int(os.getenv('SAMPLE_IMAGES_COUNT', '0'))  # 0 = save ALL images
    IMAGES_OUTPUT_DIR: str = os.getenv('IMAGES_OUTPUT_DIR', 'output/images')
    
    # API Settings
    API_HOST: str = os.getenv('API_HOST', '0.0.0.0')
    API_PORT: int = int(os.getenv('API_PORT', '8000'))
    CORS_ORIGINS: list = os.getenv('CORS_ORIGINS', 'http://localhost:3000').split(',')
    
    @classmethod
    def validate(cls) -> bool:
        """
        Validate that required configuration is present.
        """
        if not cls.FIRECRAWL_API_KEY:
            raise ValueError("FIRECRAWL_API_KEY environment variable is required.")
        
        if not cls.AWS_PROFILE and (not cls.AWS_ACCESS_KEY_ID or not cls.AWS_SECRET_ACCESS_KEY):
            raise ValueError(
                "AWS credentials not found. Please set either:\n"
                "  - AWS_PROFILE environment variable (for profile-based auth), or\n"
                "  - AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables"
            )
        
        # Check if temporary credentials (ASIA) are used without session token
        if cls.AWS_ACCESS_KEY_ID and cls.AWS_ACCESS_KEY_ID.startswith('ASIA'):
            if not cls.AWS_SESSION_TOKEN:
                raise ValueError(
                    "Temporary credentials (ASIA) detected but AWS_SESSION_TOKEN is not set.\n"
                    "Temporary credentials require a session token to work."
                )
        
        if not cls.AWS_REGION:
            raise ValueError("AWS_REGION environment variable is required.")
        return True
    
    @classmethod
    def get_boto3_config(cls) -> dict:
        """
        Get AWS configuration dictionary for boto3.
        """
        config = {'region_name': cls.AWS_REGION}
        
        if cls.AWS_PROFILE:
            return {'profile_name': cls.AWS_PROFILE, 'region_name': cls.AWS_REGION}
        elif cls.AWS_ACCESS_KEY_ID and cls.AWS_SECRET_ACCESS_KEY:
            config.update({
                'aws_access_key_id': cls.AWS_ACCESS_KEY_ID,
                'aws_secret_access_key': cls.AWS_SECRET_ACCESS_KEY
            })
            if cls.AWS_SESSION_TOKEN:
                config['aws_session_token'] = cls.AWS_SESSION_TOKEN
        
        return config




