import google.generativeai as genai
import base64
from app.config import GENAI_API_KEY

genai.configure(api_key=GENAI_API_KEY)


def analyze_medical_scan_with_context(image_data: bytes = None, mime_type: str = None, message: str = None):
    model = genai.GenerativeModel(model_name="gemini-1.5-pro")

    # Structured prompt that handles both image analysis and user query
    base_prompt = """Analyze this medical scan and provide TWO PARTS in your response:

    PART 1 - SCAN ANALYSIS:
    Provide these exact fields:
    - Scan Type:
    - Organ:
    - Tumor Type:
    - Tumor Subclass:
    - Detailed Description:
    - Possible Causes:
    - Clinical Insights:

    PART 2 - USER QUERY RESPONSE:
    Specifically address the user's question using insights from the scan analysis.
    """

    # Handle text-only queries
    if image_data is None:
        introduction = """I am a medical imaging assistant specializing in analyzing MRI scans, CT scans, and other medical imaging technologies. I can help you understand medical scans, tumor detection, and related medical concepts. 

I can:
- Analyze medical scan images
- Explain medical imaging concepts
- Discuss tumor detection and classification
- Provide general information about medical imaging

Note: My analysis is computational and not a medical diagnosis. Always consult healthcare professionals for medical advice.

Your Query: """
        
        full_prompt = f"{introduction}{message}\n\nPlease provide a detailed response focusing on medical imaging aspects."
        response = model.generate_content(full_prompt)
        return response.text

    # Create multimodal input
    input_parts = [
        {
            "mime_type": mime_type,
            "data": base64.b64encode(image_data).decode("utf-8")
        }
    ]
    
    # Add user query to prompt if provided
    if message:
        full_prompt = f"{base_prompt}\n\nUser Query: {message}"
    else:
        full_prompt = base_prompt

    input_parts.append(full_prompt)
    
    # Generate response with both image and text context
    response = model.generate_content(input_parts)
    
    return response.text
