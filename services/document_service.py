# services/document_service.py
import re
from utils.document_loader import download_pdf, download_webpage, get_pdf_urls, get_course_urls
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Dictionary to store document source mapping
doc_sources = {}

def extract_doc_ids(text, url):
    """Extract document IDs for tracking source URLs"""
    course_codes = re.findall(r'\b[A-Z]{2,4}\s\d{4}[A-Z]?\b', text)
    doc_id_to_url = {}
    for course_code in course_codes:
        doc_id_to_url[course_code] = url
    return doc_id_to_url

def process_documents(callback=None):
    """
    Process all documents and return text chunks and source mapping
    
    Args:
        callback: Optional callback function to update status
    
    Returns:
        tuple: (chunks, doc_sources)
    """
    global doc_sources
    all_texts = []
    
    # Process PDF URLs
    pdf_urls = get_pdf_urls()
    for url in pdf_urls:
        if callback:
            callback(f"Downloading PDF: {url}")
        text = download_pdf(url)
        if text:
            all_texts.append(text)
            doc_sources.update(extract_doc_ids(text, url))
            if callback:
                callback(f"Processed PDF: {url}")
        else:
            if callback:
                callback(f"Failed to download PDF: {url}")

    # Process dynamic webpage URLs constructed from course codes
    dynamic_urls = get_course_urls()
    for url in dynamic_urls:
        if callback:
            callback(f"Downloading webpage: {url}")
        text = download_webpage(url)
        if text:
            all_texts.append(text)
            doc_sources.update(extract_doc_ids(text, url))
            if callback:
                callback(f"Processed webpage: {url}")
        else:
            if callback:
                callback(f"Failed to download webpage: {url}")

    if not all_texts:
        if callback:
            callback("No text was extracted from the provided sources.", "error")
        return None, None

    # Combine all extracted texts
    combined_text = "\n\n".join(all_texts)
    if callback:
        callback(f"Extracted {len(combined_text)} characters of text from all sources.")

    # Split the text into chunks
    if callback:
        callback("Splitting text into manageable chunks...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(combined_text)
    if callback:
        callback(f"Text split into {len(chunks)} chunks.")

    return chunks, doc_sources

def identify_source_urls(content):
    """
    Identify potential source URLs based on content
    
    Args:
        content (str): The content to analyze
        
    Returns:
        list: Relevant source URLs
    """
    global doc_sources
    urls = []
    for code, url in doc_sources.items():
        if code in content:
            urls.append((code, url))
    
    # If no specific course codes found, return most relevant URLs
    if not urls:
        pdf_urls = get_pdf_urls()
        if "undergraduate" in content.lower():
            urls.append(("Undergraduate Catalog", pdf_urls[0]))
        if "graduate" in content.lower():
            urls.append(("Graduate Catalog", pdf_urls[1]))
    
    return list(set(urls))[:3]  # Return up to 3 unique sources

def get_doc_sources():
    """Get current document sources"""
    global doc_sources
    return doc_sources