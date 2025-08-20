# core/pdf_processor/pdf_pipeline.py
"""
Main PDF Translation Pipeline
Flow: Input PDF → NLP Engine → Translated PDF
"""

import os
import sys
import fitz  # PyMuPDF
import logging
from pathlib import Path
from typing import List, Optional, Dict
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# Import from your NLP engine
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'nlp_engine'))
from adaptive_translator import AdaptiveSTEMTranslator, TranslationConfig, Language

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PDFTranslationPipeline:
    """Complete pipeline: PDF → NLP Engine → Translated PDF"""
    
    def __init__(self, 
                 font_path: str = "fonts/NotoSansDevanagari-Regular.ttf",
                 db_path: str = "core/nlp_engine/Actual.db"):
        
        self.font_path = font_path
        self.db_path = db_path
        self.translator = None
        
        # Create necessary directories
        os.makedirs("input", exist_ok=True)
        os.makedirs("output", exist_ok=True)
        
        # Register Hindi font
        self._register_font()
    
    def _register_font(self):
        """Register Devanagari font for PDF generation"""
        try:
            if os.path.exists(self.font_path):
                pdfmetrics.registerFont(TTFont("NotoSansDevanagari", self.font_path))
                logger.info(f"Font registered: {self.font_path}")
                return True
            else:
                logger.error(f"Font not found: {self.font_path}")
                return False
        except Exception as e:
            logger.error(f"Font registration failed: {e}")
            return False
    
    def extract_pdf_text(self, pdf_path: str) -> List[str]:
        """Step 1: Extract text from input PDF page by page"""
        logger.info(f"Extracting text from: {pdf_path}")
        
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        pages = []
        try:
            with fitz.open(pdf_path) as doc:
                for page_num in range(len(doc)):
                    page = doc[page_num]
                    text = page.get_text("text").strip()
                    pages.append(text)
                    logger.debug(f"Page {page_num + 1}: {len(text)} characters")
            
            logger.info(f"Successfully extracted {len(pages)} pages")
            return pages
            
        except Exception as e:
            logger.error(f"PDF extraction failed: {e}")
            raise
    
    def initialize_nlp_engine(self, config: TranslationConfig):
        """Step 2: Initialize NLP engine with configuration"""
        logger.info("Initializing NLP Engine...")
        
        try:
            self.translator = AdaptiveSTEMTranslator(self.db_path, config)
            logger.info("NLP Engine initialized successfully")
        except Exception as e:
            logger.error(f"NLP Engine initialization failed: {e}")
            raise
    
    def translate_with_nlp(self, pages_text: List[str]) -> List[str]:
        """Step 3: Process text through NLP engine"""
        if not self.translator:
            raise RuntimeError("NLP Engine not initialized")
        
        logger.info(f"Translating {len(pages_text)} pages through NLP Engine...")
        
        translated_pages = []
        for i, page_text in enumerate(pages_text):
            logger.info(f"Processing page {i + 1}/{len(pages_text)}")
            
            if not page_text.strip():
                translated_pages.append("")
                continue
            
            # Handle long pages by chunking
            chunks = self._chunk_text(page_text)
            translated_chunks = []
            
            for chunk in chunks:
                try:
                    result = self.translator.translate_chunk(chunk)
                    translated_chunks.append(result.translated_text)
                    logger.debug(f"Chunk translated with confidence: {result.confidence_score:.2f}")
                except Exception as e:
                    logger.warning(f"Chunk translation failed: {e}")
                    translated_chunks.append(f"[Translation Error: {chunk}]")
            
            translated_pages.append("\n\n".join(translated_chunks))
        
        logger.info("Translation completed")
        return translated_pages
    
    def _chunk_text(self, text: str, max_chars: int = 1000) -> List[str]:
        """Split text into manageable chunks for NLP processing"""
        if len(text) <= max_chars:
            return [text]
        
        chunks = []
        paragraphs = text.split("\n\n")
        current_chunk = []
        current_length = 0
        
        for para in paragraphs:
            para_length = len(para) + 2  # +2 for \n\n
            
            if current_length + para_length > max_chars and current_chunk:
                chunks.append("\n\n".join(current_chunk))
                current_chunk = [para]
                current_length = para_length
            else:
                current_chunk.append(para)
                current_length += para_length
        
        if current_chunk:
            chunks.append("\n\n".join(current_chunk))
        
        return chunks
    
    def generate_translated_pdf(self, translated_pages: List[str], output_path: str):
        """Step 4: Generate final translated PDF"""
        logger.info(f"Generating translated PDF: {output_path}")
        
        try:
            # Create document
            doc = SimpleDocTemplate(
                output_path,
                pagesize=A4,
                leftMargin=50,
                rightMargin=50,
                topMargin=50,
                bottomMargin=50
            )
            
            # Setup styles
            styles = getSampleStyleSheet()
            hindi_style = ParagraphStyle(
                name="Hindi",
                parent=styles["Normal"],
                fontName="NotoSansDevanagari",
                fontSize=11,
                leading=16,
                spaceAfter=10,
                alignment=0  # Left alignment
            )
            
            # Build content
            story = []
            for page_idx, page_text in enumerate(translated_pages):
                if page_text.strip():
                    # Split into paragraphs
                    paragraphs = page_text.split("\n\n")
                    for para in paragraphs:
                        if para.strip():
                            # Replace line breaks with HTML breaks
                            formatted_para = para.replace("\n", "<br/>")
                            story.append(Paragraph(formatted_para, hindi_style))
                            story.append(Spacer(1, 6))
                
                # Add page break (except for last page)
                if page_idx < len(translated_pages) - 1:
                    story.append(PageBreak())
            
            # Build the PDF
            doc.build(story)
            logger.info(f"PDF generated successfully: {output_path}")
            
        except Exception as e:
            logger.error(f"PDF generation failed: {e}")
            raise
    
    def save_intermediate_files(self, pages_en: List[str], pages_hi: List[str]):
        """Save intermediate text files for debugging"""
        try:
            # Save extracted English text
            en_path = "output/extracted_english.txt"
            with open(en_path, "w", encoding="utf-8") as f:
                f.write("\n\n=== PAGE BREAK ===\n\n".join(pages_en))
            logger.info(f"Saved: {en_path}")
            
            # Save translated Hindi text
            hi_path = "output/translated_hindi.txt"
            with open(hi_path, "w", encoding="utf-8") as f:
                f.write("\n\n=== PAGE BREAK ===\n\n".join(pages_hi))
            logger.info(f"Saved: {hi_path}")
            
        except Exception as e:
            logger.warning(f"Could not save intermediate files: {e}")
    
    def translate_pdf(self, 
                     input_pdf: str, 
                     output_pdf: str, 
                     config: TranslationConfig,
                     save_intermediate: bool = True) -> Dict:
        """
        Main pipeline method: PDF → NLP Engine → Translated PDF
        
        Args:
            input_pdf: Path to input English PDF
            output_pdf: Path for output translated PDF
            config: Translation configuration for NLP engine
            save_intermediate: Save intermediate text files
            
        Returns:
            Dict with translation results and statistics
        """
        
        start_time = logger.info(f"Starting PDF translation pipeline...")
        logger.info(f"Input: {input_pdf}")
        logger.info(f"Output: {output_pdf}")
        logger.info(f"Language: {config.target_language.name}")
        logger.info(f"Grade Level: {config.grade_level}")
        
        try:
            # Step 1: Extract PDF text
            pages_english = self.extract_pdf_text(input_pdf)
            
            # Step 2: Initialize NLP Engine
            self.initialize_nlp_engine(config)
            
            # Step 3: Translate through NLP Engine
            pages_hindi = self.translate_with_nlp(pages_english)
            
            # Step 4: Generate translated PDF
            self.generate_translated_pdf(pages_hindi, output_pdf)
            
            # Optional: Save intermediate files
            if save_intermediate:
                self.save_intermediate_files(pages_english, pages_hindi)
            
            # Prepare results
            result = {
                "success": True,
                "input_pdf": input_pdf,
                "output_pdf": output_pdf,
                "pages_processed": len(pages_english),
                "target_language": config.target_language.name,
                "grade_level": config.grade_level,
                "message": "Translation completed successfully"
            }
            
            logger.info("=== TRANSLATION PIPELINE COMPLETED ===")
            logger.info(f"✅ Success: {result['pages_processed']} pages translated")
            
            return result
            
        except Exception as e:
            error_msg = f"Translation pipeline failed: {str(e)}"
            logger.error(error_msg)
            
            return {
                "success": False,
                "input_pdf": input_pdf,
                "output_pdf": output_pdf,
                "error": error_msg
            }
