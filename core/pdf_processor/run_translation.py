# run_translation.py
import os
import sys
import fitz  # PyMuPDF
import logging
from typing import List, Dict
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from deep_translator import GoogleTranslator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimplePDFTranslator:
    """Simple PDF translator without heavy models"""
    
    def __init__(self, font_path: str = "fonts/NotoSansDevanagari-Regular.ttf"):
        self.font_path = font_path
        
        # Create directories
        os.makedirs("input", exist_ok=True)
        os.makedirs("output", exist_ok=True)
        
        # Register font
        self._register_font()
    
    def _register_font(self):
        """Register Hindi font for PDF generation"""
        try:
            if os.path.exists(self.font_path):
                pdfmetrics.registerFont(TTFont("NotoSansDevanagari", self.font_path))
                logger.info(f"Font registered: {self.font_path}")
                return True
            else:
                logger.warning(f"Font not found: {self.font_path}. Will use default font.")
                return False
        except Exception as e:
            logger.error(f"Font registration failed: {e}")
            return False
    
    def extract_pdf_text(self, pdf_path: str) -> List[str]:
        """Extract text from PDF page by page"""
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
    
    def translate_text(self, pages_text: List[str], target_lang: str = "hi") -> List[str]:
        """Translate text using Google Translate"""
        logger.info(f"Translating {len(pages_text)} pages to {target_lang}...")
        
        translator = GoogleTranslator(source="auto", target=target_lang)
        translated_pages = []
        
        for i, page_text in enumerate(pages_text):
            logger.info(f"Translating page {i + 1}/{len(pages_text)}")
            
            if not page_text.strip():
                translated_pages.append("")
                continue
            
            # Handle long pages with chunking
            chunks = self._chunk_text(page_text, max_chars=4500)
            translated_chunks = []
            
            for j, chunk in enumerate(chunks):
                try:
                    translated = translator.translate(chunk)
                    translated_chunks.append(translated)
                    logger.debug(f"  Chunk {j + 1}/{len(chunks)} translated")
                except Exception as e:
                    logger.warning(f"Translation failed for chunk {j + 1}: {e}")
                    translated_chunks.append(f"[Translation Error: {chunk[:100]}...]")
            
            translated_pages.append("\n\n".join(translated_chunks))
        
        logger.info("Translation completed")
        return translated_pages
    
    def _chunk_text(self, text: str, max_chars: int = 4500) -> List[str]:
        """Split text into chunks for API limits"""
        if len(text) <= max_chars:
            return [text]
        
        chunks = []
        paragraphs = text.split("\n\n")
        current_chunk = []
        current_length = 0
        
        for para in paragraphs:
            para_with_sep = para + "\n\n"
            if current_length + len(para_with_sep) > max_chars and current_chunk:
                chunks.append("".join(current_chunk).rstrip())
                current_chunk = [para_with_sep]
                current_length = len(para_with_sep)
            else:
                current_chunk.append(para_with_sep)
                current_length += len(para_with_sep)
        
        if current_chunk:
            chunks.append("".join(current_chunk).rstrip())
        
        return chunks
    
    def generate_pdf(self, translated_pages: List[str], output_path: str):
        """Generate Hindi PDF"""
        logger.info(f"Generating PDF: {output_path}")
        
        try:
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
            
            # Try to use Hindi font, fallback to default
            font_name = "NotoSansDevanagari" if os.path.exists(self.font_path) else "Helvetica"
            
            hindi_style = ParagraphStyle(
                name="Hindi",
                parent=styles["Normal"],
                fontName=font_name,
                fontSize=11,
                leading=16,
                spaceAfter=10,
                alignment=0
            )
            
            # Build content
            story = []
            for page_idx, page_text in enumerate(translated_pages):
                if page_text.strip():
                    paragraphs = page_text.split("\n\n")
                    for para in paragraphs:
                        if para.strip():
                            formatted_para = para.replace("\n", "<br/>")
                            story.append(Paragraph(formatted_para, hindi_style))
                            story.append(Spacer(1, 6))
                
                # Add page break
                if page_idx < len(translated_pages) - 1:
                    story.append(PageBreak())
            
            doc.build(story)
            logger.info(f"PDF generated successfully: {output_path}")
            
        except Exception as e:
            logger.error(f"PDF generation failed: {e}")
            raise
    
    def save_intermediate_files(self, pages_en: List[str], pages_hi: List[str]):
        """Save intermediate text files"""
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
    
    def translate_pdf(self, input_pdf: str, output_pdf: str, target_lang: str = "hi") -> Dict:
        """Main translation pipeline"""
        logger.info("=== PDF TRANSLATION PIPELINE ===")
        logger.info(f"Input: {input_pdf}")
        logger.info(f"Output: {output_pdf}")
        logger.info(f"Target Language: {target_lang}")
        
        try:
            # Step 1: Extract
            pages_english = self.extract_pdf_text(input_pdf)
            
            # Step 2: Translate
            pages_hindi = self.translate_text(pages_english, target_lang)
            
            # Step 3: Generate PDF
            self.generate_pdf(pages_hindi, output_pdf)
            
            # Step 4: Save intermediate files
            self.save_intermediate_files(pages_english, pages_hindi)
            
            result = {
                "success": True,
                "input_pdf": input_pdf,
                "output_pdf": output_pdf,
                "pages_processed": len(pages_english),
                "target_language": target_lang,
                "message": "Translation completed successfully"
            }
            
            logger.info("=== TRANSLATION COMPLETED ===")
            return result
            
        except Exception as e:
            error_msg = f"Translation failed: {str(e)}"
            logger.error(error_msg)
            
            return {
                "success": False,
                "input_pdf": input_pdf,
                "output_pdf": output_pdf,
                "error": error_msg
            }

def main():
    """Main function to run PDF translation"""
    
    # File paths (relative to project root)
    input_pdf = "input/english_document.pdf"
    output_pdf = "output/hindi_document.pdf"
    
    print("🚀 PDF Translation Pipeline")
    print("=" * 40)
    
    # Check if input file exists
    if not os.path.exists(input_pdf):
        print(f"❌ Input PDF not found: {input_pdf}")
        print("\n📝 To fix this:")
        print(f"   1. Create the input folder: mkdir input")
        print(f"   2. Place your English PDF as: {input_pdf}")
        print(f"   3. Run this script again")
        return
    
    # Initialize translator
    translator = SimplePDFTranslator()
    
    # Run translation
    print(f"📄 Input: {input_pdf}")
    print(f"📁 Output: {output_pdf}")
    print()
    
    result = translator.translate_pdf(
        input_pdf=input_pdf,
        output_pdf=output_pdf,
        target_lang="hi"  # Hindi
    )
    
    # Display results
    print("\n" + "=" * 40)
    if result["success"]:
        print("✅ TRANSLATION SUCCESSFUL!")
        print(f"   📄 Pages processed: {result['pages_processed']}")
        print(f"   📁 Output saved: {result['output_pdf']}")
        print(f"   🗣️  Language: {result['target_language']}")
        print("\n📂 Generated files:")
        print(f"   • {result['output_pdf']} (Hindi PDF)")
        print(f"   • output/extracted_english.txt")
        print(f"   • output/translated_hindi.txt")
    else:
        print("❌ TRANSLATION FAILED!")
        print(f"   Error: {result['error']}")

if __name__ == "__main__":
    main()
