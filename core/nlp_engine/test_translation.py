# test_translation.py
import os
import logging
from dotenv import load_dotenv
from adaptive_translator import AdaptiveSTEMTranslator, TranslationConfig, Language

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s'
)
logger = logging.getLogger(__name__)

def run_test():
    try:
        # Get API key from environment
        gemini_key = os.getenv("GEMINI_API_KEY")
        if not gemini_key:
            logger.error("GEMINI_API_KEY not found in environment")
            return False

        # Configuration with Gemini enabled
        cfg = TranslationConfig(
            target_language=Language.HINDI,
            grade_level=9,
            use_gemini=True,
            gemini_api_key=gemini_key,
            cultural_adaptation=True
        )
        
        # Initialize translator
        translator = AdaptiveSTEMTranslator("Actual.db", cfg)
        
        # Sample STEM text
        sample = """
        Economic growth refers to the increase in GDP over time. 
        When productivity increases, exports rise and the balance of payments improves.
        Inflation occurs when too much money chases too few goods.
        """
        
        # Translate
        result = translator.translate_chunk(sample)
        
        # Print results
        print("\n=== STEM Translation Result ===")
        print("Original:", result.original_text)
        print("Translated:", result.translated_text)
        print(f"Grade Level: {result.grade_level:.1f}")
        print(f"Confidence: {result.confidence_score:.2f}")
        print(f"Technical Terms: {result.technical_terms_used}")
        print(f"Cultural Adaptations: {result.cultural_adaptations}")
        print(f"Processing Time: {result.processing_time:.2f}s")
        
        return True
    except Exception as e:
        logger.error(f"Test failed: {e}")
        return False

if __name__ == "__main__":
    run_test()