from flask import Flask, request, jsonify
from adaptive_translator import AdaptiveSTEMTranslator, TranslationConfig, Language
import logging
from flask_cors import CORS

app = Flask(__name__)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s'
)
logger = logging.getLogger(__name__)

# Configure translator
cfg = TranslationConfig(
    target_language=Language.HINDI,
    grade_level=9,
    use_gemini=True,
    gemini_api_key= ,  #from .env
    cultural_adaptation=True
)
translator = AdaptiveSTEMTranslator("Actual.db", cfg)

CORS(app)

@app.route("/translate", methods=["POST"])
def translate():
    try:
        data = request.get_json()
        text = data.get("text", "")
        
        # Overriding the grade level configuration
        grade_level = data.get("grade_level", 9)
        translator.config.grade_level = grade_level

        if not text.strip():
            return jsonify({"error": "No text provided"}), 400
         
        result = translator.translate_chunk(text)

        return jsonify({
            "original_text": result.original_text,
            "translated_text": result.translated_text,
            "grade_level": result.grade_level,
            "confidence_score": result.confidence_score,
            "technical_terms_used": result.technical_terms_used,
            "cultural_adaptations": result.cultural_adaptations,
            "processing_time": result.processing_time
        })
    except Exception as e:
        logger.error(f"Translation failed: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
