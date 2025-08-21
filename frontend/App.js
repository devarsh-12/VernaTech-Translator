import React, { useState } from "react";
import "./App.css";

export default function App() {
  const [text, setText] = useState("");
  const [grade, setGrade] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  
  const handleTranslate = async () => {
    setLoading(true);
    try {
      const res = await fetch("http://127.0.0.1:5000/translate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text, grade_level: grade ? parseInt(grade) : undefined })
      });
      const data = await res.json();
      setResult(data);
    } catch (error) {
      console.error("Error:", error);
    }
    setLoading(false);
  };

  return (
    <div>
        <nav className="navbar">
        <h2 className="app-title">Adaptive STEM Translator</h2>
        <ul className="nav-links">
          <li><a href="#home">Home</a></li>
          <li><a href="#features">Features</a></li>
          <li><a href="#about">About</a></li>
        </ul>
      </nav>

      <div className="app-container">
        <textarea
            className="input-textarea"
            rows="4"
            placeholder="Enter text to translate..."
            value={text}
            onChange={(e) => setText(e.target.value)}
        />

        <input
            className="input-grade"
            type="number"
            placeholder="Grade level (default: 9)"
            value={grade}
            onChange={(e) => setGrade(e.target.value)}
        />

        <button
            className="translate-btn"
            onClick={handleTranslate}
            disabled={loading}
        >
            {loading ? "Translating..." : "Translate"}
        </button>
      </div>
    

      {result && (
        <div className="results-card">
          <h3>Results</h3>
          <p><b>Original:</b> {result.original_text}</p>
          <p><b>Translated:</b> {result.translated_text}</p>
          <p><b>Grade Level:</b> {result.grade_level}</p>
          <p><b>Confidence:</b> {result.confidence_score}</p>
          <p><b>Technical Terms:</b> {result.technical_terms_used}</p>
          <p><b>Cultural Adaptations:</b> {result.cultural_adaptations}</p>
          <p><b>Processing Time:</b> {result.processing_time}s</p>
        </div>
      )}
    </div>
  );
}
