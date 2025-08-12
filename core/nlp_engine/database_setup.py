# database_setup.py
import sqlite3
import os

def initialize_database(db_path="Actual.db"):
    """Create and initialize the glossary database"""
    try:
        # Connect to database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Create glossary table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS glossary (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                english_translation TEXT NOT NULL,
                hindi_term TEXT NOT NULL,
                definition TEXT
            )
        """)
        
        # Add sample data
        sample_terms = [
            ("GDP", "सकल घरेलू उत्पाद", "Total market value of goods and services"),
            ("inflation", "मुद्रास्फीति", "General increase in prices"),
            ("export", "निर्यात", "Sending goods to another country for sale"),
            ("import", "आयात", "Bringing goods from another country for sale"),
            ("investment", "निवेश", "Putting money into something to make a profit"),
            ("trade", "व्यापार", "Buying and selling of goods and services"),
            ("market", "बाजार", "Place where buyers and sellers meet"),
            ("economy", "अर्थव्यवस्था", "System of production, distribution, and consumption")
        ]
        
        cursor.executemany(
            "INSERT INTO glossary (english_translation, hindi_term, definition) VALUES (?, ?, ?)",
            sample_terms
        )
        
        # Commit and close
        conn.commit()
        conn.close()
        
        print(f"Database initialized successfully at: {os.path.abspath(db_path)}")
        return True
    except Exception as e:
        print(f"Error initializing database: {e}")
        return False

if __name__ == "__main__":
    initialize_database()