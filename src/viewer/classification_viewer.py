#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ==============================================================================
# NASA Deep Space Image Classification & Neural Training Pipeline
#
# Author:       Raúl Salas Sahuquillo
# Repository:   https://github.com/RaulSalasSahuquillo/nasa-deep-space-classifier
# License:      Creative Commons Attribution-NonCommercial-ShareAlike 4.0 (CC BY-NC-SA 4.0)
# File:         src/viewer/classification_viewer.py
# Description:  Flask-powered web server that displays classification audit
#               records from SQLite grouped into category tables.
# ==============================================================================

import os
import sqlite3
from flask import Flask, render_template

# Locate directory paths dynamically
VIEWER_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(VIEWER_DIR, '..', '..'))
DB_PATH = os.path.join(PROJECT_ROOT, 'network_sorting', 'classified_images.db')
TEMPLATES_DIR = os.path.join(VIEWER_DIR, 'templates')

app = Flask(__name__, template_folder=TEMPLATES_DIR)

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

@app.route('/')
def index():
    if not os.path.exists(DB_PATH):
        return render_template('index.html', grouped_data={}, total_count=0, error=f"Database file not found at: {DB_PATH}")
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Verify whether the 'classifications' table exists
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='classifications'")
    table_exists = cursor.fetchone()
    
    if not table_exists:
        conn.close()
        return render_template('index.html', grouped_data={}, total_count=0, error="Table 'classifications' does not exist in the database.")
    
    # Query all classification records
    cursor.execute("SELECT id, filename, source_directory, predicted_class, destination_path, timestamp FROM classifications ORDER BY id ASC")
    rows = cursor.fetchall()
    conn.close()
    
    # Group records by category (predicted_class)
    grouped_data = {}
    for row in rows:
        category = row['predicted_class'] if row['predicted_class'] else 'Unclassified'
        if category not in grouped_data:
            grouped_data[category] = []
        grouped_data[category].append(row)
    
    return render_template('index.html', grouped_data=grouped_data, total_count=len(rows), error=None)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
