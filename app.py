"""
LAS File Quality Control Web Application
Flask backend for LAS file analysis and visualization
"""
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
import os
import json
import uuid
import pandas as pd
import numpy as np
from las_qc import LASQCAnalyzer

# Get the directory containing this file (for PythonAnywhere compatibility)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(BASE_DIR, 'templates')
STATIC_DIR = os.path.join(BASE_DIR, 'static')
UPLOAD_DIR = os.path.join(BASE_DIR, 'uploads')

app = Flask(__name__, template_folder=TEMPLATE_DIR, static_folder=STATIC_DIR)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size

# In-memory storage for uploaded files
uploaded_files = {}


def serialize_data(obj):
    """Convert numpy/pandas types to native Python types for JSON serialization"""
    if isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: serialize_data(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [serialize_data(v) for v in obj]
    return obj


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() == 'las'


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Handle LAS file upload"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        try:
            file_id = str(uuid.uuid4())
            filename = secure_filename(file.filename)
            filepath = os.path.join(UPLOAD_DIR, f"{file_id}_{filename}")

            os.makedirs(UPLOAD_DIR, exist_ok=True)
            file.save(filepath)

            analyzer = LASQCAnalyzer(filepath)
            if not analyzer.load_las():
                os.remove(filepath)
                return jsonify({'error': 'Failed to load LAS file'}), 400

            qc_results = analyzer.run_quality_control()

            uploaded_files[file_id] = {
                'analyzer': analyzer,
                'filepath': filepath,
                'filename': filename,
            }

            return jsonify(serialize_data({
                'file_id': file_id,
                'filename': filename,
                'well_info': analyzer.well_info,
                'curves_info': analyzer.curves_info,
                'qc_results': qc_results,
                'available_curves': list(analyzer.df.columns),
            }))

        except Exception as e:
            return jsonify({'error': str(e)}), 500

    return jsonify({'error': 'Invalid file type. Only .las files allowed.'}), 400


@app.route('/api/files/<file_id>/qc', methods=['GET'])
def get_qc_results(file_id):
    if file_id not in uploaded_files:
        return jsonify({'error': 'File not found'}), 404

    analyzer = uploaded_files[file_id]['analyzer']
    qc = analyzer.run_quality_control()
    return jsonify(qc)


@app.route('/api/files/<file_id>/statistics', methods=['GET'])
def get_statistics(file_id):
    if file_id not in uploaded_files:
        return jsonify({'error': 'File not found'}), 404

    analyzer = uploaded_files[file_id]['analyzer']
    stats = analyzer.qc_results.get('curve_statistics', {})
    return jsonify(stats)


@app.route('/api/files/<file_id>/curve/<curve_name>', methods=['GET'])
def get_curve_data(file_id, curve_name):
    if file_id not in uploaded_files:
        return jsonify({'error': 'File not found'}), 404

    analyzer = uploaded_files[file_id]['analyzer']

    if curve_name not in analyzer.df.columns:
        return jsonify({'error': 'Curve not found'}), 404

    depth_col = analyzer.df.columns[0]

    data = {
        'depth': analyzer.df[depth_col].tolist(),
        'values': analyzer.df[curve_name].tolist(),
        'depth_unit': analyzer.curves_info.get(depth_col, {}).get('unit', ''),
        'curve_unit': analyzer.curves_info.get(curve_name, {}).get('unit', ''),
        'curve_info': analyzer.curves_info.get(curve_name, {}),
    }

    return jsonify(data)


@app.route('/api/files/<file_id>/crossplot', methods=['POST'])
def get_crossplot(file_id):
    if file_id not in uploaded_files:
        return jsonify({'error': 'File not found'}), 404

    data = request.json
    x_curve = data.get('x_curve')
    y_curve = data.get('y_curve')
    z_curve = data.get('z_curve')

    analyzer = uploaded_files[file_id]['analyzer']
    plot_data = analyzer.get_crossplot_data(x_curve, y_curve, z_curve)

    return jsonify(plot_data)


@app.route('/api/files/<file_id>/logplot', methods=['POST'])
def get_logplot(file_id):
    if file_id not in uploaded_files:
        return jsonify({'error': 'File not found'}), 404

    data = request.json
    curves = data.get('curves', [])
    depth_range = data.get('depth_range')

    analyzer = uploaded_files[file_id]['analyzer']
    plot_data = analyzer.get_log_plot_data(curves, depth_range)

    return jsonify(plot_data)


@app.route('/api/files/<file_id>/interpretation', methods=['GET'])
def get_interpretation(file_id):
    if file_id not in uploaded_files:
        return jsonify({'error': 'File not found'}), 404

    analyzer = uploaded_files[file_id]['analyzer']
    interpreted_df = analyzer.interpret_lithology()

    depth_col = interpreted_df.columns[0]

    result = {
        'depth': interpreted_df[depth_col].tolist(),
        'depth_unit': analyzer.curves_info.get(depth_col, {}).get('unit', ''),
        'vshale': interpreted_df['VSHALE'].tolist() if 'VSHALE' in interpreted_df.columns else [],
        'porosity': interpreted_df['POROSITY_EST'].tolist() if 'POROSITY_EST' in interpreted_df.columns else [],
        'sw': interpreted_df['SW_EST'].tolist() if 'SW_EST' in interpreted_df.columns else [],
        'lithology': interpreted_df['LITHOLOGY'].tolist() if 'LITHOLOGY' in interpreted_df.columns else [],
    }

    return jsonify(result)


@app.route('/api/files/<file_id>/export/csv', methods=['GET'])
def export_csv(file_id):
    if file_id not in uploaded_files:
        return jsonify({'error': 'File not found'}), 404

    analyzer = uploaded_files[file_id]['analyzer']
    filepath = uploaded_files[file_id]['filepath'].replace('.las', '_export.csv')

    analyzer.df.to_csv(filepath, index=False)

    return send_file(filepath, as_attachment=True, download_name='las_export.csv')


@app.route('/api/files/<file_id>/histogram', methods=['POST'])
def get_histogram(file_id):
    if file_id not in uploaded_files:
        return jsonify({'error': 'File not found'}), 404

    data = request.json
    curve = data.get('curve')
    bins = data.get('bins', 50)

    analyzer = uploaded_files[file_id]['analyzer']
    values = analyzer.df[curve].dropna()

    hist, bin_edges = np.histogram(values, bins=bins)

    return jsonify({
        'curve': curve,
        'hist': hist.tolist(),
        'bin_edges': bin_edges.tolist(),
        'unit': analyzer.curves_info.get(curve, {}).get('unit', ''),
    })


@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large. Maximum size is 50MB.'}), 413


if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)

    # Local debug mode
    debug_mode = os.environ.get('FLASK_DEBUG', 'True').lower() == 'true'

    # Render / cloud compatible port
    port = int(os.environ.get("PORT", 5000))

    app.run(debug=debug_mode, host='0.0.0.0', port=port)
