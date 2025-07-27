import os
import uuid
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
import torch
from PIL import Image
import io

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
os.environ['OMP_NUM_THREADS'] = '1'import base64

# Import style transfer utilities
from utils.style_transfer import StyleTransfer

app = Flask(__name__, static_folder='../frontend/build', static_url_path='')

# Configure CORS to allow requests from the React development server
cors = CORS(app, resources={
    r"/api/*": {"origins": ["http://localhost:3000", "http://127.0.0.1:3000"]},
    r"/uploads/*": {"origins": ["http://localhost:3000", "http://127.0.0.1:3000"]},
    r"/outputs/*": {"origins": ["http://localhost:3000", "http://127.0.0.1:3000"]}
})

# Configuration
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'jfif'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload and output directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Initialize style transfer model
style_transfer = StyleTransfer()

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'pytorch_available': torch.cuda.is_available(),
        'device': str(style_transfer.device)
    }), 200

@app.route('/api/upload', methods=['POST', 'OPTIONS'])
def upload_file():
    """Handle file uploads for content and style images"""
    if request.method == 'OPTIONS':
        # Handle preflight request
        response = jsonify({'status': 'preflight'})
        response.headers.add('Access-Control-Allow-Origin', 'http://localhost:3000')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST')
        return response
        
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        try:
            # Generate a unique filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            original_filename = secure_filename(file.filename)
            file_ext = original_filename.rsplit('.', 1)[1].lower()
            filename = f"{timestamp}_{uuid.uuid4().hex}.{file_ext}"
            
            # Ensure upload directory exists
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            
            # Save the file
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            response = jsonify({
                'message': 'File uploaded successfully',
                'filename': filename,
                'url': f'/uploads/{filename}'
            })
            response.headers.add('Access-Control-Allow-Origin', 'http://localhost:3000')
            return response
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'File type not allowed. Please upload a PNG, JPG, or JPEG file.'}), 400

@app.route('/api/process', methods=['POST'])
def process_image():
    """Process the style transfer between content and style images with enhanced parameters"""
    print("\n=== Starting Enhanced Style Transfer Process ===")
    print(f"Current working directory: {os.getcwd()}")
    
    # Check if required data is present
    if 'content' not in request.json or 'style' not in request.json:
        error_msg = 'Missing content or style image in request'
        print(f"Error: {error_msg}")
        return jsonify({'error': error_msg}), 400
    
    # Get filenames
    content_filename = request.json['content']
    style_filename = request.json['style']
    
    # Get style transfer parameters with defaults
    num_steps = int(request.json.get('num_steps', 300))  # Increased for better quality
    style_weight = float(request.json.get('style_weight', 1000000))  # Higher for more style
    content_weight = float(request.json.get('content_weight', 1))
    tv_weight = float(request.json.get('tv_weight', 1e-5))  # Total variation weight
    image_size = int(request.json.get('image_size', 384))  # Higher resolution for better details
    
    print(f"Processing content: {content_filename}, style: {style_filename}")
    print(f"Parameters: steps={num_steps}, style_weight={style_weight}, "
          f"content_weight={content_weight}, tv_weight={tv_weight}, size={image_size}")
    
    # Validate file paths
    content_path = os.path.join(app.config['UPLOAD_FOLDER'], content_filename)
    style_path = os.path.join(app.config['UPLOAD_FOLDER'], style_filename)
    
    if not os.path.exists(content_path) or not os.path.exists(style_path):
        error_msg = f"One or both image files not found. Content exists: {os.path.exists(content_path)}, Style exists: {os.path.exists(style_path)}"
        print(error_msg)
        return jsonify({'error': error_msg}), 404
    
    try:
        # Generate output filename with parameters
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_filename = f"result_{timestamp}.jpg"
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        print(f"Output will be saved to: {output_path}")
        
        # Process the style transfer with the specified image size
        print("Loading and processing images...")
        content_img = style_transfer.load_image(content_path, imsize=image_size)
        style_img = style_transfer.load_image(style_path, imsize=image_size)
        
        # Use content image as input for style transfer
        input_img = content_img.clone()
        
        print("Starting enhanced style transfer...")
        # Run style transfer with the specified parameters
        output = style_transfer.run_style_transfer(
            content_img=content_img,
            style_img=style_img,
            input_img=input_img,
            num_steps=num_steps,
            style_weight=style_weight,
            content_weight=content_weight,
            tv_weight=tv_weight
        )
        
        print("Style transfer completed. Saving output...")
        # Save the output image with maximum quality
        style_transfer.save_image(output, output_path)
        
        if not os.path.exists(output_path):
            raise Exception("Failed to save output image")
            
        print(f"Style transfer completed successfully. Output saved to {output_path}")
        print("==========================================\n")
        
        return jsonify({
            'message': 'Style transfer completed successfully',
            'result': output_filename,
            'url': f'/outputs/{output_filename}'
        }), 200
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        error_msg = f"Error during style transfer: {str(e)}\n{error_details}"
        print(error_msg)
        app.logger.error(error_msg)
        
        return jsonify({
            'error': 'Failed to process images',
            'details': str(e),
            'traceback': error_details
        }), 500

@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    """Serve uploaded files"""
    try:
        response = send_from_directory(
            app.config['UPLOAD_FOLDER'],
            filename,
            as_attachment=False
        )
        response.headers.add('Access-Control-Allow-Origin', 'http://localhost:3000')
        response.headers.add('Cache-Control', 'no-cache, no-store, must-revalidate')
        response.headers.add('Pragma', 'no-cache')
        response.headers.add('Expires', '0')
        return response
    except FileNotFoundError:
        return jsonify({'error': 'File not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/outputs/<path:filename>')
def output_file(filename):
    """Serve processed output files with proper download headers"""
    try:
        # Get the full path to the file
        file_path = os.path.join(app.config['OUTPUT_FOLDER'], filename)
        
        # Check if file exists
        if not os.path.isfile(file_path):
            return jsonify({'error': 'File not found'}), 404
            
        # Get file extension for content type
        _, ext = os.path.splitext(filename)
        mime_types = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.gif': 'image/gif',
            '.webp': 'image/webp'
        }
        
        # Set content type based on file extension, default to octet-stream
        content_type = mime_types.get(ext.lower(), 'application/octet-stream')
        
        # Create response with file
        with open(file_path, 'rb') as f:
            file_data = f.read()
            
        response = app.response_class(
            response=file_data,
            status=200,
            mimetype=content_type
        )
        
        # Set headers for download and CORS
        response.headers['Content-Disposition'] = f'attachment; filename={filename}'
        response.headers['Content-Length'] = os.path.getsize(file_path)
        response.headers['Access-Control-Allow-Origin'] = 'http://localhost:3000'
        response.headers['Access-Control-Allow-Methods'] = 'GET, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
        
        return response
        
    except FileNotFoundError:
        return jsonify({'error': 'File not found'}), 404
    except Exception as e:
        app.logger.error(f"Error serving file {filename}: {str(e)}")
        return jsonify({'error': 'Failed to serve file', 'details': str(e)}), 500

@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({
        'error': 'File too large',
        'message': 'The file is larger than the maximum allowed size (16MB)'
    }), 413

if __name__ == '__main__':
    # Print available device information
    print(f"PyTorch is using: {style_transfer.device}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    
    # Run the Flask app
    app.run(host='0.0.0.0', port=5000, debug=True)
