from flask import Flask, render_template, request, jsonify, redirect, url_for
import os
import base64
from io import BytesIO
import re
from dotenv import load_dotenv
from supabase import create_client
from azure.storage.blob import BlobServiceClient
import google.generativeai as genai
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
import atexit
import logging
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Configure Supabase client
supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

# Configure Gemini API
def configure_gemini_api(api_key, model_name):
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(model_name)

model = configure_gemini_api(os.getenv("GEMINI_API_KEY"), "gemini-1.5-flash")

# Azure Storage Configuration
AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
AZURE_CONTAINER_NAME = os.getenv("AZURE_CONTAINER_NAME")

# Generate a response from Gemini API
def generate_response(model, prompt, img):
    try:
        img_base64 = base64.b64encode(img.read()).decode("utf-8")
        payload = [{"mime_type": "image/jpeg", "data": img_base64}, prompt]
        response = model.generate_content(payload)
        return response.text if response and hasattr(response, "text") else "Error: No valid response from Gemini."
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        return str(e)

# Generate prompt for MRI triage
def generate_prompt():
    return (
        "You are an expert radiologist triaging brain MRI images based on severity, using a scale from 1 to 10."
        "\n- **9.00–10.00 (Critical):** Life-threatening conditions needing immediate intervention."
        "\n- **7.00–8.99 (Urgent):** Serious but non-immediate conditions."
        "\n- **4.00–6.99 (Moderate):** Non-urgent but medically relevant conditions."
        "\n- **1.00–3.99 (Low):** Normal MRI findings or minor, non-urgent abnormalities."
        "\n\n**Instructions:**\n- Provide a severity score with at least two decimal places."
        "\n- Explain the abnormality concisely."
        "\n- End with 'Hence its severity score is <rating>'."
    )

# Parse Gemini response
def parse_gemini_response(response):
    severity_rating = None
    comment = "No comment provided."
    if response:
        severity_match = re.search(r"Hence its severity score is (\d+\.\d+|\d+)", response)
        if severity_match:
            severity_rating = float(severity_match.group(1))
        comment_match = re.search(r"(.*?)\s*Hence its severity score is", response, re.DOTALL)
        if comment_match:
            comment = comment_match.group(1).strip()
    return severity_rating, comment

# Get processed image names
def get_processed_image_names():
    try:
        response = supabase.table('mri_triage_results').select('image_name').execute()
        if not response.data:
            return []
        return [item['image_name'] for item in response.data]
    except Exception as e:
        logger.error(f"Error fetching processed images: {str(e)}")
        return []

# Fetch images from Azure Storage
def fetch_images_from_azure():
    try:
        blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
        container_client = blob_service_client.get_container_client(AZURE_CONTAINER_NAME)
        images = []
        processed_images = get_processed_image_names()

        for blob in container_client.list_blobs():
            if blob.name in processed_images:
                continue  # Skip already processed images
            blob_client = container_client.get_blob_client(blob.name)
            image_data = BytesIO(blob_client.download_blob().readall())
            images.append((blob.name, image_data))
        
        return images
    except Exception as e:
        logger.error(f"Error fetching images from Azure: {str(e)}")
        return str(e)

# Upload image to Azure Storage
def upload_image_to_azure(file_data, filename):
    try:
        blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
        container_client = blob_service_client.get_container_client(AZURE_CONTAINER_NAME)
        
        # Generate a unique filename if needed (prevents overwriting)
        if not filename or filename in get_processed_image_names():
            unique_id = str(uuid.uuid4())
            file_extension = os.path.splitext(filename)[1] if filename else '.jpg'
            filename = f"manual_upload_{unique_id}{file_extension}"
        
        # Upload the file
        blob_client = container_client.get_blob_client(filename)
        file_data.seek(0)
        blob_client.upload_blob(file_data, overwrite=True)
        
        return filename
    except Exception as e:
        logger.error(f"Error uploading image to Azure: {str(e)}")
        return None

# Process a single uploaded image
def process_uploaded_image(file_data, filename):
    try:
        # First, upload the image to Azure for consistency
        azure_filename = upload_image_to_azure(file_data, filename)
        if not azure_filename:
            return {"error": "Failed to upload image to Azure Storage"}
        
        # Now process the image
        file_data.seek(0)
        severity_rating, comment = parse_gemini_response(generate_response(model, generate_prompt(), file_data))
        
        # Prepare the image data for display and storage
        file_data.seek(0)
        image_base64 = base64.b64encode(file_data.read()).decode("utf-8")
        
        result = {
            "image_name": azure_filename,
            "severity_rating": severity_rating,
            "comment": comment,
            "image_data": image_base64
        }
        
        # Save to Supabase
        save_results_to_supabase([result])
        
        return result
    except Exception as e:
        logger.error(f"Error processing uploaded image: {str(e)}")
        return {"error": str(e)}

# Save results to Supabase
def save_results_to_supabase(results):
    try:
        logger.info(f"Saving {len(results)} results to Supabase")
        response = supabase.table('mri_triage_results').insert(results).execute()
        if not response.data:
            return "Error saving to database."
        return f"Results saved successfully: {len(results)} images processed."
    except Exception as e:
        logger.error(f"Error saving results: {str(e)}")
        return f"Error: {str(e)}"

# Fetch history from Supabase
def fetch_history_from_supabase():
    try:
        response = supabase.table('mri_triage_results').select('*').order('created_at', desc=True).execute()
        return response.data if response.data else []  # Ensure it always returns an array
    except Exception as e:
        logger.error(f"Error fetching history: {str(e)}")
        return []  # Return an empty list instead of None

# Automated job to fetch and process images
def auto_fetch_and_process_images():
    logger.info("Starting scheduled job: Fetching and processing new MRI images")
    images = fetch_images_from_azure()
    
    if isinstance(images, str):
        logger.error(f"Error in scheduled job: {images}")
        return
    
    if not images:
        logger.info("No new images to process")
        return
    
    logger.info(f"Found {len(images)} new images to process")
    results = []
    for image_name, image_data in images:
        logger.info(f"Processing image: {image_name}")
        severity_rating, comment = parse_gemini_response(generate_response(model, generate_prompt(), image_data))
        image_data.seek(0)
        image_base64 = base64.b64encode(image_data.getvalue()).decode("utf-8")
        results.append({
            "image_name": image_name,
            "severity_rating": severity_rating,
            "comment": comment,
            "image_data": image_base64
        })
    
    if results:
        save_result = save_results_to_supabase(results)
        logger.info(save_result)

# Initialize scheduler
scheduler = BackgroundScheduler()
scheduler.start()

# Schedule job to run every hour (adjust interval as needed)
scheduler.add_job(
    func=auto_fetch_and_process_images,
    trigger=IntervalTrigger(minutes=60),
    id='fetch_images_job',
    name='Fetch new MRI images every hour',
    replace_existing=True
)

# Shut down the scheduler when exiting the app
atexit.register(lambda: scheduler.shutdown())

# Routes
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/fetch-images", methods=["POST"])
def fetch_images():
    images = fetch_images_from_azure()
    if isinstance(images, str):
        return jsonify({"error": images}), 500
    
    results = []
    processed_count = 0
    
    for image_name, image_data in images:
        severity_rating, comment = parse_gemini_response(generate_response(model, generate_prompt(), image_data))
        image_data.seek(0)
        image_base64 = base64.b64encode(image_data.getvalue()).decode("utf-8")
        results.append({
            "image_name": image_name,
            "severity_rating": severity_rating,
            "comment": comment,
            "image_data": image_base64
        })
        processed_count += 1
    
    return jsonify({
        "results": results, 
        "processed_count": processed_count,
        "skipped_count": 0
    })

@app.route("/upload-image", methods=["POST"])
def upload_image():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    # Check if the file is an allowed type
    allowed_extensions = {'png', 'jpg', 'jpeg', 'gif'}
    if not ('.' in file.filename and file.filename.rsplit('.', 1)[1].lower() in allowed_extensions):
        return jsonify({"error": "File type not allowed. Please upload an image."}), 400
    
    # Process the uploaded image
    file_data = BytesIO(file.read())
    result = process_uploaded_image(file_data, file.filename)
    
    if "error" in result:
        return jsonify(result), 500
    
    return jsonify({"result": result})

@app.route("/save-history", methods=["POST"])
def save_history():
    data = request.json.get("results", request.json)
    result = save_results_to_supabase(data)
    return jsonify({"message": result})

@app.route("/view-history", methods=["GET"])
def view_history():
    history = fetch_history_from_supabase()
    return jsonify(history)  # Always return an array

@app.route("/scheduler-status", methods=["GET"])
def scheduler_status():
    jobs = []
    for job in scheduler.get_jobs():
        jobs.append({
            "id": job.id,
            "name": job.name,
            "next_run": job.next_run_time.strftime("%Y-%m-%d %H:%M:%S") if job.next_run_time else None
        })
    return jsonify({
        "status": "running" if scheduler.running else "stopped",
        "jobs": jobs
    })

@app.route("/trigger-job", methods=["POST"])
def trigger_job():
    try:
        logger.info("Manually triggering job")
        auto_fetch_and_process_images()
        return jsonify({"message": "Job triggered successfully"})
    except Exception as e:
        logger.error(f"Error triggering job: {str(e)}")
        return jsonify({"error": str(e)}), 500
        
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)
