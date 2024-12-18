import os
import matplotlib
matplotlib.use('Agg')  # Menggunakan backend non-interaktif
import pandas as pd
from flask import Flask, render_template, request, send_file
from metaopt_pipeline import MetaheuristicPipeline
from google.cloud import storage
import io
import pickle

app = Flask(__name__)

# Instantiate the pipeline
pipeline = MetaheuristicPipeline(filepath=None, task_type=None, label=None)

# Function to check if running in cloud environment
def is_cloud_environment():
    try:
        import google.auth
        _, project = google.auth.default()
        return bool(project)
    except Exception:
        return False

is_cloud_env = is_cloud_environment()
print(f"Is Cloud Environment: {is_cloud_env}")

# Set your bucket names
bucket_name = "save-plot-eda"  # Replace with your actual bucket name
bucket_name_model = "best_model_ml"

def clear_bucket_folder(bucket_name, folder_prefix=None):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blobs = list(bucket.list_blobs(prefix=folder_prefix))
    if not blobs:
        prefix_info = f"'{folder_prefix}'" if folder_prefix else "the bucket"
        print(f"No files found in {prefix_info} to delete.")
        return
    for blob in blobs:
        print(f"Deleting file: {blob.name}")
        blob.delete()
    print(f"All files in '{folder_prefix or 'the bucket'}' have been deleted.")

def fetch_images_from_bucket(bucket_name, folder_prefix):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blobs = list(bucket.list_blobs(prefix=folder_prefix))
    image_urls = [blob.public_url for blob in blobs if blob.name.endswith(('.png', '.jpg', '.jpeg'))]
    return image_urls

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if not file:
            return "No file selected", 400
        
        try:
            data = pd.read_csv(file)
            print("CSV file loaded successfully.")
        except Exception as e:
            return f"Error reading CSV file: {e}", 400

        task_type = request.form.get('task_type')
        label = request.form.get('label')

        # Handle label as optional
        if label and label != 'None' and label in data.columns:
            y = data[label]
            X = data.drop(columns=[label])
        else:
            y = None
            X = data.copy()
            if task_type in ['regression', 'classification'] and label == 'None':
                print("Warning: No label selected for regression/classification task. Proceeding without a label.")
        
        if y is not None:
            data = pd.concat([X, y], axis=1)

        # Prepare data overview
        data_overview_html = data.head(10).to_html(classes="data-overview-table", index=False)

        # Clear previous files from buckets
        try:
            clear_bucket_folder(bucket_name, folder_prefix="eda_plots/")
            clear_bucket_folder(bucket_name, folder_prefix="feature_importance_plots/")
            clear_bucket_folder(bucket_name_model)
        except Exception as e:
            return f"Error during clearing bucket folders: {e}", 500
        #Preprocessing
        try:
            processed_data = pipeline.preprocessing(data=data, task_type=task_type, label=label)
        except Exception as e:
            return f"Error during preprocessing: {e}", 500
        #EDA Visualization
        try:
            pipeline.eda_visualization(
                data=processed_data, 
                task_type=task_type, 
                is_cloud_env=is_cloud_env, 
                label=label, 
                bucket_name=bucket_name
            )
        except Exception as e:
            return f"Error during EDA visualization: {e}", 500

        ant_colony = None
        genetic_algorithm = None
        particle_swarm = None
        simulated_annealing = None

        try:
            result_df, best_ml_model, data_dict = pipeline.evaluate(
                processed_data, task_type, bucket_name_model, ant_colony, genetic_algorithm, particle_swarm, simulated_annealing
            )
            print("Model evaluation completed.")
            
            # Save best model to bucket
            if best_ml_model:
                model_filename = 'best_model_ml/best_model.pkl'
                blob = storage.Client().bucket(bucket_name_model).blob(model_filename)
                with io.BytesIO() as model_file:
                    pickle.dump(best_ml_model, model_file)
                    model_file.seek(0)
                    blob.upload_from_file(model_file)   
                print(f"Model successfully uploaded to bucket as {model_filename}")
            
            best_model_params = str(best_ml_model)
        except Exception as e:
            return f"Error during model evaluation: {e}", 500

        try:
            pipeline.final_visualization(result_df, task_type, data_dict, is_cloud_env, bucket_name)
        except Exception as e:
            return f"Error during final visualization: {e}", 500
            
        # Fetch best model pickle file from bucket
        try:
            folder_prefix_model = "best_model_ml/"
            blob_model = storage.Client().bucket(bucket_name_model).list_blobs(prefix=folder_prefix_model)
            model_files = [blob.name for blob in blob_model if blob.name.endswith(".pkl")]
            download_link = model_files[0] if model_files else None
        except Exception as e:
            download_link = None

        eda_images = fetch_images_from_bucket(bucket_name, folder_prefix="eda_plots/eda_plots/")
        feature_importance_images = fetch_images_from_bucket(bucket_name, folder_prefix="eda_plots/feature_importance_plots/")


        # Tambahkan kelas CSS "metrics-table" untuk tabel Model Metrics
        metrics_html = result_df.drop(result_df.columns[[1, 2]], axis=1).to_html(classes="metrics-table", index=False)

        return render_template(
            'result.html',
            result_html=best_model_params,
            metrics=metrics_html,
            eda_images=eda_images,
            feature_importance_images=feature_importance_images,
            data_overview=data_overview_html,
            model_download_link=download_link
        )
    return render_template('index.html')


@app.route('/download/<path:file_path>')
def download_file(file_path):
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name_model)
        blob = bucket.blob(file_path)
        file_stream = io.BytesIO()
        blob.download_to_file(file_stream)
        file_stream.seek(0)
        return send_file(file_stream, as_attachment=True, download_name=file_path.split('/')[-1])
    except Exception as e:
        return f"Error downloading file: {e}", 500

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=8080)
