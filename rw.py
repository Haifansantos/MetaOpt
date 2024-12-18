from google.cloud import storage

def verify_gcs_access():
    try:
        client = storage.Client()
        buckets = list(client.list_buckets())
        print("Available buckets:", [bucket.name for bucket in buckets])
    except Exception as e:
        print("Error accessing Google Cloud Storage:", e)

# Jalankan verifikasi
verify_gcs_access()