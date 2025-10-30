# Hunyuan3D-2.1 Docker Backend API

Simple Docker setup to run the model backend that your frontend can call.

## Start the Backend

```bash
docker-compose up -d --build
```

That's it! First build takes ~60 mins to download models.

## API

**Backend URL:** `http://localhost:8081`  
**API Docs:** http://localhost:8081/docs

### Endpoints

| Endpoint | Method | What it does |
|----------|--------|--------------|
| `/health` | GET | Check if running |
| `/send` | POST | Submit image (returns uid) |
| `/status/{uid}` | GET | Get result |

### Request Format

```json
POST /send
{
  "image": "base64_encoded_image",
  "remove_background": true,
  "texture": true
}
```

### Response

```json
// Step 1: Get UID
{ "uid": "abc-123" }

// Step 2: Poll /status/{uid}
{ "status": "processing" }
{ "status": "texturing" }
{ "status": "completed", "model_base64": "..." }
```

## Frontend Code (JavaScript)

```javascript
// 1. Submit image
const { uid } = await fetch('http://localhost:8081/send', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ 
    image: base64Image,
    remove_background: true,
    texture: true 
  })
}).then(r => r.json());

// 2. Poll for result (every 5 seconds)
const checkStatus = async () => {
  const data = await fetch(`http://localhost:8081/status/${uid}`)
    .then(r => r.json());
  
  if (data.status === 'completed') {
    // Decode base64 to GLB file
    const bytes = Uint8Array.from(atob(data.model_base64), c => c.charCodeAt(0));
    const blob = new Blob([bytes], { type: 'model/gltf-binary' });
    const url = URL.createObjectURL(blob);
    // Use the url to display/download 3D model
  } else {
    setTimeout(checkStatus, 5000); // Check again in 5s
  }
};
checkStatus();
```

## What You Get

The API returns a **textured GLB file** with PBR materials (albedo, metallic, roughness).

**Generation time:** 30-60 seconds

## Commands

```bash
# Start
docker-compose up -d --build

# View logs
docker-compose logs -f

# Stop
docker-compose down

# Health check
curl http://localhost:8081/health
```

## Requirements

- Docker + Docker Compose
- NVIDIA GPU with 21GB+ VRAM
- NVIDIA Container Toolkit

## Files

- `Dockerfile` - Container setup
- `docker-compose.yml` - Service config
- `.dockerignore` - Build optimization

