# HR Management System - Docker Setup

## Quick Start

### Prerequisites
- Docker Desktop installed
- Docker Compose installed
- Existing database connection (Supabase/PostgreSQL)

### Setup and Run

1. **Copy environment file:**
   ```bash
   copy .env.example .env
   ```

2. **Edit .env file with your OpenAI API key:**
   ```
   OPENAI_API_KEY=your_actual_openai_api_key_here
   ```

3. **Start the application:**
   ```bash
   docker-compose up -d
   ```

4. **Access the application:**
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:8000
   - API Documentation: http://localhost:8000/docs

## Services

### 🌐 Frontend (React + Vite)

- URL: http://localhost:3000 (port 3000)
- Built with Nginx for production

### 🚀 Backend (FastAPI)
- API: http://localhost:8000
- Documentation: http://localhost:8000/docs
- **Note:** Uses your existing database connection (configured in backend/database.py)

## Database Configuration

This setup uses your **existing database** (Supabase PostgreSQL) instead of creating a new one in Docker. 

The database connection is configured in:
- `backend/database.py` - Contains the database URL and connection logic
- `HR_DB_test.ipynb` - Contains database testing and exploration code

## Useful Commands

```bash
# Start services (backend + frontend only)
docker-compose up -d

# Stop services
docker-compose down

# View logs
docker-compose logs -f

# Rebuild services
docker-compose build

# View backend logs specifically
docker-compose logs backend

# View frontend logs specifically  
docker-compose logs frontend
```

## File Structure
```
├── docker-compose.yml          # Main Docker configuration
├── .env.example               # Environment template
├── .dockerignore              # Docker ignore rules
├── backend/
│   ├── Dockerfile             # Backend Docker image
│   ├── init.sql              # Database initialization
│   └── requirements.txt       # Python dependencies
└── frontend/
    ├── Dockerfile             # Frontend Docker image
    └── package.json           # Node.js dependencies
```

## Troubleshooting

### Port Issues
If port 80 is already in use, you can change the frontend port:
```bash
# Edit docker-compose.yml and change:
ports:
  - "8080:80"  # Use port 8080 instead
```

### Database Issues
```bash
# Reset database
docker-compose down -v
docker-compose up -d
```

### Check Service Status
```bash
docker-compose ps
```

That's it! Your HR Management System is ready to use with Docker. 🚀
