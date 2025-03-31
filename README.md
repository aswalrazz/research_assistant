# Research Assistant

A powerful research paper analysis tool that helps researchers analyze academic papers, extract key information, and generate insights using Google's Gemini AI. The application supports both English and Arabic interfaces and provides bibliometric analysis capabilities.

## Features

- Research paper analysis using Gemini AI
- PDF document processing and analysis
- Bilingual interface (English/Arabic)
- Bibliometric analysis
- Data visualization
- Research paper archiving
- Interactive query system for paper analysis

## Prerequisites

- Docker installed on your system
- Google Gemini API key
- Internet connection for API access

## Environment Setup

### Development Environment

1. Create a `.env` file in the project root with your API key:
```
GEMINI_API_KEY=your_actual_api_key_here
```

### Production Environment

#### Option 1: Using Docker Secrets (Recommended for Production)

1. Create a Docker secret:
```bash
echo "your_api_key" | docker secret create gemini_api_key -
```

2. Run the container with the secret:
```bash
docker run -p 8504:8501 --secret gemini_api_key research-assistant:latest
```

#### Option 2: Using Environment Variables

1. Set the environment variable directly:
```bash
docker run -p 8504:8501 -e GEMINI_API_KEY=your_api_key research-assistant:latest
```

#### Option 3: Using .env File (Development Only)

1. Create `.env` file with your API key:
```
GEMINI_API_KEY=your_api_key_here
```

2. Run with the .env file:
```bash
docker run -p 8504:8501 --env-file .env research-assistant:latest
```

## Running with Docker

### Building the Docker Image

```bash
# Build the Docker image
docker build -t research-assistant .
```

### Running the Container

#### Basic Run (Development)
```bash
docker run -p 8504:8501 research-assistant:latest
```

#### Run with Environment Variables
```bash
docker run -it --rm -p 8504:8501 --env-file .env research-assistant
```

#### Run in Background
```bash
docker run -d -p 8504:8501 --env-file .env research-assistant
```

#### Run with Auto-restart
```bash
docker run -d --restart unless-stopped -p 8504:8501 --env-file .env research-assistant
```

### Port Configuration

- Internal container port: 8501
- External access port: 8504 (configurable)
- Access the application at: http://localhost:8504

### Docker Commands Reference

```bash
# Stop all running containers
docker stop $(docker ps -q)

# Remove all stopped containers
docker rm $(docker ps -a -q)

# View container logs
docker logs $(docker ps -q)

# List running containers
docker ps

# List all containers (including stopped)
docker ps -a
```

### Troubleshooting

1. **Port Conflict**
   - If port 8504 is already in use, you can map to a different port:
   ```bash
   docker run -it --rm -p 8505:8501 --env-file .env research-assistant
   ```

2. **Container Won't Start**
   - Check if the .env file exists and contains the API key
   - Verify Docker is running
   - Check container logs for errors:
   ```bash
   docker logs $(docker ps -a -q)
   ```

3. **Application Not Accessible**
   - Ensure the container is running: `docker ps`
   - Try accessing with different URLs:
     - http://localhost:8504
     - http://127.0.0.1:8504
   - Check if your firewall is blocking the port

## Project Structure

```
research_assistant/
├── research_assistant.py    # Main application file
├── Dockerfile              # Docker configuration
├── requirements.txt        # Python dependencies
├── .env                   # Environment variables
├── utilities/             # Utility functions
├── research_paper_analyst/ # Paper analysis module
└── gemini_interface/      # Gemini AI interface
```

## Dependencies

- streamlit==1.30.0
- google-generativeai==0.4.0
- requests==2.31.0
- pandas==2.2.0
- plotly==5.20.0
- python-dotenv==1.0.1

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 