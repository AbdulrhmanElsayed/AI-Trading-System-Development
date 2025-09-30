# AI Trading System Setup Script for Windows

Write-Host "🚀 Setting up AI Trading System..." -ForegroundColor Green

# Create virtual environment
Write-Host "📦 Creating virtual environment..." -ForegroundColor Yellow
python -m venv venv

# Activate virtual environment
Write-Host "🔄 Activating virtual environment..." -ForegroundColor Yellow
& "venv\Scripts\Activate.ps1"

# Upgrade pip
Write-Host "⬆️ Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip

# Install requirements
Write-Host "📚 Installing Python dependencies..." -ForegroundColor Yellow
pip install -r requirements.txt

# Create environment file from template
if (-not (Test-Path ".env")) {
    Write-Host "🔧 Creating .env file from template..." -ForegroundColor Yellow
    Copy-Item ".env.template" ".env"
    Write-Host "⚠️ Please edit .env file with your actual API keys and passwords!" -ForegroundColor Red
}

# Create necessary directories
Write-Host "📁 Creating directories..." -ForegroundColor Yellow
$directories = @("logs", "data\raw", "data\processed", "data\models", "notebooks", "sql", "monitoring")
foreach ($dir in $directories) {
    New-Item -Path $dir -ItemType Directory -Force | Out-Null
}

# Create .gitignore
Write-Host "🔒 Creating .gitignore..." -ForegroundColor Yellow
@"
# Environment and secrets
.env
*.log
logs/

# Python
__pycache__/
*.py[cod]
*`$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environment
venv/
env/
ENV/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Data
data/raw/*
data/processed/*
!data/raw/.gitkeep
!data/processed/.gitkeep

# Models
data/models/*
!data/models/.gitkeep

# Jupyter
.ipynb_checkpoints/

# Database
*.db
*.sqlite3

# Docker
docker-compose.override.yml
"@ | Out-File -FilePath ".gitignore" -Encoding utf8

# Create placeholder files
"" | Out-File -FilePath "data\raw\.gitkeep" -Encoding utf8
"" | Out-File -FilePath "data\processed\.gitkeep" -Encoding utf8
"" | Out-File -FilePath "data\models\.gitkeep" -Encoding utf8

Write-Host "✅ Setup complete!" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "1. Edit .env file with your API keys and configuration"
Write-Host "2. Run 'docker-compose up -d' to start the infrastructure"
Write-Host "3. Run 'python src/main.py' to start the trading system"
Write-Host "4. Access Grafana at http://localhost:3000 (admin/grafana123)"
Write-Host "5. Access Jupyter at http://localhost:8888 (token: trading123)"
Write-Host ""
Write-Host "For paper trading, ensure TRADING_MODE=paper in your .env file" -ForegroundColor Yellow
Write-Host "⚠️ WARNING: Only use live trading after thorough testing!" -ForegroundColor Red