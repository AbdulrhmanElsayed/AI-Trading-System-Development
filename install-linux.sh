#!/bin/bash

# AI Trading System - Linux Installation Script
# This script installs all required tools and dependencies for running the AI Trading System on Linux

set -e  # Exit on any error

# Color codes for better output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING:${NC} $1"
}

log_error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR:${NC} $1"
}

log_info() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] INFO:${NC} $1"
}

# Check if running as root
check_root() {
    if [[ $EUID -eq 0 ]]; then
        log_error "This script should not be run as root for security reasons"
        log_info "Please run as a regular user with sudo privileges"
        exit 1
    fi
}

# Detect Linux distribution
detect_distro() {
    if [[ -f /etc/os-release ]]; then
        . /etc/os-release
        DISTRO=$ID
        VERSION=$VERSION_ID
    else
        log_error "Cannot detect Linux distribution"
        exit 1
    fi
    log "Detected OS: $PRETTY_NAME"
}

# Update system packages
update_system() {
    log "Updating system packages..."
    case $DISTRO in
        ubuntu|debian)
            sudo apt update && sudo apt upgrade -y
            ;;
        centos|rhel|rocky|almalinux)
            sudo yum update -y || sudo dnf update -y
            ;;
        fedora)
            sudo dnf update -y
            ;;
        arch|manjaro)
            sudo pacman -Syu --noconfirm
            ;;
        *)
            log_warning "Unknown distribution. Please update manually."
            ;;
    esac
}

# Install basic system tools
install_system_tools() {
    log "Installing basic system tools..."
    case $DISTRO in
        ubuntu|debian)
            sudo apt install -y \
                curl \
                wget \
                git \
                vim \
                htop \
                tree \
                unzip \
                build-essential \
                software-properties-common \
                apt-transport-https \
                ca-certificates \
                gnupg \
                lsb-release \
                jq \
                net-tools \
                telnet \
                nmap \
                iotop \
                iftop
            ;;
        centos|rhel|rocky|almalinux)
            sudo yum groupinstall -y "Development Tools" || sudo dnf groupinstall -y "Development Tools"
            sudo yum install -y \
                curl \
                wget \
                git \
                vim \
                htop \
                tree \
                unzip \
                jq \
                net-tools \
                telnet \
                nmap \
                iotop \
                iftop || \
            sudo dnf install -y \
                curl \
                wget \
                git \
                vim \
                htop \
                tree \
                unzip \
                jq \
                net-tools \
                telnet \
                nmap \
                iotop \
                iftop
            ;;
        fedora)
            sudo dnf groupinstall -y "Development Tools"
            sudo dnf install -y \
                curl \
                wget \
                git \
                vim \
                htop \
                tree \
                unzip \
                jq \
                net-tools \
                telnet \
                nmap \
                iotop \
                iftop
            ;;
        arch|manjaro)
            sudo pacman -S --noconfirm \
                curl \
                wget \
                git \
                vim \
                htop \
                tree \
                unzip \
                base-devel \
                jq \
                net-tools \
                nmap \
                iotop \
                iftop
            ;;
    esac
}

# Install Python 3.11
install_python() {
    log "Installing Python 3.11..."
    
    if command -v python3.11 &> /dev/null; then
        log_info "Python 3.11 already installed"
        return 0
    fi
    
    case $DISTRO in
        ubuntu|debian)
            # Add deadsnakes PPA for Python 3.11
            sudo add-apt-repository -y ppa:deadsnakes/ppa
            sudo apt update
            sudo apt install -y \
                python3.11 \
                python3.11-dev \
                python3.11-venv \
                python3-pip \
                python3.11-distutils
            ;;
        centos|rhel|rocky|almalinux)
            # Install Python 3.11 from source or EPEL
            sudo yum install -y python3 python3-pip python3-devel || \
            sudo dnf install -y python3 python3-pip python3-devel
            ;;
        fedora)
            sudo dnf install -y python3 python3-pip python3-devel
            ;;
        arch|manjaro)
            sudo pacman -S --noconfirm python python-pip
            ;;
    esac
    
    # Set Python 3.11 as default python3 if available
    if command -v python3.11 &> /dev/null; then
        sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1
    fi
    
    # Upgrade pip (handle externally-managed environment)
    if ! python3 -m pip install --upgrade pip 2>/dev/null; then
        log_warning "Pip upgrade failed due to externally-managed environment, continuing..."
    fi
    
    log "Python version: $(python3 --version)"
}

# Fix apt_pkg module compatibility issues
fix_apt_pkg() {
    log "Fixing apt_pkg module compatibility..."
    
    case $DISTRO in
        ubuntu|debian)
            # Check if apt_pkg import fails
            if ! python3 -c "import apt_pkg" 2>/dev/null; then
                log_warning "apt_pkg module not compatible with current Python version"
                
                # Check available Python versions
                if command -v python3.12 &> /dev/null; then
                    log_info "Python 3.12 detected, updating alternatives..."
                    
                    # Remove and reinstall python3-apt to ensure compatibility
                    sudo apt remove --purge -y python3-apt 2>/dev/null || true
                    sudo apt install -y python3-apt python-apt-common
                    
                    # Set up alternatives with Python 3.12 having higher priority
                    sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 2>/dev/null || true
                    sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 2 2>/dev/null || true
                    
                    # Verify the fix
                    if python3 -c "import apt_pkg" 2>/dev/null; then
                        log "✅ apt_pkg module fixed successfully"
                    else
                        log_warning "⚠️ apt_pkg module still has issues but continuing..."
                    fi
                else
                    log_info "Attempting to reinstall python3-apt..."
                    sudo apt remove --purge -y python3-apt 2>/dev/null || true
                    sudo apt install -y python3-apt python-apt-common
                fi
            else
                log_info "apt_pkg module working correctly"
            fi
            ;;
        *)
            log_info "apt_pkg fix only needed for Ubuntu/Debian systems"
            ;;
    esac
}

# Install Node.js and npm
install_nodejs() {
    log "Installing Node.js 18 LTS..."
    
    if command -v node &> /dev/null && command -v npm &> /dev/null && [[ $(node -v | cut -d'.' -f1 | tr -d 'v') -ge 18 ]]; then
        log_info "Node.js 18+ and npm already installed"
        return 0
    fi
    
    case $DISTRO in
        ubuntu|debian)
            # Try NodeSource repository first
            if curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -; then
                sudo apt update
                sudo apt install -y nodejs
            else
                log_warning "NodeSource repository failed, trying snap installation..."
                sudo apt install -y snapd
                sudo snap install node --classic
            fi
            
            # If npm is still missing, install it separately
            if ! command -v npm &> /dev/null; then
                log_info "Installing npm separately..."
                sudo apt install -y npm || sudo snap install npm --classic
            fi
            ;;
        centos|rhel|rocky|almalinux)
            # Install Node.js repository
            curl -fsSL https://rpm.nodesource.com/setup_18.x | sudo bash -
            sudo yum install -y nodejs npm || sudo dnf install -y nodejs npm
            ;;
        fedora)
            # Install Node.js repository
            curl -fsSL https://rpm.nodesource.com/setup_18.x | sudo bash -
            sudo dnf install -y nodejs npm
            ;;
        arch|manjaro)
            sudo pacman -S --noconfirm nodejs npm
            ;;
    esac
    
    # Verify installation
    if command -v node &> /dev/null; then
        log "Node.js version: $(node --version)"
    else
        log_error "Node.js installation failed"
    fi
    
    if command -v npm &> /dev/null; then
        log "npm version: $(npm --version)"
    else
        log_error "npm installation failed"
    fi
}

# Install Docker
install_docker() {
    log "Installing Docker..."
    
    if command -v docker &> /dev/null; then
        log_info "Docker already installed"
        return 0
    fi
    
    case $DISTRO in
        ubuntu|debian)
            # Add Docker's official GPG key
            curl -fsSL https://download.docker.com/linux/$DISTRO/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
            
            # Add Docker repository
            echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/$DISTRO $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
            
            sudo apt update
            sudo apt install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
            ;;
        centos|rhel|rocky|almalinux)
            sudo yum install -y yum-utils || sudo dnf install -y dnf-utils
            sudo yum-config-manager --add-repo https://download.docker.com/linux/centos/docker-ce.repo || \
            sudo dnf config-manager --add-repo https://download.docker.com/linux/centos/docker-ce.repo
            sudo yum install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin || \
            sudo dnf install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
            ;;
        fedora)
            sudo dnf config-manager --add-repo https://download.docker.com/linux/fedora/docker-ce.repo
            sudo dnf install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
            ;;
        arch|manjaro)
            sudo pacman -S --noconfirm docker docker-compose
            ;;
    esac
    
    # Add user to docker group
    sudo usermod -aG docker $USER
    
    # Start and enable Docker
    sudo systemctl start docker
    sudo systemctl enable docker
    
    log "Docker version: $(docker --version)"
}

# Install Docker Compose (standalone)
install_docker_compose() {
    log "Installing Docker Compose standalone..."
    
    if command -v docker-compose &> /dev/null; then
        log_info "Docker Compose already installed"
        return 0
    fi
    
    # Install Docker Compose
    COMPOSE_VERSION=$(curl -s https://api.github.com/repos/docker/compose/releases/latest | jq -r '.tag_name')
    sudo curl -L "https://github.com/docker/compose/releases/download/${COMPOSE_VERSION}/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    sudo chmod +x /usr/local/bin/docker-compose
    
    # Create symlink for docker compose command
    sudo ln -sf /usr/local/bin/docker-compose /usr/bin/docker-compose
    
    log "Docker Compose version: $(docker-compose --version)"
}

# Install Kubernetes tools
install_kubernetes_tools() {
    log "Installing Kubernetes tools (kubectl, helm)..."
    
    # Install kubectl
    if ! command -v kubectl &> /dev/null; then
        curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
        sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl
        rm kubectl
    fi
    
    # Install Helm
    if ! command -v helm &> /dev/null; then
        curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash
    fi
    
    log "kubectl version: $(kubectl version --client --short 2>/dev/null || echo 'Client only')"
    log "Helm version: $(helm version --short 2>/dev/null || echo 'Not configured')"
}

# Install PostgreSQL client
install_postgresql_client() {
    log "Installing PostgreSQL client..."
    
    if command -v psql &> /dev/null; then
        log_info "PostgreSQL client already installed"
        return 0
    fi
    
    case $DISTRO in
        ubuntu|debian)
            sudo apt install -y postgresql-client
            ;;
        centos|rhel|rocky|almalinux)
            sudo yum install -y postgresql || sudo dnf install -y postgresql
            ;;
        fedora)
            sudo dnf install -y postgresql
            ;;
        arch|manjaro)
            sudo pacman -S --noconfirm postgresql
            ;;
    esac
}

# Install Redis client
install_redis_client() {
    log "Installing Redis client..."
    
    if command -v redis-cli &> /dev/null; then
        log_info "Redis client already installed"
        return 0
    fi
    
    case $DISTRO in
        ubuntu|debian)
            sudo apt install -y redis-tools
            ;;
        centos|rhel|rocky|almalinux)
            sudo yum install -y redis || sudo dnf install -y redis
            ;;
        fedora)
            sudo dnf install -y redis
            ;;
        arch|manjaro)
            sudo pacman -S --noconfirm redis
            ;;
    esac
}

# Install monitoring tools
install_monitoring_tools() {
    log "Installing monitoring and debugging tools..."
    
    case $DISTRO in
        ubuntu|debian)
            sudo apt install -y \
                htop \
                iotop \
                iftop \
                nethogs \
                ncdu \
                tree \
                stress \
                sysstat \
                dstat
            ;;
        centos|rhel|rocky|almalinux)
            sudo yum install -y \
                htop \
                iotop \
                iftop \
                nethogs \
                ncdu \
                tree \
                stress \
                sysstat \
                dstat || \
            sudo dnf install -y \
                htop \
                iotop \
                iftop \
                nethogs \
                ncdu \
                tree \
                stress \
                sysstat \
                dstat
            ;;
        fedora)
            sudo dnf install -y \
                htop \
                iotop \
                iftop \
                nethogs \
                ncdu \
                tree \
                stress \
                sysstat \
                dstat
            ;;
        arch|manjaro)
            sudo pacman -S --noconfirm \
                htop \
                iotop \
                iftop \
                nethogs \
                ncdu \
                tree \
                stress \
                sysstat \
                dstat
            ;;
    esac
}

# Install security tools
install_security_tools() {
    log "Installing security tools..."
    
    # Install fail2ban
    case $DISTRO in
        ubuntu|debian)
            sudo apt install -y fail2ban ufw
            ;;
        centos|rhel|rocky|almalinux)
            sudo yum install -y fail2ban firewalld || sudo dnf install -y fail2ban firewalld
            ;;
        fedora)
            sudo dnf install -y fail2ban firewalld
            ;;
        arch|manjaro)
            sudo pacman -S --noconfirm fail2ban ufw
            ;;
    esac
    
    # Configure basic firewall
    if command -v ufw &> /dev/null; then
        sudo ufw --force enable
        sudo ufw default deny incoming
        sudo ufw default allow outgoing
        sudo ufw allow ssh
        sudo ufw allow 8000/tcp  # Trading app
        sudo ufw allow 3000/tcp  # Grafana
        sudo ufw allow 9090/tcp  # Prometheus
    elif command -v firewalld &> /dev/null; then
        sudo systemctl start firewalld
        sudo systemctl enable firewalld
        sudo firewall-cmd --permanent --add-port=8000/tcp
        sudo firewall-cmd --permanent --add-port=3000/tcp
        sudo firewall-cmd --permanent --add-port=9090/tcp
        sudo firewall-cmd --reload
    fi
}

# Create trading user and setup environment
setup_trading_environment() {
    log "Setting up trading system environment..."
    
    # Create trading directory
    TRADING_DIR="/opt/trading-system"
    sudo mkdir -p $TRADING_DIR
    sudo chown $USER:$USER $TRADING_DIR
    
    # Create systemd service for trading system
    sudo tee /etc/systemd/system/trading-system.service > /dev/null <<EOF
[Unit]
Description=AI Trading System
After=network.target docker.service postgresql.service redis.service
Requires=docker.service

[Service]
Type=forking
User=$USER
Group=$USER
WorkingDirectory=$TRADING_DIR
ExecStart=/usr/local/bin/docker-compose up -d
ExecStop=/usr/local/bin/docker-compose down
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

    # Create log rotation
    sudo tee /etc/logrotate.d/trading-system > /dev/null <<EOF
$TRADING_DIR/logs/*.log {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    notifempty
    create 0644 $USER $USER
}
EOF

    log_info "Trading system environment configured at $TRADING_DIR"
}

# Optimize system for trading
optimize_system() {
    log "Optimizing system for low-latency trading..."
    
    # Network optimizations
    sudo tee /etc/sysctl.d/99-trading-optimization.conf > /dev/null <<EOF
# Network optimizations for trading
net.core.rmem_default = 262144
net.core.rmem_max = 16777216
net.core.wmem_default = 262144
net.core.wmem_max = 16777216
net.ipv4.tcp_rmem = 4096 87380 16777216
net.ipv4.tcp_wmem = 4096 65536 16777216
net.core.netdev_max_backlog = 5000
net.ipv4.tcp_congestion_control = bbr

# Memory optimizations
vm.swappiness = 1
vm.dirty_ratio = 15
vm.dirty_background_ratio = 5

# File handle limits
fs.file-max = 2097152
EOF

    # Apply sysctl settings
    sudo sysctl -p /etc/sysctl.d/99-trading-optimization.conf

    # Set resource limits
    sudo tee -a /etc/security/limits.conf > /dev/null <<EOF
# Trading system resource limits
$USER soft nofile 65536
$USER hard nofile 65536
$USER soft memlock unlimited
$USER hard memlock unlimited
EOF

    log_info "System optimized for trading workloads"
}

# Install additional Python packages
install_python_packages() {
    log "Installing essential Python packages..."
    
    # For externally-managed Python environments, use system packages where possible
    case $DISTRO in
        ubuntu|debian)
            # Install available system packages first
            sudo apt install -y \
                python3-venv \
                python3-pip \
                python3-virtualenv \
                python3-numpy \
                python3-pandas \
                python3-matplotlib \
                python3-seaborn \
                python3-requests \
                python3-psutil \
                python3-docker \
                python3-jupyter-core 2>/dev/null || true
            
            # Install pipx (try package first, then pip as fallback)
            if ! sudo apt install -y pipx 2>/dev/null; then
                log_info "Installing pipx via pip..."
                python3 -m pip install --user pipx 2>/dev/null || python3 -m pip install --break-system-packages --user pipx 2>/dev/null || true
                # Ensure pipx is in PATH
                export PATH="$HOME/.local/bin:$PATH"
            fi
            
            # Install packages that aren't available as system packages using pipx
            if command -v pipx &> /dev/null; then
                log_info "Installing additional packages using pipx..."
                pipx install poetry 2>/dev/null || log_warning "Failed to install poetry via pipx"
                pipx install jupyterlab 2>/dev/null || log_warning "Failed to install jupyterlab via pipx"
            else
                log_warning "pipx not available, some packages may not be installed"
            fi
            ;;
        *)
            # For other distributions, try pip with --user flag, fallback to --break-system-packages if needed
            if ! pip3 install --user \
                virtualenv \
                pipenv \
                poetry \
                jupyter \
                jupyterlab \
                numpy \
                pandas \
                matplotlib \
                seaborn \
                requests \
                psutil \
                docker 2>/dev/null; then
                
                log_warning "Standard pip install failed, trying with --break-system-packages..."
                pip3 install --break-system-packages --user \
                    virtualenv \
                    pipenv \
                    poetry \
                    jupyter \
                    jupyterlab \
                    numpy \
                    pandas \
                    matplotlib \
                    seaborn \
                    requests \
                    psutil \
                    docker 2>/dev/null || log_warning "Some Python packages may not be installed"
            fi
            ;;
    esac
    
    # Ensure user's local bin is in PATH
    if [[ ":$PATH:" != *":$HOME/.local/bin:"* ]]; then
        echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
        log_info "Added ~/.local/bin to PATH in ~/.bashrc"
    fi
}

# Verify installations
verify_installations() {
    log "Verifying installations..."
    
    local errors=0
    
    # Check commands
    local commands=("python3" "pip3" "node" "npm" "docker" "docker-compose" "kubectl" "helm" "psql" "redis-cli" "git" "curl" "wget")
    
    for cmd in "${commands[@]}"; do
        if command -v "$cmd" &> /dev/null; then
            log_info "✅ $cmd: $(which $cmd)"
        else
            log_error "❌ $cmd: Not found"
            ((errors++))
        fi
    done
    
    # Check Docker service
    if systemctl is-active --quiet docker; then
        log_info "✅ Docker service: Running"
    else
        log_error "❌ Docker service: Not running"
        ((errors++))
    fi
    
    # Check Python modules
    local modules=("numpy" "pandas" "requests" "psutil")
    for module in "${modules[@]}"; do
        if python3 -c "import $module" 2>/dev/null; then
            log_info "✅ Python module $module: Available"
        else
            log_warning "⚠️ Python module $module: Not available system-wide (may need virtual environment)"
        fi
    done
    
    # Check for Python development tools
    local dev_tools=("venv" "pip")
    for tool in "${dev_tools[@]}"; do
        if python3 -m $tool --help &>/dev/null; then
            log_info "✅ Python $tool module: Available"
        else
            log_warning "⚠️ Python $tool module: Not available"
        fi
    done
    
    # Check apt_pkg module (Ubuntu/Debian only)
    if [[ "$DISTRO" == "ubuntu" || "$DISTRO" == "debian" ]]; then
        if python3 -c "import apt_pkg" 2>/dev/null; then
            log_info "✅ apt_pkg module: Working correctly"
        else
            log_warning "⚠️ apt_pkg module: May have compatibility issues"
        fi
    fi
    
    return $errors
}

# Main installation function
main() {
    echo -e "${CYAN}"
    echo "████████╗██████╗  █████╗ ██████╗ ██╗███╗   ██╗ ██████╗ "
    echo "╚══██╔══╝██╔══██╗██╔══██╗██╔══██╗██║████╗  ██║██╔════╝ "
    echo "   ██║   ██████╔╝███████║██║  ██║██║██╔██╗ ██║██║  ███╗"
    echo "   ██║   ██╔══██╗██╔══██║██║  ██║██║██║╚██╗██║██║   ██║"
    echo "   ██║   ██║  ██║██║  ██║██████╔╝██║██║ ╚████║╚██████╔╝"
    echo "   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝╚═════╝ ╚═╝╚═╝  ╚═══╝ ╚═════╝ "
    echo ""
    echo "    AI Trading System - Linux Installation Script"
    echo -e "${NC}"
    echo ""
    
    check_root
    detect_distro
    
    log "Starting installation process..."
    log_warning "This script will install system packages and modify system configuration"
    log_warning "Press Ctrl+C within 10 seconds to cancel..."
    sleep 10
    
    # Installation steps
    update_system
    install_system_tools
    install_python
    fix_apt_pkg
    install_nodejs
    install_docker
    install_docker_compose
    install_kubernetes_tools
    install_postgresql_client
    install_redis_client
    install_monitoring_tools
    install_security_tools
    install_python_packages
    setup_trading_environment
    optimize_system
    
    # Verification
    log "Installation completed! Verifying..."
    if verify_installations; then
        echo ""
        echo -e "${GREEN}🎉 Installation completed successfully!${NC}"
        echo ""
        echo -e "${YELLOW}Important Notes:${NC}"
        echo "1. Log out and log back in to apply Docker group membership"
        echo "2. Reboot the system to apply kernel optimizations"
        echo "3. Clone the trading system repository to /opt/trading-system"
        echo "4. Configure environment variables in .env file"
        echo "5. Run 'docker-compose up -d' to start the system"
        echo ""
        echo -e "${BLUE}Quick Start Commands:${NC}"
        echo "cd /opt/trading-system"
        echo "git clone <your-repo> ."
        echo "cp .env.example .env"
        echo "# Edit .env with your configuration"
        echo ""
        echo "# Create Python virtual environment for the project:"
        echo "python3 -m venv venv"
        echo "source venv/bin/activate"
        echo "pip install -r requirements.txt"
        echo ""
        echo "# Start the system:"
        echo "docker-compose up -d"
        echo ""
        echo -e "${PURPLE}Monitoring URLs (after starting):${NC}"
        echo "• Trading System: http://localhost:8000"
        echo "• Grafana: http://localhost:3000 (admin/admin123)"
        echo "• Prometheus: http://localhost:9090"
        echo "• Jupyter Lab: http://localhost:8888 (token: trading123)"
        echo ""
        log_warning "Remember to configure firewall rules for production use!"
    else
        log_error "Some installations failed. Please check the errors above."
        exit 1
    fi
}

# Run main function
main "$@"